from __future__ import annotations

__all__ = [
    "ListDataset", "ParquetDataset", "FlatParquetDataset",
    "FlatArrowRowGroupParquetDataset", "FlatRowgroupParquetDataset",
    "WeightedFlatRowgroupParquetDataset",
]

from collections.abc import Iterable, Mapping, Collection
from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import Any, Callable, Sequence
from columnflow.columnar_util import (
    get_ak_routes, Route, remove_ak_column, EMPTY_FLOAT, EMPTY_INT,
    flat_np_view, set_ak_column,
)
import law
logger = law.logger.get_logger(__name__)

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")
ak = maybe_import("awkward")
pa = maybe_import("pyarrow")
pq = maybe_import("pyarrow.parquet")

ListDataset = MockModule("ListDataset")  # type: ignore
ParquetDataset = MockModule("ParquetDataset")  # type: ignore
FlatParquetDataset = MockModule("FlatParquetDataset")  # type: ignore

if not isinstance(torchdata, MockModule):
    from torch.utils.data import Dataset
    from typing import Literal
    import re

    class ListDataset(Dataset):  # noqa: F811

        def __init__(self, len: int, prefix: str = "data"):
            self.len = len
            self.prefix = prefix
            self.data = [f"{self.prefix}_{i}" for i in range(self.len)]
            self.weights = np.linspace(0.1, 1.0, self.len)

        def __len__(self):
            return self.len

        def __getitem__(self, i: int) -> tuple[str, float]:
            return (self.data[i], self.weights[i])

        def to_list(self) -> list[str]:
            return list(zip(self.data, self.weights))

    class ParquetDataset(Dataset):  # noqa: F811
        def __init__(
            self,
            input: Sequence[str] | ak.Array,
            columns: Sequence[str] | None = None,
            target: str | int | Iterable[str | int] | None = None,
            open_options: dict[str, Any] | None = None,
            batch_transform: Callable | None = None,
            global_transform: Callable | None = None,
            categorical_target_transform: Callable | None = None,
            data_type_transform: Callable | None = None,
            device: str | None = None,
        ):
            self.open_options = open_options or {}
            self.columns = columns or set()
            # container for target columns
            self.target_columns = set()
            # container for integer targets
            self.int_targets = set()
            self.batch_transform = batch_transform
            self.categorical_target_transform = categorical_target_transform
            self.data_type_transform = data_type_transform
            self.global_transform = global_transform

            self.input = input
            self._data: ak.Array | None = None
            self._input_data: ak.Array | None = None
            self._target_data: ak.Array | None = None
            self.class_target: int

            self._resolved_trafo_inputs: set[Route] = set()
            self._resolved_trafo_outputs: set[Route] = set()

            # container for meta data of parquet file(s)
            # None if input is an ak.Array
            self.meta_data = None
            self.all_columns = set()

            if isinstance(input, (str, list)):
                if all(isinstance(x, str) for x in input):
                    self.path = input
                    # idea: write sampler that sub samples each partition individually
                    # the __getitem(s)__ method should then check which partition
                    # is currently read, open the corresponding partition with
                    # line below, and return the requested item(s).
                    # If a new partition is requested, close/delete the current array
                    # and load the next one.
                    # Would require reading the parquet file multiple times after
                    # each reset call (= overhead?), but would limit the memory consumption
                    self.meta_data = DotDict.wrap(ak.metadata_from_parquet(self.path))
                elif all(isinstance(x, ParquetDataset) for x in input):
                    # initialize from inputs
                    self._init_from_datasets(input)
                else:
                    raise ValueError(f"List-type nputs must be contain string or ParquetDataset, received {input}")
            elif isinstance(input, ak.Array):
                self._data = input

            # if the inputs are not of type ParquetDataset, there won't be any columns
            # parsed. In this case, do the logic to resolve them
            if not self.all_columns:
                self._parse_columns()
                self._parse_target(target=target)

                self._validate()

                if len(self.int_targets) > 0:
                    self.class_target = list(self.int_targets)[0]
                self.data_columns = self._extract_data_columns()

                # parse all strings to Route objects
                self.data_columns: set[Route] = set(Route(x) for x in self.data_columns)
                self.target_columns: set[Route] = set(Route(x) for x in self.target_columns)
                self.all_columns: set[Route] = set(Route(x) for x in self.all_columns)

        def _init_from_datasets(self, datasets: Sequence[ParquetDataset]) -> None:
            # take most information from first dataset in the list
            first = datasets[0]
            self.batch_transform = first.batch_transform
            self.categorical_target_transform = first.categorical_target_transform
            self.data_type_transform = first.data_type_transform
            self.global_transform = first.global_transform
            # make sure all datasets have the same columns
            def _get_columnset(attr):
                foo = getattr(first, attr)
                if not all(foo == getattr(x, attr) for x in datasets):
                    raise ValueError(f"All datasets must have the same columns in '{attr}'")
                return foo

            # get the already resolved columns from the input ParquetDatasets
            self.all_columns = _get_columnset("all_columns")
            self.target_columns = _get_columnset("target_columns")
            self.data_columns = _get_columnset("data_columns")
            self._resolved_trafo_inputs = _get_columnset("_resolved_trafo_inputs")
            self._resolved_trafo_outputs = _get_columnset("_resolved_trafo_outputs")

            # after this point, all columns must be the same, so it should be
            # safe to merge the underlying data
            self._data = ak.concatenate([x.data for x in datasets], axis=0)
            # datasets can have different categorical targets
            # we can circumvent the issues by transforming this into a
            # column first and then add this to the target column
            for x in datasets:
                x.data_type_transform = None

            if all(len(x.int_targets) > 0 for x in datasets):
                int_target_data = ak.concatenate([x._create_class_target(len(x.data)) for x in datasets], axis=0)
                self._data = set_ak_column(self._data, "categorical_target", int_target_data)
                self.target_columns.add(Route("categorical_target"))
            elif any(len(x.int_targets) > 0 for x in datasets):
                raise ValueError("Cannot merge datasets: some but not all datasets define integer classes!")

        def _extract_data_columns(self) -> set[Route]:

            return_columns = self.all_columns.symmetric_difference(self.target_columns)
            # add requested columns if they match columns that are produced by transformations
            # if there aren't any specific columns that are requested
            # assume that all columns produced by transformations are needed
            return_columns |= self._resolved_trafo_outputs

            # also clean up the input columns from the data columns
            return_columns -= self._resolved_trafo_inputs
            return return_columns

        def _parse_columns(self) -> None:
            if self.columns:
                self.columns = set(Route(x) for x in self.columns)
            elif isinstance(self._data, ak.Array):
                self.all_columns = set(x for x in get_ak_routes(self.data))

            # check if transformations define columns to use and
            # make sure that the needed inputs are loaded, a.k.a. add them to all_columns
            self.all_columns |= self._extract_transform_columns()
            trafo_inputs = self._extract_transform_columns()
            if self.meta_data:
                self.all_columns = set(x.replace(".list.item", "") for x in self.meta_data.columns)
                # columns are not explicitely considered when loading the meta data
                tmp_cols = set()
                # so filter the full set of columns accordingly
                cols_to_check = set((self.columns or {".*"}))
                # also take the inputs from transformations into account
                for x in self.all_columns:
                    for col in cols_to_check:
                        resolved_route = self._check_against_pattern(x, col)
                        if resolved_route:
                            tmp_cols.add(resolved_route)
                    # also check and resolve the inputs for tranformations
                    for col in trafo_inputs:
                        resolved_route = self._check_against_pattern(x, col)
                        if resolved_route:
                            tmp_cols.add(resolved_route)
                            self._resolved_trafo_inputs.add(resolved_route)

                self.all_columns = tmp_cols

            # check if the output of transformations is requested
            # and remove any from the columns to load
            tranform_outputs = self._extract_transform_columns(attr="produces")
            # compare element-wise
            all_columns = self.all_columns.copy()
            for col in all_columns:
                if any(self._check_against_pattern(col.string_column, x) for x in tranform_outputs):
                    self.all_columns.remove(col)
                    self._resolved_trafo_outputs.add(col)

            # make sure that columns that do not exist in the data but
            # that are still requested in columns are also resolved properly
            if self.columns:
                for col in self.columns:
                    if any(self._check_against_pattern(col.string_column, x) for x in tranform_outputs):
                        self._resolved_trafo_outputs.add(col)

            if len(self.all_columns) == 0:
                raise ValueError("No columns specified and no metadata found")

        def _check_against_pattern(self, target: str, col: Route | str) -> Route | None:
            slice_dict: dict[int, tuple[slice]] = {}
            str_col: str = str(col)

            if isinstance(col, Route):
                slice_dict = {
                    i: field for i, field in enumerate(col.fields) if isinstance(field, tuple)
                }
                str_col = col.string_column

            # make sure there aren't any special characters that aren't caught
            str_col = str_col.replace("{", "(").replace("}", ")").replace(",", "|")
            pattern = re.compile(f"^{str_col}$")

            try:
                if not pattern.match(target):
                    return
            except Exception as e:
                from IPython import embed
                embed(header=f"raised Exception '{e}', {target=}, {col=}")
            # if there is a match, insert possible slices
            # from IPython import embed
            # embed(header=f"found match for target '{target}' and pattern '{col}'")
            parts = target.split(".")
            for index in reversed(slice_dict.keys()):
                parts.insert(index, slice_dict[index])

            return Route(Route.join(parts))

        def _extract_transform_columns(self, attr: Literal["uses", "produces"] = "uses") -> set[Route]:
            """
            Small function to extract columns from transformations

            :param attr: attribute to extract from transformations, either "uses" or "produces"
            :returns: Set with resolved Routes to columns in awkward array (braces are expanded)
            """
            transform_inputs: set[Route] = set()
            for t in [
                self.batch_transform,
                self.global_transform,
                self.data_type_transform,
                self.categorical_target_transform,
            ]:
                transform_inputs.update(
                    *list(map(Route, law.util.brace_expand(obj)) for obj in getattr(t, attr, [])),
                )
            return transform_inputs

        def _parse_target(self, target: str | int | Iterable[str | int]) -> None:
            # if the target is not a list, cast it
            def _add_target(target):
                if isinstance(target, str):
                    # target might be regex, so resolve it against all columns
                    self.target_columns.update(
                        x for x in self.all_columns if self._check_against_pattern(x, target)
                    )
                elif isinstance(target, (int, float)):
                    self.int_targets.add(int(target))
                else:
                    raise ValueError(f"Target must be string or int, received {target=}")

            if target is not None and not isinstance(target, Iterable):
                _add_target(target)
            elif target:
                for t in target:
                    _add_target(t)

        def _validate(self) -> None:
            if self.columns and not isinstance(self.columns, Iterable):
                raise ValueError(f"columns must be an iterable of strings, received {self.columns}")
            # sanity checks for targets

            for target in self.target_columns:
                # if target is a string and specific columns are supposed to be
                # loaded, check whether the target is also in the columns

                # targets can also be produced by a transformation, so first collect all
                # columns in one super set
                full_column_set = self.all_columns | self._extract_transform_columns(attr="produces")

                if not any(self._check_against_pattern(str(target), col) for col in full_column_set):
                    raise ValueError(f"target {target} not found in columns {full_column_set=}")

            # if target is an integer, this is a class index
            # this should be >= 0
            if any(target < 0 for target in self.int_targets):
                raise ValueError(f"int targets must be >= 0, received {self.int_targets}")
            if len(self.int_targets) > 1:
                raise ValueError(
                    "There cannot be more than one categorical target per dataset"
                    f", received {self.int_targets}",
                )

        @property
        def data(self) -> ak.Array:
            if self._data is None:
                self.open_options["columns"] = [x.string_column for x in self.all_columns]
                self._data = ak.from_parquet(self.path, **self.open_options)

                if self.global_transform:
                    self._data = self.global_transform(self._data)
            return self._data

        def _load_data(self, columns_to_remove: set[Route] | None = None) -> ak.Array:
            input_data = self.data
            for col in (columns_to_remove or set()):
                input_data = remove_ak_column(input_data, col.string_column, silent=True)
            return input_data

        @property
        def input_data(self) -> ak.Array:
            if self._input_data is None:
                self._input_data = self._load_data(columns_to_remove=self.target_columns)
                if self.data_type_transform:
                    self._input_data = self.data_type_transform(self._input_data)
            return self._input_data

        @property
        def target_data(self) -> ak.Array:
            if self._target_data is None and len(self.target_columns) > 0:
                self._target_data = self._load_data(columns_to_remove=self.data_columns)
                if self.data_type_transform:
                    self._target_data = self.data_type_transform(self._target_data)
            return self._target_data

        def __len__(self):
            length: int = 0
            if self._data is not None:
                length = len(self._data)
            elif self.meta_data and self.meta_data.get("col_counts", None):
                length = sum(self.meta_data["col_counts"])
            else:
                length = len(self.data)
            return length

        def _get_data(self, i: int | Sequence[int], input_data: ak.Array | None = None) -> ak.Array:
            data: ak.Array
            if input_data is None:
                data = self.input_data
            else:
                data = input_data

            return data[i]

        def _create_class_target(self, length: int, input_int_targets: int | None = None) -> ak.Array:
            int_target: int = input_int_targets or self.class_target
            arr = ak.Array([int_target] * int(length))
            if self.categorical_target_transform:
                arr = self.categorical_target_transform(arr)
            if self.data_type_transform:
                arr = self.data_type_transform(arr)
            return arr

        def __getitem__(
            self, i: int | Sequence[int],
        ) -> ak.Array | tuple[ak.Array, ak.Array] | tuple[ak.Array, ak.Array, ak.Array]:
            # from IPython import embed
            # embed(header=f"entering {self.__class__.__name__}.__getitem__ for index {i}")
            return_data = [self._get_data(i)]
            if len(self.target_columns) == 0 and len(self.int_targets) == 0:
                return_data = return_data[0]
            else:
                if self.target_data:
                    return_data.append(self._get_data(i, self.target_data))
                if len(self.int_targets) > 0:
                    return_data.append(self._create_class_target(ak.num(return_data[0], axis=0)))

            if self.batch_transform:
                return_data = self.batch_transform(return_data)
            return tuple(return_data) if isinstance(return_data, list) else return_data

        def __getitems__(self, idx: Sequence[int]) -> ak.Array:
            return self.__getitem__(idx)

        def to_list(self) -> list[dict[str, Any]]:
            return self.data.to_list()

    class FlatParquetDataset(ParquetDataset):  # noqa: F811
        def __init__(
            self,
            *args,
            padd_value_float: float = EMPTY_FLOAT,
            padd_value_int: int = EMPTY_INT,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.padd_values = {
                t: padd_value_float
                for t in [np.float16, np.float32, np.float64, np.float128]
            }
            self.padd_values.update({
                t: padd_value_int
                for t in [
                    np.uint8, np.uint16, np.uint32, np.uint64,
                    np.int8, np.int16, np.int32, np.int64,
                ]
            })

            self._input_data: Mapping[str, ak.Array] | None = None
            self._target_data: Mapping[str, ak.Array] | None = None
            self.class_target_name: str = "categorical_target"

        def _extract_columns(self, array: ak.Array, route: Route):
            # first, get super set of column
            super_route = Route(route.string_column)
            total_array = super_route.apply(array)

            # determine the type of the array values
            view = flat_np_view(total_array)
            val_type = view.dtype.type
            padding = self.padd_values.get(val_type, None)

            if padding is None:
                from IPython import embed
                embed(header=f"Error for route {route}, val_type={val_type}")
                raise ValueError(f"Could not determine padding value for type {val_type}")

            return route.apply(array, padding)

        @property
        def input_data(self) -> Mapping[str, ak.Array]:
            if self._input_data is None:
                self._input_data = self._load_data(columns_to_remove=self.target_columns)
                self._input_data = {
                    str(r): self._extract_columns(self._input_data, r)
                    for r in self.data_columns
                }
                if self.data_type_transform:
                    self._input_data = self.data_type_transform(self._input_data)
            return self._input_data

        @property
        def target_data(self) -> Mapping[str, ak.Array]:
            if self._target_data is None and len(self.target_columns) > 0:
                self._target_data = self._load_data(columns_to_remove=self.data_columns)
                self._target_data = {
                    str(r): self._extract_columns(self._target_data, r)
                    for r in self.target_columns
                }
                if self.data_type_transform:
                    self._target_data = self.data_type_transform(self._target_data)
            return self._target_data

        def __getitem__(self, i: int | Sequence[int]) -> Any | tuple | tuple:
            # from IPython import embed
            # embed(header=f"entering {self.__class__.__name__}.__getitem__ for index {i}")
            return_data = [{key: self._get_data(i, data) for key, data in self.input_data.items()}]
            if len(self.target_columns) == 0 and len(self.int_targets) == 0:
                return_data = return_data[0]
            else:
                if self.target_data:
                    return_data.append({key: self._get_data(i, data) for key, data in self.target_data.items()})
                if len(self.int_targets) > 0:
                    first_key = list(return_data[0].keys())[0]

                    return_data.append({
                        self.class_target_name: self._create_class_target(
                            len(return_data[0][first_key]), input_int_targets=self.class_target,
                        ),
                    })

            if self.batch_transform:
                return_data = self.batch_transform(return_data)
            return tuple(return_data) if isinstance(return_data, list) else return_data

    class FlatRowgroupParquetDataset(FlatParquetDataset):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._current_rowgroups: set[int] = set()
            self._allowed_rowgroups: set[int] = set()
            if not self.meta_data:
                raise ValueError("No metadata found, cannot determine rowgroups")

            # try to find information about rowgroups
            # the set of row groups could be in the open_options so check
            if self.open_options.get("row_groups", None):
                self._allowed_rowgroups = set(self.open_options["row_groups"])
            else:
                # otherwise, build from metadata
                self._allowed_rowgroups = set(range(self.meta_data["num_row_groups"]))

        def __len__(self):
            length: int = 0
            if self._data is not None:
                length = len(self._data)
            elif self.meta_data and self.meta_data.get("col_counts", None):
                # only consider allowed rowgroups
                col_counts = np.array(self.meta_data["col_counts"])
                length = sum(col_counts[list(self._allowed_rowgroups)])
            else:
                length = len(self.data)
            return length

        @property
        def current_rowgroups(self) -> set[int]:
            return self._current_rowgroups

        @current_rowgroups.setter
        def current_rowgroups(self, value: int | Iterable[int]) -> None:
            value_set: set[int] = set()
            if isinstance(value, int) or (isinstance(value, Iterable) and all(isinstance(x, int) for x in value)):
                value_set = set(value)
            elif isinstance(value, Iterable):
                value_set = set(*value)
            if not value_set == self.current_rowgroups:
                if not self._allowed_rowgroups.issuperset(value_set):
                    raise ValueError(
                        f"Rowgroup '{value_set}' contains unallowed rowgroups, whole set: {self._allowed_rowgroups}",
                    )
                self._current_rowgroups = value_set
                self._data = None
                self._input_data = None
                self._target_data = None
                self.open_options["row_groups"] = self.current_rowgroups

        def _concat_data(self, data1, data2) -> Any:
            if all(isinstance(x, dict) for x in [data1, data2]):
                keys1 = set(data1.keys())
                keys2 = set(data2.keys())
                if keys1 != keys2:
                    raise ValueError(f"cannot concatenate data of dict type with mismatching keys: {keys1=}, {keys2=}")
                return {key: self._concat_data(data1[key], data2[key]) for key in keys1}
            elif all(isinstance(x, ak.Array) for x in [data1, data2]):
                return ak.concatenate([data1, data2], axis=0)
            elif all(isinstance(x, torch.Tensor) for x in [data1, data2]):
                return torch.cat([data1, data2], axis=0)
            else:
                raise ValueError(f"Cannot concatenate data of type {type(data1)} and {type(data2)}")

        def __getitem__(
            self,
            i: (tuple[Collection[int], Collection[int]] |
                tuple[Collection[int], int] |
                Mapping[Collection[int], Collection[int]] |
                Mapping[Collection[int], int]),
        ) -> Any | tuple:
            return_data = None
            index_iter = iter(i)
            if isinstance(i, Mapping):
                index_iter = iter(i.items())
            for rowgroup, indices in index_iter:
                self.current_rowgroups = rowgroup

                chunk = super().__getitem__(indices)
                if not return_data:
                    return_data = chunk
                else:
                    return_data = (self._concat_data(a, b) for a, b in zip(return_data, chunk))
            return return_data

    class WeightedFlatRowgroupParquetDataset(FlatRowgroupParquetDataset):

        def __init__(
            self,
            *args,
            cls_weight: float | None = None,
            weight_columns: Collection[str, Route] | None = None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.cls_weight = cls_weight or 1.
            self.weight_columns: set[Route] = set()
            if weight_columns:
                self.weight_columns = set(Route(x) for x in weight_columns)

            self.all_columns |= self.weight_columns

        def _calculate_weights(self, indices: ak.Array) -> ak.Array:
            # calculate the weights for the given indices
            if self.weight_columns:
                weights = self._get_data(indices, self.data)[self.weight_columns]
                weights = ak.prod(weights, axis=-1)
            else:
                weights = ak.ones_like(indices, dtype=np.float32)
            final_weights = weights * self.cls_weight
            if self.data_type_transform:
                final_weights = self.data_type_transform(final_weights)

            if self.batch_transform:
                final_weights = self.batch_transform(final_weights)
            return final_weights

        def __getitem__(
            self,
            i: (tuple[Collection[int], Collection[int]] |
                tuple[Collection[int], int] |
                Mapping[Collection[int], Collection[int]] |
                Mapping[Collection[int], int]),
        ) -> Any | tuple:
            return_data = None
            index_iter = iter(i)
            if isinstance(i, Mapping):
                index_iter = iter(i.items())
            for rowgroup, indices in index_iter:
                self.current_rowgroups = rowgroup

                chunk = super().__getitem__(((rowgroup, indices),))
                if isinstance(chunk, tuple) and isinstance(chunk[0], dict):
                    # if the chunk is a tuple, we need to calculate the weights
                    # append the weight array to the tuple
                    weights = self._calculate_weights(indices)
                    chunk[0]["weights"] = weights
                elif isinstance(chunk, dict):
                    # if the chunk is a dict, we need to calculate the weights
                    # append the weight array to the dict
                    weights = self._calculate_weights(indices)
                    chunk["weights"] = weights

                if not return_data:
                    return_data = chunk
                else:
                    return_data = (self._concat_data(a, b) for a, b in zip(return_data, chunk))
            return return_data

    class FlatArrowRowGroupParquetDataset(FlatRowgroupParquetDataset):
        def __init__(self, *args, filters: pa._compute.Expression | list[str] | None = None, **kwargs):
            self.parquet_columns = None
            self.filter = filters
            self.rowgroup_fragments = []
            super().__init__(*args, **kwargs)

            first = self.path
            if isinstance(self.path, (list, tuple, set)):
                first = next(iter(self.path))

            # extract column structure from first parquet file
            self.parquet_columns = self._load_paquet_columns(pq.ParquetFile(first))

            for fragment in pq.ParquetDataset(self.path).fragments:
                self.rowgroup_fragments.extend(fragment.split_by_row_group())
            self.rowgroup_fragments = np.array(self.rowgroup_fragments)
            if self.filter:
                self._update_meta_data()

        def _load_paquet_columns(self, metadata: pq.ParquetFile) -> list[str]:
            # from awkward source code
            list_indicator = "list.item"
            for column_metadata in metadata.schema:
                if (
                    column_metadata.max_repetition_level > 0 and
                    ".list.element" in column_metadata.path
                ):
                    list_indicator = "list.element"
                    break
            subform = ak._connect.pyarrow.form_handle_arrow(
                metadata.schema_arrow, pass_empty_field=True,
            )
            if self.all_columns is not None:
                subform = subform.select_columns([str(x) for x in self.all_columns])

            # Handle empty field at root
            if metadata.schema_arrow.names == [""]:
                column_prefix = ("",)
            else:
                column_prefix = ()
            return subform.columns(
                list_indicator=list_indicator, column_prefix=column_prefix,
            )

        def _update_meta_data(self) -> None:
            # update the metadata with the new rowgroups
            if self.meta_data:
                self.meta_data["col_counts"] = [
                    rowgroup.count_rows(filter=self.filter)
                    for rowgroup in self.rowgroup_fragments
                ]
                self.meta_data["num_row_groups"] = len(self.rowgroup_fragments)
            else:
                raise ValueError("No metadata found, cannot update row groups")

        @property
        def data(self) -> ak.Array:
            if self._data is None:
                self.open_options["read_dictionary"] = self.parquet_columns
                self.open_options["filters"] = self.filter
                if "row_groups" in self.open_options:
                    del self.open_options["row_groups"]
                rg_idx = Ellipsis
                if self.current_rowgroups:
                    rg_idx = list(self.current_rowgroups)
                logger.info(f"Loading rowgroups {rg_idx} from {self.path}")
                from IPython import embed
                embed(header=f"debugging pq.read_table with {self.open_options=}")
                self._data = ak.concatenate([
                    ak.from_arrow(pq.read_table(x.open(), **self.open_options))
                    for x in self.rowgroup_fragments[rg_idx]],
                    axis=0,
                )

                if self.global_transform:
                    self._data = self.global_transform(self._data)
            return self._data
