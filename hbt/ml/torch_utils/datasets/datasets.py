from __future__ import annotations

__all__ = [
    "ListDataset", "ParquetDataset", "FlatParquetDataset",
    "FlatArrowRowGroupParquetDataset", "FlatRowgroupParquetDataset",
    "WeightedFlatRowgroupParquetDataset",
    "TensorParquetDataset", "WeightedRgTensorParquetDataset",
    "RgTensorParquetDataset", "WeightedTensorParquetDataset",
]

from collections.abc import Iterable, Mapping, Collection, Container, Sequence

from hbt.ml.torch_utils.datasets.mixins import PaddingMixin, RowgroupMixin, RowgroupWeightMixin, WeightMixin

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
FlatArrowRowGroupParquetDataset = MockModule("FlatArrowRowGroupParquetDataset")  # type: ignore
FlatRowgroupParquetDataset = MockModule("FlatRowgroupParquetDataset")  # type: ignore
WeightedFlatRowgroupParquetDataset = MockModule("WeightedFlatRowgroupParquetDataset")  # type: ignore
TensorParquetDataset = MockModule("TensorParquetDataset")  # type: ignore
RgTensorParquetDataset = MockModule("RgTensorParquetDataset")  # type: ignore
WeightedRgTensorParquetDataset = MockModule("WeightedRgTensorParquetDataset")  # type: ignore
WeightedTensorParquetDataset = MockModule("WeightedTensorParquetDataset")

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
            input_data_transform: Callable | None = None,
            categorical_target_transform: Callable | None = None,
            data_type_transform: Callable | None = None,
            data_loader_func: Callable | None = None,
            idx: Sequence[int] | None = None,
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
            self.input_data_transform = input_data_transform
            self.global_transform = global_transform

            # variables for data loading
            self.data_loader_func = data_loader_func or ak.from_parquet
            self.idx: Sequence[int] | None = idx

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
            self.input_data_transform = first.input_data_transform
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
                self.all_columns = self.columns.copy()
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
                from IPython import embed
                embed(header=f"columns={self.columns}, all_columns={self.all_columns}, ")
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
                self.input_data_transform,
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
                self._data = self.data_loader_func(self.path, **self.open_options)

                if self.idx:
                    self._data = self._data[self.idx]

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
                if self.input_data_transform:
                    self._input_data = self.input_data_transform(self._input_data)
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
            elif self.idx:
                length = len(self.idx)
            elif self.meta_data and self.meta_data.get("col_counts", None):
                length = sum(self.meta_data["col_counts"])
            else:
                length = len(self.data)
            return length

        def _get_data(self, i: int | Sequence[int] | slice, input_data: ak.Array | None = None) -> ak.Array:
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
            self, i: int | Sequence[int] | slice,
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

        def __getitems__(self, idx: Sequence[int] | slice) -> ak.Array:
            return self.__getitem__(idx)

        def to_list(self) -> list[dict[str, Any]]:
            return self.data.to_list()

    class FlatParquetDataset(  # noqa: F811
        PaddingMixin,
        ParquetDataset,
    ):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)

            self._input_data: Mapping[str, ak.Array] | None = None
            self._target_data: Mapping[str, ak.Array] | None = None
            self.class_target_name: str = "categorical_target"

        @property
        def input_data(self) -> Mapping[str, ak.Array]:
            if self._input_data is None:
                self._input_data = self._load_data(columns_to_remove=self.target_columns)
                self._input_data = {
                    str(r): self._extract_columns(self._input_data, r)
                    for r in self.data_columns
                }
                if self.input_data_transform:
                    self._input_data = self.input_data_transform(self._input_data)
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

        def __getitem__(self, i: int | Sequence[int] | slice) -> Any | tuple | tuple:
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

    class TensorParquetDataset(  # noqa: F811
        PaddingMixin,
        ParquetDataset,
    ):  # noqa: F811
        def __init__(
            self,
            *args,
            continuous_features: Container[str] | None = None,
            categorical_features: Container[str] | None = None,
            **kwargs,
        ):
            """ParquetDataset that loads the data as torch tensors.
            Input features are split into continuous and categorical features.
            Corresponding columns are loaded according to the string representation
            of the column names.

            Output of :py:func:`__getitem__` is a tuple of the form `list[input_data, target_data]`.
            In this representation, `input_data` can be the following:

            - `torch.Tensor` if either categorical or continuous features are defined, but not both
            - `list[torch.Tensor, torch.Tensor]` if both categorical and continuous features are defined.
                First tensor is categorical, second is continuous.

            In case `cls_weights` are defined, they are appended to `input_data`, i.e. are the last values
            in the list.

            :param continuous_features: List, tuple or set of continuous features to load
            :param categorical_features: List, tuple or set of categorical features to load
            :param args: Arguments to pass to upstream classes
            :param kwargs: Additional arguments to pass to upstream classes
            """

            # overwrite the columns to load
            self.continuous_features = continuous_features or {}
            self.categorical_features = categorical_features or {}
            columns = set(self.continuous_features) | set(self.categorical_features)
            if getattr(kwargs, "columns", None) is not None:
                logger.warning(
                    f"{self.__class__.__name__} only supports input feature columns "
                    "that are split by categorical and continuous features. Will ignore "
                    f"columns={kwargs['columns']} and use columns={columns} instead.",
                )
            kwargs["columns"] = columns
            super().__init__(*args, **kwargs)

            # resolve continuous and categorical features
            tmp_cont_fetures = set()
            tmp_cat_features = set()

            for col in self.all_columns:
                if self.target_columns and col in self.target_columns:
                    continue
                if (
                    any(self._check_against_pattern(str(cont_feature), col.string_column)
                    for cont_feature in self.continuous_features)
                ):
                    tmp_cont_fetures.add(col)
                elif (
                    any(self._check_against_pattern(str(cat_feature), col.string_column)
                    for cat_feature in self.categorical_features)
                ):
                    tmp_cat_features.add(col)
            self.continuous_features = tmp_cont_fetures
            self.categorical_features = tmp_cat_features

            self._input_data: torch.Tensor | list[torch.Tensor] | None = None
            self._target_data: torch.Tensor | list[torch.Tensor] | None = None
            self.class_target_name: str = "categorical_target"

        def _array_set_to_tensor(self, features: list[str | Route]) -> torch.Tensor:
            return torch.cat(
                [
                    ak.to_torch(self._extract_columns(self.data, r)).reshape(-1, 1)
                    for r in sorted(features, key=lambda x: str(x))
                ],
                axis=-1,
            )

        @property
        def input_data(self) -> torch.Tensor | list[torch.Tensor]:
            if self._input_data is None:
                if not all(len(x) > 0 for x in [self.continuous_features, self.categorical_features]):
                    to_fill = self.continuous_features or self.categorical_features
                    self._input_data = self._array_set_to_tensor(to_fill)
                else:
                    cat_features = self._array_set_to_tensor(self.categorical_features)
                    cont_features = self._array_set_to_tensor(self.continuous_features)
                    self._input_data = [cat_features, cont_features]
                if self.input_data_transform:
                    self._input_data = self.input_data_transform(self._input_data)
                if self.data_type_transform:
                    self._input_data = self.data_type_transform(self._input_data)
            return self._input_data

        @property
        def target_data(self) -> torch.Tensor:
            if self._target_data is None and len(self.target_columns) > 0:
                self._target_data = self._array_set_to_tensor(self.target_columns)
                if self.data_type_transform:
                    self._target_data = self.data_type_transform(self._target_data)
            return self._target_data

        def _create_class_target(self, length: int, input_int_targets: int | None = None) -> ak.Array:
            int_target: int = input_int_targets or self.class_target
            arr = torch.ones(length) * int_target
            if self.categorical_target_transform:
                arr = self.categorical_target_transform(arr)
            if self.data_type_transform:
                arr = self.data_type_transform(arr)
            return arr

        def __getitem__(self, i: int | Sequence[int] | slice) -> Any | tuple | tuple:
            # from IPython import embed
            # embed(header=f"entering {self.__class__.__name__}.__getitem__ for index {i}")
            if isinstance(self.input_data, torch.Tensor):
                return_data = [self._get_data(i, self.input_data)]
            else:
                return_data = [list(map(lambda tensor: self._get_data(i, tensor), self.input_data))]
            if len(self.target_columns) == 0 and len(self.int_targets) == 0:
                return_data = return_data[0]
            else:
                if self.target_data:
                    return_data.append(self._get_data(i, self.target_data))
                if len(self.int_targets) > 0:
                    return_data.append(
                        self._create_class_target(
                            len(i), input_int_targets=self.class_target,
                        ),
                    )

            if self.batch_transform:
                return_data = self.batch_transform(return_data)
            return tuple(return_data) if isinstance(return_data, list) else return_data

    class FlatRowgroupParquetDataset(RowgroupMixin, FlatParquetDataset):
        pass

    class WeightedFlatRowgroupParquetDataset(RowgroupWeightMixin, FlatRowgroupParquetDataset):
        pass

    class WeightedTensorParquetDataset(WeightMixin, TensorParquetDataset):

        def _calculate_weights(self, indices: ak.Array) -> ak.Array:
            # calculate the weights for the given indices
            if self.weight_columns:
                weights = self._get_data(indices, self.data)[self.weight_columns]
                weights = ak.prod(weights, axis=-1)
            else:
                weights = ak.ones_like(indices, dtype=np.float32)
            final_weights = ak.to_torch(weights * self.cls_weight)

            if self.data_type_transform:
                final_weights = self.data_type_transform(final_weights)

            if self.batch_transform:
                final_weights = self.batch_transform(final_weights)
            return final_weights

    class RgTensorParquetDataset(RowgroupMixin, TensorParquetDataset):
        pass

    class WeightedRgTensorParquetDataset(RowgroupMixin, WeightedTensorParquetDataset):
        pass

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
