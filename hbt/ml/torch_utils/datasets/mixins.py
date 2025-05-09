from __future__ import annotations

from collections.abc import Iterable, Mapping, Collection
from columnflow.columnar_util import Route, EMPTY_FLOAT, EMPTY_INT, flat_np_view
from columnflow.util import maybe_import
from columnflow.types import Any

np = maybe_import("numpy")
ak = maybe_import("awkward")
torch = maybe_import("torch")


class RowgroupMixin:

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


class PaddingMixin:
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


class WeightMixin:

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

        if self.cls_weight:
            weights = weights * self.cls_weight
        if self.data_type_transform:
            weights = self.data_type_transform(weights)

        if self.batch_transform:
            weights = self.batch_transform(weights)
        return weights

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
            if self.weight_columns or self.cls_weight:
                weights = self._calculate_weights(indices)
                if isinstance(chunk, (tuple, list)) and isinstance(chunk[0], dict):
                    # if the chunk is a tuple, we need to calculate the weights
                    # append the weight array to the tuple
                    chunk[0]["weights"] = weights
                elif isinstance(chunk, (tuple, list)) and isinstance(chunk[0], (list)):
                    chunk[0].append(weights)
                elif isinstance(chunk, (tuple, list)) and isinstance(chunk[0], torch.Tensor):
                    chunk = ([chunk[0], weights], chunk[1])
                elif isinstance(chunk, dict):
                    # if the chunk is a dict, we need to calculate the weights
                    # append the weight array to the dict
                    chunk["weights"] = weights

            if not return_data:
                return_data = chunk
            else:
                return_data = (self._concat_data(a, b) for a, b in zip(return_data, chunk))
        return return_data