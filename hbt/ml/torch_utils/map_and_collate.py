from __future__ import annotations

__all__ = [
    "MapAndCollate",
    "FlatMapAndCollate",
    "NestedMapAndCollate",
    "NestedDictMapAndCollate",
    "FlatListRowgroupMapAndCollate",
    "NestedListRowgroupMapAndCollate",
]

from collections.abc import Collection, Mapping
from abc import abstractmethod
from columnflow.util import MockModule, maybe_import
from columnflow.types import T, Callable, Sequence

torch = maybe_import("torch")
ak = maybe_import("awkward")
np = maybe_import("numpy")

MapAndCollate = MockModule("MapAndCollate")  # type: ignore
FlatMapAndCollate = MockModule("FlatMapAndCollate")  # type: ignore
NestedMapAndCollate = MockModule("NestedMapAndCollate")  # type: ignore
NestedDictMapAndCollate = MockModule("NestedDictMapAndCollate")  # type: ignore

if not isinstance(torch, MockModule):
    from hbt.ml.torch_utils.datasets import FlatRowgroupParquetDataset
    from hbt.ml.torch_utils.utils import reorganize_idx

    class TensorMixin:
        """Mixin class to add tensor support to MapAndCollate classes."""

        def _concat_tensors(self, input_arrays: Sequence[torch.Tensor], *args, **kwargs) -> torch.Tensor:
            # helper function to concatenate different types of objects
            first_tensor = input_arrays[0]
            if isinstance(first_tensor, torch.Tensor):
                # if the first tensor is a torch tensor, use torch.cat
                return torch.cat(input_arrays, *args, **kwargs)
            elif isinstance(first_tensor, (list, tuple)):
                # if the first tensor is a list or tuple, use torch.cat
                return [self._concat_tensors(x, *args, **kwargs) for x in zip(*input_arrays)]

    class DictMixin:
        def _concat_dicts(
            self,
            input_arrays: Sequence[dict[str, T]],
            *args,
            **kwargs,
        ) -> dict[str, T]:
            # helper function to concatenate different types of objects
            return_dict = dict()
            first_dict = input_arrays[0]
            try:
                for key in first_dict.keys():
                    sub_arrays = list(map(lambda x: x.get(key), input_arrays))
                    collate_fn = ak.concatenate
                    if all(isinstance(x, torch.Tensor) for x in sub_arrays):
                        collate_fn = torch.cat
                    try:
                        return_dict[key] = collate_fn(sub_arrays, *args, **kwargs)
                    except Exception as e:
                        print(e)
                        from IPython import embed
                        embed(header=f"Encountered error for key {key} in {self.__class__.__name__}._concat_dict")
            except Exception as e:
                print(e)
                from IPython import embed
                embed(header=f"Encountered error in {self.__class__.__name__}._concat_dict when looping through keys")
                raise e

            return return_dict

    class BaseBackendMixin:

        def __init__(
            self,
            dataset: Collection[T],
            collate_fn: Callable,
            weights: Collection[float] | Mapping[str, float] | None = None,
            *args,
            **kwargs,
        ):
            self.dataset = dataset
            self.weights = weights

            # collate_fn is artifact from old implementation, leave it for compatibility
            self.collate_fn = collate_fn

        @abstractmethod
        def _unit_collate_fn(self, input_arrays: Sequence[T], *args, **kwargs) -> T:
            """A method to collate a single unit of data, e.g. a single row or a single sample."""
            raise NotImplementedError("This method should be implemented in subclasses.")

        def _concat_batches(
            self,
            batch: list[T],
            current_batch: Sequence[T],
            concat_fn: Callable,
            *args,
            **kwargs,
        ) -> Sequence[T]:
            if isinstance(current_batch, (tuple, list)):
                if len(batch) == 0:
                    batch = list(current_batch)
                else:
                    for idx, item in enumerate(current_batch):
                        batch[idx] = concat_fn((batch[idx], item), *args, **kwargs)
            else:
                batch = concat_fn((batch, current_batch), *args, **kwargs)
            return batch

    class ListBackendMixin(BaseBackendMixin):
        dataset: Collection[T]
        weights: Collection[float] | None

        def __call__(self, idx: dict[tuple[int], Sequence[int]]) -> Sequence[T]:
            """A simple transform that takes a batch of indices, maps with dataset, and then applies
            collate.
            """
            batch: list[T] = []

            # the indices are dictionaries with multiple entries, so loop
            idx = reorganize_idx(idx)
            for (dataset_idx,), entry_idx in idx.items():
                try:
                    dataset = self.dataset[dataset_idx]
                    if self.weights:
                        weight = self.weights[dataset_idx]
                        dataset.cls_weight = weight
                    else:
                        dataset.cls_weight = None
                    current_batch = dataset[entry_idx]
                    batch = self._concat_batches(
                        batch=batch,
                        current_batch=current_batch,
                        concat_fn=self._unit_collate_fn,
                    )
                except Exception as e:  # noqa: F841
                    from IPython import embed
                    embed(header=f"Detected problem in {self.__class__.__name__}")

            return batch

    class NestedBackendMixin(ListBackendMixin):
        dataset: Mapping[str, T]
        weights: Mapping[str, float] | None

        def __call__(self, idx: dict[str, Sequence[int]]) -> Sequence[T]:
            batch: list[T] = []

            for key, indices in idx.items():
                if self.weights:
                    weight = self.weights[key]
                    self.dataset[key].cls_weight = weight
                try:
                    current_batch = self.dataset[key][indices]
                except Exception as e:
                    from IPython import embed
                    embed(header=f"failed to load entries for key {key} in {self.__class__.__name__}")
                    raise e
                batch = self._concat_batches(batch=batch, current_batch=current_batch, concat_fn=self._unit_collate_fn)

            return batch

    class FlatRowgroupBackendMixin(BaseBackendMixin):
        """A simple transform that takes a batch of indices, maps with dataset, and then applies
        collate.
        TODO: make this a standard utility in torchdata.nodes
        """
        dataset: Collection[T]
        weights: Collection[float] | None

        def __call__(self, idx: dict[tuple[int, int], Sequence[int]]) -> Sequence[T]:
            batch: list[T] = []

            # the indices are dictionaries with multiple entries, so loop
            idx = reorganize_idx(idx)
            for (dataset_idx, rowgroup), entry_idx in idx.items():
                try:
                    dataset = self.dataset[dataset_idx]
                    if self.weights:
                        weight = self.weights[dataset_idx]
                        dataset.cls_weight = weight
                    else:
                        dataset.cls_weight = None
                    current_batch = dataset[((rowgroup, entry_idx),)]
                    batch = self._concat_batches(batch=batch, current_batch=current_batch, concat_fn=self._unit_collate_fn)
                except Exception as e:  # noqa: F841
                    from IPython import embed
                    embed(header=f"Detected problem in {self.__class__.__name__}")

            return batch

    class NestedRowgroupBackendMixin(FlatRowgroupBackendMixin):
        all_datasets: Mapping[str, Collection[T]]

        def __call__(self, idx: dict[str, dict[tuple[int, int], Sequence[int]]]) -> Sequence[T]:
            batch: list[T] = []
            keys = np.array(list(idx.keys()))

            worker_info = torch.utils.data.get_worker_info()
            if worker_info and worker_info.num_workers > 1 and worker_info.id is not None:
                key_idx = np.indeces(keys.shape)
                mask = ((key_idx + 1) % (worker_info.id + 1)) == 0
                keys = keys[key_idx[mask]]
                print(f"Worker {worker_info.id}: {keys=}")

            for key in keys:
                indices = idx[key]
                self.dataset = self.all_datasets[key]
                current_batch = super().__call__(indices)
                batch = self._concat_batches(batch=batch, current_batch=current_batch, concat_fn=self._unit_collate_fn)

            return batch

    class MapAndCollate:  # noqa: F811
        """A simple transform that takes a batch of indices, maps with dataset, and then applies
        collate.
        TODO: make this a standard utility in torchdata.nodes
        """

        def __init__(
            self,
            dataset: Collection[T],
            collate_fn: Callable,
        ):
            self.dataset = dataset
            self.collate_fn = collate_fn
            self.weights: Collection[float] | None = None

        def __call__(self, batch_of_indices: list[int]):
            batch = [self.dataset[i] for i in batch_of_indices]
            return self.collate_fn(batch)

    class FlatMapAndCollate(MapAndCollate):  # noqa: F811
        """A simple transform that takes a batch of indices, maps with dataset, and then applies
        collate.
        TODO: make this a standard utility in torchdata.nodes
        """
        def __call__(self, idx: int | Sequence[int]):
            if self.weights:
                self.dataset.cls_weight = self.weights
            batch = self.dataset[idx]
            return self.collate_fn(batch)

    class NestedMapAndCollate(NestedBackendMixin):  # noqa: F811
        def __init__(
            self,
            dataset: dict[str, Collection[T]],
            collate_fn: Callable | None = None,
            weights: Collection[float] | Mapping[str, float] | None = None,
        ):
            self.dataset = dataset
            self.weights = weights

        def _unit_collate_fn(self, input_arrays: Sequence[T], *args, **kwargs) -> T:
            current_batch = input_arrays[0]
            concat_fn = ak.concatenate
            if isinstance(current_batch, (list, tuple)):
                if all(isinstance(x, torch.Tensor) for x in current_batch):
                    concat_fn = torch.cat
            elif isinstance(current_batch, torch.Tensor):
                concat_fn = torch.cat
            from IPython import embed
            embed(header=f"loading events, {concat_fn=}")
            return concat_fn(input_arrays, *args, **kwargs)

    class NestedDictMapAndCollate(DictMixin, NestedBackendMixin):  # noqa: F811

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._unit_collate_fn = self._concat_dicts

    class FlatListRowgroupMapAndCollate(DictMixin, FlatRowgroupBackendMixin):
        """A simple transform that takes a batch of indices, maps with dataset, and then applies
        collate.
        TODO: make this a standard utility in torchdata.nodes
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._unit_collate_fn = self._concat_dicts

    class NestedListRowgroupMapAndCollate(DictMixin, NestedRowgroupBackendMixin):
        dataset: dict[str, Sequence[FlatRowgroupParquetDataset]]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.all_datasets = self.dataset
            self._unit_collate_fn = self._concat_dicts

    class TensorListRowgroupMapAndCollate(TensorMixin, FlatRowgroupBackendMixin):
        """A simple transform that takes a batch of indices, maps with dataset, and then applies
        collate.
        TODO: make this a standard utility in torchdata.nodes
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._unit_collate_fn = self._concat_tensors

    class TensorListMapAndCollate(TensorMixin, ListBackendMixin):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._unit_collate_fn = self._concat_tensors

    class NestedTensorListRowgroupMapAndCollate(TensorMixin, NestedRowgroupBackendMixin):
        dataset: dict[str, Sequence[FlatRowgroupParquetDataset]]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.all_datasets = self.dataset
            self._unit_collate_fn = self._concat_tensors

    class NestedTensorMapAndCollate(TensorMixin, NestedBackendMixin):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._unit_collate_fn = self._concat_tensors
