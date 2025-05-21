from __future__ import annotations

__all__ = [
    "NodesDataLoader", "CompositeDataLoader",
]

from collections.abc import Iterable, Mapping
from columnflow.util import MockModule, maybe_import
from columnflow.types import Any, Callable

from hbt.ml.torch_utils.map_and_collate import MapAndCollate, FlatMapAndCollate, NestedMapAndCollate
from hbt.ml.torch_utils.batcher import BatchedMultiNodeWeightedSampler
from hbt.ml.torch_utils.datasets import ParquetDataset


torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")

NodesDataLoader = MockModule("NodesDataLoader")  # type: ignore
CompositeDataLoader = MockModule("CompositeDataLoader")  # type: ignore


if not isinstance(torchdata, MockModule):
    import torchdata.nodes as tn
    from torch.utils.data import RandomSampler, SequentialSampler, default_collate
    from typing import Literal, Sized

    # To keep things simple, let's assume that the following args are provided by the caller
    def NodesDataLoader(  # noqa: F811
        dataset: Sized,
        shuffle: bool,
        num_workers: int,
        collate_fn: Callable | None,
        pin_memory: bool,
        parallelize_method: Literal["thread", "process"] = "process",
        mapping_base_cls: MapAndCollate | None = None,
    ) -> tn.Loader[Sized]:
        # Assume we're working with a map-style dataset
        assert hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__")
        # Start with a sampler, since caller did not provide one
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        # Sampler wrapper converts a Sampler to a BaseNode
        node = tn.SamplerWrapper(sampler)

        # Now let's batch sampler indices together
        # node = tn.Batcher(node, batch_size=batch_size, drop_last=drop_last)

        # Create a Map Function that accepts a list of indices, applies getitem to it, and
        # then collates them
        if not mapping_base_cls:
            map_and_collate = FlatMapAndCollate(dataset, collate_fn or default_collate)
        else:
            map_and_collate = mapping_base_cls(dataset, collate_fn or default_collate)

        # MapAndCollate is doing most of the heavy lifting, so let's parallelize it. We could
        # choose process or thread workers. Note that if you're not using Free-Threaded
        # Python (eg 3.13t) with -Xgil=0, then multi-threading might result in GIL contention,
        # and slow down training.
        node = tn.ParallelMapper(
            node,
            map_fn=map_and_collate,
            num_workers=num_workers,
            method=parallelize_method,  # Set this to "thread" for multi-threading
            in_order=True,
        )

        # Optionally apply pin-memory, and we usually do some pre-fetching
        if pin_memory:
            node = tn.PinMemory(node)
        node = tn.Prefetcher(node, prefetch_factor=num_workers * 2)

        # Note that node is an iterator, and once it's exhausted, you'll need to call .reset()
        # on it to start a new Epoch.
        # Insteaad, we wrap the node in a Loader, which is an iterable and handles reset. It
        # also provides state_dict and load_state_dict methods.
        return tn.Loader(node)

    class CompositeDataLoader(object):  # noqa: F811

        def __init__(
                self,
                data_map: Mapping[str, Sized] | Iterable[Sized] | None = None,
                weight_dict: Mapping[str, float | Mapping[str, float]] | Iterable[Sized] | None = None,
                shuffle: bool = True,
                batch_size: int = 256,
                num_workers: int = 0,
                parallelize_method: Literal["thread", "process"] = "process",
                collate_fn: Callable | None = None,
                batch_sampler_cls: Callable | None = None,
                batcher_options: dict[str, Any] | None = None,
                index_sampler_cls: Callable | None = None,
                map_and_collate_cls: Callable | None = None,
                pin_memory: bool = False,
                device=None,
        ):

            self.data_map = data_map
            self.weight_dict = weight_dict
            self.shuffle = shuffle
            self.batch_size = batch_size
            self.batcher_options: dict[str, Any] = batcher_options or dict()
            self.num_workers = num_workers
            self.parallelize_method: Literal["thread", "process"] = parallelize_method
            self.collate_fn = collate_fn
            self.batch_sampler_cls = batch_sampler_cls
            self.index_sampler_cls = index_sampler_cls
            self.map_and_collate_cls = map_and_collate_cls
            self.pin_memory = pin_memory
            self.device = device

            # property for maximum number of batches
            self._num_batches: int | None = None

            self._resolve_defaults()

            self.data_loader, self.batcher = self._create_composite_node()

        def _resolve_defaults(self):
            self.index_sampler_cls: Callable
            if not self.index_sampler_cls:
                from torch.utils.data import RandomSampler, SequentialSampler
                if self.shuffle:
                    self.index_sampler_cls = RandomSampler
                else:
                    self.index_sampler_cls = SequentialSampler

            self.batch_sampler_cls: Callable = self.batch_sampler_cls or BatchedMultiNodeWeightedSampler

            self.map_cls: Callable = self.map_and_collate_cls or NestedMapAndCollate
            if issubclass(self.batch_sampler_cls, BatchedMultiNodeWeightedSampler):
                node_dict = {
                    key: tn.SamplerWrapper(self.index_sampler_cls(dataset))
                    for key, dataset in self.data_map.items()
                }
                self.batcher_options["source_nodes"] = node_dict
                # if self.pin_memory:
                #     self.batcher_options["source_nodes"] = {
                #         key: tn.PinMemory(data) for key, data in node_dict.items()
                #     }
                self.batcher_options["weights"] = self.weight_dict

        def _create_composite_node(self) -> tuple[tn.ParallelMapper, Any]:

            batcher = self.batch_sampler_cls(
                batch_size=self.batch_size, **self.batcher_options,
            )

            mapping = self.map_cls(self.data_map, collate_fn=self.collate_fn)
            mp_context = None
            if (self.device and self.device.type == "cuda") and self.num_workers > 1:
                # mp_context = torch.multiprocessing.get_context("spawn")
                mp_context = "spawn"

            parallel_node = tn.ParallelMapper(
                batcher,
                map_fn=mapping,
                num_workers=self.num_workers,
                method=self.parallelize_method,  # Set this to "thread" for multi-threading
                multiprocessing_context=mp_context,
                in_order=True,
            )
            if self.pin_memory:
                parallel_node = tn.PinMemory(parallel_node)

            return (parallel_node, batcher)

        def __len__(self):
            output = 0
            if isinstance(self.data_map, Mapping):
                output = sum(len(x) for x in self.data_map.values())
            else:
                output = len(self.data_map)
            return output

        @property
        def num_batches(self):
            if not self._num_batches:
                datasets: list[ParquetDataset] = list(self.data_map.values())
                dataset_names = list(self.data_map.keys())
                max_dataset_idx: int = np.argmax([len(data) for data in datasets])
                max_composition: int = self.batcher._batch_composition[dataset_names[max_dataset_idx]]
                self._num_batches = int(len(datasets[max_dataset_idx]) / max_composition)
            return self._num_batches
