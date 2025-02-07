from __future__ import annotations

__all__ = [
    "ListDataset", "MapAndCollate", "FlatMapAndCollate", "NodesDataLoader"
]

from columnflow.util import MockModule, maybe_import
from columnflow.types import ModuleType

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")

ListDataset = MockModule("ListDataset")
MapAndCollate = MockModule("MapAndCollate")
FlatMapAndCollate = MockModule("FlatMapAndCollate")
NodesDataLoader = MockModule("NodesDataLoader")

if not isinstance(torchdata, MockModule):
    import torchdata.nodes as tn
    from torch.utils.data import RandomSampler, SequentialSampler, default_collate, Dataset
    from typing import Literal, Sized
    class ListDataset(Dataset):

        def __init__(self, len: int, prefix: str = "data"):
            self.len = len
            self.prefix = prefix
        
        def __len__(self):
            return self.len
        
        def __getitem__(self, i: int) -> str:
            return f"{self.prefix}_{i}"
        
    class MapAndCollate:
        """A simple transform that takes a batch of indices, maps with dataset, and then applies
        collate.
        TODO: make this a standard utility in torchdata.nodes
        """
        
        def __init__(self, dataset, collate_fn):
            self.dataset = dataset
            self.collate_fn = collate_fn
            
        def __call__(self, batch_of_indices: list[int]):
            batch = [self.dataset[i] for i in batch_of_indices]
            return self.collate_fn(batch)
        
    class FlatMapAndCollate:
        """A simple transform that takes a batch of indices, maps with dataset, and then applies
        collate.
        TODO: make this a standard utility in torchdata.nodes
        """
        
        def __init__(self, dataset, collate_fn):
            self.dataset = dataset
            self.collate_fn = collate_fn
            
        def __call__(self, idx: int):
            batch = self.dataset[idx]
            return self.collate_fn(batch)


    # To keep things simple, let's assume that the following args are provided by the caller
    def NodesDataLoader(
        dataset: Sized,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        collate_fn: Callable | None,
        pin_memory: bool,
        drop_last: bool,
        parallelize_method: Literal["thread", "process"] = "process",
    ):
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
        map_and_collate = FlatMapAndCollate(dataset, collate_fn or default_collate)

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