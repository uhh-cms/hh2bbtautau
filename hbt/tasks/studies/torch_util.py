from __future__ import annotations

__all__ = [
    "ListDataset", "MapAndCollate", "FlatMapAndCollate", "NodesDataLoader"
]

from collections import defaultdict
from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import T

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")

ListDataset = MockModule("ListDataset")
MapAndCollate = MockModule("MapAndCollate")
FlatMapAndCollate = MockModule("FlatMapAndCollate")
NodesDataLoader = MockModule("NodesDataLoader")

if not isinstance(torchdata, MockModule):
    import torchdata.nodes as tn
    from torchdata.nodes import MultiNodeWeightedSampler
    from torch.utils.data import RandomSampler, SequentialSampler, default_collate, Dataset
    from torchdata.nodes.samplers.stop_criteria import StopCriteria
    from torchdata.nodes.samplers.multi_node_weighted_sampler import _WeightedSampler
    from typing import Literal, Sized

    class BatchedMultiNodeWeightedSampler(MultiNodeWeightedSampler):

        def __init__(
            self,
            *args,
            batch_size: int,
            weights: dict[str, float | dict[str, float]],  # type: ignore
            **kwargs
        ):
            self.batch_size = batch_size
            super().__init__(*args, weights = weights, **kwargs)

            # the weights dictionary is used to determine the composition of each batch
            # these weights should be equal or larger than one to indicate that
            # a dataset should be overrepresented in the batch

            # the weight can also be a dictionary of weights for each key.
            # In this case, the weights are used to sample the contribution of 
            # each sub dataset to the batch. The sum of the weights are used to
            # calculate the batch contribution
            
            # setup batches per sample
            total_weight_sum = 0
            # dictionary to store meta information: is a weighted sampler
            # needed for a top-level dataset?
            self._weight_samplers = list()
            for key, weight in self.weights:

                # if the weight is a float number, add it to the sum
                if isinstance(weight, (int, float)):
                    total_weight_sum += weight
                    # in this case, we don't need to create a sampler
                elif isinstance(weight, dict):
                    total_weight_sum += sum(weight.values())
                    # in these cases, we will use a weight sampler for the
                    # sub datasets
                    self._weight_samplers.append(key)

            # calculate the composition of the batches
            self._batch_composition = {
                key: (weight*self.batch_size // total_weight_sum
                    if isinstance(weight, (int, float))
                    else sum(weight.values())*self.batch_size // total_weight_sum
                )
                for key, weight in self.weights.items()
            }
            
            # default dictionary to store weighted samplers where necessary
            self._weighted_sampler = defaultdict(None)
            
        def _get_new_weighted_sampler(self, initial_state=None):
            for key in self._weight_samplers:
                self._weighted_sampler[key] = _WeightedSampler(
                    weights=self.weights[key],
                    seed=self.seed,
                    rank=self.rank,
                    world_size=self.world_size,
                    epoch=self._epoch,
                    initial_state=(initial_state[self.WEIGHTED_SAMPLER_STATE_KEY] if initial_state is not None else None),
                )

        
        def _validate(self) -> None:
            if self.stop_criteria not in [
                StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
                StopCriteria.ALL_DATASETS_EXHAUSTED,
                StopCriteria.FIRST_DATASET_EXHAUSTED,
            ]:
                raise ValueError(
                    f"Invalid {self.stop_criteria=}. stop_criteria must be one of: CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED, FIRST_DATASET_EXHAUSTED, ALL_DATASETS_EXHAUSTED"
                )

            # Validate if keys of source_nodes and weights are the same
            if set(self.dataset_names) != set(self.weights.keys()) or len(self.dataset_names) != len(self.weights):
                raise ValueError(
                    f"Invalid {self.weights=}. For multi-dataset weighted sampling, keys of source_nodes and weights must be the same",
                )
            
            if not isinstance(self.batch_size, int) and not self.batch_size >= 1:
                raise ValueError(f"batch_size argument must be >= 1, received {self.batch_size}")


            def _weight_check(weight):
                if not isinstance(weight, float) or weight <= 0:
                    raise ValueError(
                        f"""Invalid {self.weights=}. For multi-dataset weighted sampling, weights must be a 1d sequence, non-negative, and non-zero.
                        Weights are used to sample from source nodes. Zero weight means the source node will never be sampled from, and can cause
                        unexpected behavior depending on the stop criteris. Weights are used as inputs to torch.multinomial, please refer to
                        https://pytorch.org/docs/stable/generated/torch.multinomial.html on how to use weights for sampling."""
                    )

            all_keys = set(self.weights.keys())
            for weight in self.weights.values():
                if isinstance(weight, dict):
                    all_keys.update(weight.keys())
                    for w in weight.values():
                        _weight_check(w)
                else:
                    _weight_check(weight)
            
            # check if all keys in weights are also accounted for in the source_nodes
            difference = all_keys.symmetric_difference(set(self.source_nodes.keys()))
            if len(difference) >= 1:
                raise ValueError(
                    "Following keys are defined in either source nodes or weight dict, but not the other: "
                    ", ".join(difference)
                )
                

        def next_tautauNN(self) -> T:
            # prepare indices for random sampling
            indices = [np.array([], dtype=np.int32) for _ in range(self.n_datasets)]
            offsets = [0] * self.n_datasets

            # start iterating
            while True:
                # determine batch sizes per dataset for this chunk
                batch_sizes = torch.multinomial(self.batch_size, self.weights.values)

                # fill chunks per dataset that eventually form a batch
                chunks = []
                for i, (arrays, _indices, batch_size, offset) in enumerate(zip(
                        self.source_nodes.values, indices, batch_sizes, offsets
                    )):
                    # update indices and offset
                    if len(_indices) - offset < batch_size:
                        new_indices = np.arange(len(arrays[0]), dtype=np.int32)
                        np.random.shuffle(new_indices)
                        _indices = indices[i] = np.concatenate([_indices[offset:], new_indices], axis=0)
                        offset = 0

                    # fill the chunk and adjust the offset
                    chunks.append([a[_indices[offset:offset + batch_size]] for a in arrays])
                    offsets[i] = offset + batch_size

                # yield
                data = tuple(
                    np.concatenate([chunk[i] for chunk in chunks], axis=0)
                    for i in range(self.tuple_length)
                )
                data = transform_data(self, *data)
                chunks.clear()

                yield tuple(map(torch.convert_to_tensor, data))
                self.batches_seen += 1
        
        def next(self) -> T:
            self._started = True
            while True:
                self._check_for_stop_iteration()

                # Fetch the next item's key from the weighted sampler
                key = next(self._weighted_sampler)
                try:
                    if self._datasets_exhausted[key] and self.stop_criteria == StopCriteria.ALL_DATASETS_EXHAUSTED:
                        # Before fetching a new item check if key corresponds to an already
                        # exhaused dataset and StopCriteria is ALL_DATASETS_EXHAUSTED, move to next key
                        continue
                    item = next(self.source_nodes[key])
                except StopIteration:
                    # Mark the dataset as exhausted
                    self._datasets_exhausted[key] = True

                    # Based on updated _check_for_stop_iteration, check if we should raise StopIteration
                    self._check_for_stop_iteration()

                    # If StopCriteria is ALL_DATASETS_EXHAUSTED, move to next key
                    if self.stop_criteria == StopCriteria.ALL_DATASETS_EXHAUSTED:
                        continue

                    # If StopCriteria is CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
                    # reset the iterator and try again
                    self.source_nodes[key].reset()
                    item = next(self.source_nodes[key])
                break

            # If we did't throw StopIteration, increment the number of items yielded and return the item
            self._num_yielded += 1
            return item

        

    class ListDataset(Dataset):

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