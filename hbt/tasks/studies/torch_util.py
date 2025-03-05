from __future__ import annotations

__all__ = [
    "ListDataset", "MapAndCollate", "FlatMapAndCollate", "NodesDataLoader"
]

from collections import Iterable, Mapping, Collection
from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import T, Any, Callable, Sequence
from columnflow.columnar_util import (
    get_ak_routes, Route, remove_ak_column, EMPTY_FLOAT, EMPTY_INT,
    flat_np_view,
)
import copy

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")
ak = maybe_import("awkward")

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
    import re

    class ParquetDataset(Dataset):
        def __init__(
            self,
            input: Sequence[str] | ak.Array,
            columns: Sequence[str] | None = None,
            target: str | int | Iterable[str | int] | None = None,
            open_options: dict[str, Any] | None = None,
            transform: Callable | None = None,
        ):
            self.open_options = open_options or {}
            self.columns = columns or set()
            # container for target columns
            self.target_columns = set()
            # container for integer targets
            self.int_targets = set()
            self.transform = transform

            self.input = input
            self._data: ak.Array | None = None
            self._input_data: ak.Array | None = None
            self._target_data: ak.Array | None = None
            self.class_target: int

            
            # container for meta data of parquet file(s)
            # None if input is an ak.Array
            self.meta_data = None

            if isinstance(input, (str, list)):
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
            elif isinstance(input, ak.Array):
                self._data = input
            
            self.all_columns = set()
            self._parse_columns()
            
            self._parse_target(target=target)

            self._validate()

            if len(self.int_targets) > 0:
                self.class_target = list(self.int_targets)[0]
            self.data_columns = self.all_columns.symmetric_difference(self.target_columns)

            # parse all strings to Route objects
            self.data_columns: set[Route] = set(Route(x) for x in self.data_columns)
            self.target_columns: set[Route] = set(Route(x) for x in self.target_columns)
            self.all_columns: set[Route] = set(Route(x) for x in self.all_columns)

        def _parse_columns(self) -> None:
            if self.columns:
                self.columns = set(Route(x) for x in self.columns)
            elif isinstance(self.data, ak.Array):
                self.all_columns = set(x for x in get_ak_routes(self.data))

            if self.meta_data:
                self.all_columns = set(x.replace(".list.item", "") for x in self.meta_data.columns)
                # columns are not explicitely considered when loading the meta data
                # so filter the full set of columns accordingly
                tmp_cols = set()
                for x in self.all_columns:
                    for col in (self.columns or (".*",)):
                        resolved_route = self._check_against_pattern(x, col)
                        if resolved_route:
                            tmp_cols.add(resolved_route)
                
                self.all_columns = tmp_cols 
            
            if len(self.all_columns) == 0:
                raise ValueError("No columns specified and no metadata found")

        def _check_against_pattern(self, target: str, col: Route) -> Route | None:
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
            
            if not pattern.match(target):
                return
            # if there is a match, insert possible slices
            # from IPython import embed
            # embed(header=f"found match for target '{target}' and pattern '{col}'")
            parts = target.split(".")
            for index in reversed(slice_dict.keys()):
                parts.insert(index, slice_dict[index])
            
            return Route(Route.join(parts))


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

                if not any(self._check_against_pattern(target, col) for col in self.all_columns):
                    raise ValueError(f"target {target} not found in columns")
                
            # if target is an integer, this is a class index
            # this should be >= 0
            if any(target < 0 for target in self.int_targets):
                raise ValueError(f"int targets must be >= 0, received {self.int_targets}")
            if len(self.int_targets) > 1:
                raise ValueError("There cannot be more than one categorical target per dataset"
                                 f", received {self.int_targets}"
                )

        @property
        def data(self) -> ak.Array:
            if self._data is None:
                self.open_options["columns"] = [x.string_column for x in self.all_columns]
                self._data = ak.from_parquet(self.path, **self.open_options)
            return self._data

        @property
        def input_data(self) -> ak.Array:
            if self._input_data is None:
                self._input_data = self.data
                for col in self.target_columns:
                    self._input_data = remove_ak_column(self._input_data, col.string_column)
            return self._input_data
        
        @property
        def target_data(self) -> ak.Array:
            if self._target_data is None and len(self.target_columns) > 0:
                self._target_data = self.data
                for col in self.data_columns:
                    self._target_data = remove_ak_column(self._target_data, col.string_column)
            return self._target_data

        def __len__(self):
            return len(self.data)

        def _get_data(self, i: int| Sequence[int], input_data: ak.Array | None = None) -> ak.Array:
            data: ak.Array
            if input_data is None:
                data = self.input_data
            else:
                data = input_data
            return data[i]

        def _create_class_target(self, length: int, input_int_targets: int | None = None) -> ak.Array:
            int_target: int = input_int_targets or self.class_target

            return ak.Array([int_target]*int(length))
        
        def __getitem__(self, i: int | Sequence[int]) -> ak.Array | tuple[ak.Array, ak.Array] | tuple[ak.Array, ak.Array, ak.Array]:
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
                    
            if self.transform:
                return_data = self.transform(return_data)
            return tuple(return_data) if isinstance(return_data, list) else return_data
        
        def __getitems__(self, idx: Sequence[int]) -> ak.Array:
            return self.__getitem__(idx)
        
        def to_list(self) -> list[dict[str, Any]]:
            return self.data.to_list()


    class FlatParquetDataset(ParquetDataset):
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
                self._input_data = super().input_data
                self._input_data = {
                    str(r): self._extract_columns(self._input_data, r)
                    for r in self.data_columns
                }
            return self._input_data
        
        @property
        def target_data(self) -> Mapping[str, ak.Array]:
            if self._target_data is None:
                self._target_data = super().target_data
                self._target_data = {
                    str(r): self._extract_columns(self._target_data, r)
                    for r in self.target_columns
                }
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
                        "categorical_target": self._create_class_target(
                            ak.num(return_data[0][first_key], axis=0), input_int_targets=self.class_target
                        )
                    })
                    
            if self.transform:
                return_data = self.transform(return_data)
            return tuple(return_data) if isinstance(return_data, list) else return_data

    class BatchedMultiNodeWeightedSampler(MultiNodeWeightedSampler):

        def __init__(
            self,
            *args,
            batch_size: int,
            weights: dict[str, float | dict[str, float]],  # type: ignore
            drop_last: bool = False,
            **kwargs
        ):
            self.batch_size = batch_size
            self.drop_last = drop_last
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
            for key, weight in self.weights.items():

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
            self._batch_composition: dict[str, int] = {
                key: int(weight*self.batch_size // total_weight_sum
                    if isinstance(weight, (int, float))
                    else sum(weight.values())*self.batch_size // total_weight_sum
                )
                for key, weight in self.weights.items()
            }
            
            # due to the integer division above, the sum of the batch composition
            # might not add up to the requested batch size. In this case, we adjust
            # the batch size to the sum of the batch composition
            _real_total_size = sum(self._batch_composition.values())
            if _real_total_size != self.batch_size:
                print("Warning: requested batch size is not equal to the sum of the computed batch composition sizes. "
                      f"Adjusting batch size from {self.batch_size} to {_real_total_size}")
                self.batch_size = _real_total_size
            # default dictionary to store weighted samplers where necessary
            self._weighted_sampler = DotDict()
            
        def _get_new_weighted_sampler(self, initial_state=None) -> DotDict[str, _WeightedSampler]:
            _weighted_sampler = DotDict()
            for key in self._weight_samplers:
                initial_sampler_state = None
                if isinstance(initial_state, dict):
                    initial_sampler_state = initial_state[self.WEIGHTED_SAMPLER_STATE_KEY].get(key, None)
                _weighted_sampler[key] = _WeightedSampler(
                    weights=self.weights[key],
                    seed=self.seed,
                    rank=self.rank,
                    world_size=self.world_size,
                    epoch=self._epoch,
                    initial_state=initial_sampler_state,
                    # explicitely give size of random numbers to draw to ensure
                    # that the sub batch composition adds up correctly
                    random_tensor_batch_size=self._batch_composition[key],
                )
            return _weighted_sampler

        
        def _validate(self) -> None:
            if self.stop_criteria not in [
                StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
                StopCriteria.ALL_DATASETS_EXHAUSTED,
                StopCriteria.FIRST_DATASET_EXHAUSTED,
            ]:
                raise ValueError(
                    f"Invalid {self.stop_criteria=}. stop_criteria must be one of: CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED, FIRST_DATASET_EXHAUSTED, ALL_DATASETS_EXHAUSTED"
                )
            
            if not isinstance(self.batch_size, int) and not self.batch_size >= 1:
                raise ValueError(f"batch_size argument must be >= 1, received {self.batch_size}")

            if not isinstance(self.drop_last, bool):
                raise ValueError(f"drop_last argument must be a boolean, received {self.drop_last}")

            def _weight_check(weight):
                if not isinstance(weight, float) or weight <= 0:
                    raise ValueError(
                        f"""Invalid {self.weights=}. For multi-dataset weighted sampling, weights must be a 1d sequence, non-negative, and non-zero.
                        Weights are used to sample from source nodes. Zero weight means the source node will never be sampled from, and can cause
                        unexpected behavior depending on the stop criteris. Weights are used as inputs to torch.multinomial, please refer to
                        https://pytorch.org/docs/stable/generated/torch.multinomial.html on how to use weights for sampling."""
                    )

            all_keys = set(self.weights.keys())
            for key, weight in self.weights.items():
                if isinstance(weight, dict):
                    all_keys.remove(key)
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
                

        # def next_tautauNN(self) -> T:
        #     # prepare indices for random sampling
        #     indices = [np.array([], dtype=np.int32) for _ in range(self.n_datasets)]
        #     offsets = [0] * self.n_datasets

        #     # start iterating
        #     while True:
        #         # determine batch sizes per dataset for this chunk
        #         batch_sizes = torch.multinomial(self.batch_size, self.weights.values)

        #         # fill chunks per dataset that eventually form a batch
        #         chunks = []
        #         for i, (arrays, _indices, batch_size, offset) in enumerate(zip(
        #                 self.source_nodes.values, indices, batch_sizes, offsets
        #             )):
        #             # update indices and offset
        #             if len(_indices) - offset < batch_size:
        #                 new_indices = np.arange(len(arrays[0]), dtype=np.int32)
        #                 np.random.shuffle(new_indices)
        #                 _indices = indices[i] = np.concatenate([_indices[offset:], new_indices], axis=0)
        #                 offset = 0

        #             # fill the chunk and adjust the offset
        #             chunks.append([a[_indices[offset:offset + batch_size]] for a in arrays])
        #             offsets[i] = offset + batch_size

        #         # yield
        #         data = tuple(
        #             np.concatenate([chunk[i] for chunk in chunks], axis=0)
        #             for i in range(self.tuple_length)
        #         )
        #         data = transform_data(self, *data)
        #         chunks.clear()

        #         yield tuple(map(torch.convert_to_tensor, data))
        #         self.batches_seen += 1
        
        def _next_per_dataset(self, key: str, force: bool = False):
            # print(f"entering _next_per_dataset for node {key}")
            item = None
            try:
                if not(self._datasets_exhausted[key] and self.stop_criteria == StopCriteria.ALL_DATASETS_EXHAUSTED):
                    # Before fetching a new item check if key corresponds to an already
                    # exhaused dataset and StopCriteria is ALL_DATASETS_EXHAUSTED, move to next key
                    item = next(self.source_nodes[key])
            except StopIteration as e:
                # Mark the dataset as exhausted
                self._datasets_exhausted[key] = True

                # Based on updated _check_for_stop_iteration, check if we should raise StopIteration
                # optionally disable this in case an update is needed regardless of external criteria
                if not force:
                    self._check_for_stop_iteration()

                # If StopCriteria is ALL_DATASETS_EXHAUSTED, move to next key
                if self.stop_criteria == StopCriteria.ALL_DATASETS_EXHAUSTED and not force:
                    return

                # If StopCriteria is CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED,
                # reset the iterator and try again
                self.source_nodes[key].reset()
                item = next(self.source_nodes[key])
            # from IPython import embed
            # embed(header=f"obtained item for node {key}")
            return item

        def next(self) -> dict[str, list[T]]:
            self._started = True

            self._check_for_stop_iteration()

            batch = dict()
            for source_name in self.weights:
                batch_size = self._batch_composition[source_name]
                key = source_name
                sub_batch = list()
                for _ in range(batch_size):
                    sampler = self._weighted_sampler.get(source_name, None)
                    if sampler:
                        key = next(sampler)
                    try:
                        item = self._next_per_dataset(key)
                        if item is not None:
                            sub_batch.append(item)
                    except StopIteration as e:
                        if not self.drop_last and len(sub_batch) > 0 and self.stop_criteria == StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED:
                            item = self._next_per_dataset(key, force=True)
                            if item is not None:
                                sub_batch.append(item)
                        elif self.drop_last:
                            self._check_for_stop_iteration()
                        # in this case, the stop criteria (e.g. ALL_DATASETS_EXHAUSTED)
                        # are met and we can break the loop
                        if not self.stop_criteria == StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED:
                            break
                
                # if stop criterium is ALL_DATASETS_EXHAUSTED, allow for partial batches
                if len(sub_batch) == batch_size or self.stop_criteria == StopCriteria.ALL_DATASETS_EXHAUSTED:
                    batch[source_name] = sub_batch
            
            # if the batch is not completely full, check if we should raise a StopIteration
            if sum((len(x) for x in batch.values())) < self.batch_size and not self.stop_criteria == StopCriteria.ALL_DATASETS_EXHAUSTED:
                # at this point
                # StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED should produce a full batch
                # StopCriteria.FIRST_DATASET_EXHAUSTED should have already raised a StopIteration
                from IPython import embed
                embed(header="DANGERZONE: batch is not full")
                raise StopIteration()

            
            # # check again that the datasets have something left to give
            # for source_name in self.source_nodes:
            #     # skip check if dataset is already marked as exhausted
            #     if self._datasets_exhausted[source_name]:
            #         continue
            #     dataset = self.source_nodes[source_name]
            #     dataset_state = dataset.state_dict()
            #     # if the dataset has already yielded all items, mark it as exhausted
            #     self._datasets_exhausted = dataset_state[dataset.NUM_YIELDED_KEY] == len(dataset)
            # If we did't throw StopIteration, increment the number of items yielded and return the item
            self._num_yielded += 1
            return batch

        def get_state(self) -> dict[str, Any]:
            return {
                self.DATASETS_EXHAUSTED_KEY: copy.deepcopy(self._datasets_exhausted),
                self.DATASET_NODE_STATES_KEY: {k: self.source_nodes[k].state_dict() for k in self.dataset_names},
                self.EPOCH_KEY: self._epoch,
                self.NUM_YIELDED_KEY: self._num_yielded,
                self.WEIGHTED_SAMPLER_STATE_KEY: {k: self._weighted_sampler[k].state_dict() for k in self._weight_samplers},
            }
        

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
        
        def __init__(self,
            dataset: Sized,
            collate_fn: Callable,
        ):
            self.dataset = dataset
            self.collate_fn = collate_fn
            
        def __call__(self, batch_of_indices: list[int]):
            batch = [self.dataset[i] for i in batch_of_indices]
            return self.collate_fn(batch)
        
    class FlatMapAndCollate(MapAndCollate):
        """A simple transform that takes a batch of indices, maps with dataset, and then applies
        collate.
        TODO: make this a standard utility in torchdata.nodes
        """
        def __call__(self, idx: int):
            batch = self.dataset[idx]
            return self.collate_fn(batch)
        
    class NestedMapAndCollate(MapAndCollate):
        def __init__(self,
            dataset: dict[str, Sized],
            collate_fn: Callable | None = None,
        ):
            self.dataset = dataset
            self.collate_fn: Callable = collate_fn or self._default_collate

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

        def _default_collate(self, idx: dict[str, Sequence[int]]) -> Sequence[T]:
            batch: list[T] = []

            # helper function to concatenate different types of objects
            

            for key, indices in idx.items():
                current_batch = self.dataset[key][indices]
                concat_fn = ak.concatenate
                if isinstance(current_batch, (list, tuple)):
                    if all(isinstance(x, torch.Tensor) for x in current_batch):
                        concat_fn = torch.cat
                elif isinstance(current_batch, torch.Tensor):
                    concat_fn = torch.cat
                
                batch = self._concat_batches(batch=batch, current_batch=current_batch, concat_fn=concat_fn)
            
            return batch

        def __call__(self, idx: dict[str, Sequence[int]]) -> Sequence[T]:
            
            return self.collate_fn(idx)
        
    
    class NestedDictMapAndCollate(NestedMapAndCollate):
        def _default_collate(self, idx: dict[str, Sequence[int]]) -> Sequence[T]:
            batch: list[T] = []

            # helper function to concatenate different types of objects
            def _concat_dicts(
                input_arrays: Sequence[dict[str, T]],
                *args,
                **kwargs,
            ) -> dict[str, T]:
                return_dict = dict()
                first_dict = input_arrays[0]
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

                return return_dict


            for key, indices in idx.items():
                current_batch = self.dataset[key][indices]
                batch = self._concat_batches(batch=batch, current_batch=current_batch, concat_fn=_concat_dicts)
            
            return batch

    # To keep things simple, let's assume that the following args are provided by the caller
    def NodesDataLoader(
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

    class CompositeDataLoader(object):
    
        def __init__(
                self,
                data_map: Mapping[str, Sized],
                weight_dict: Mapping[str, float | Mapping[str, float]],
                shuffle: bool=True,
                batch_size: int = 256,
                num_workers: int = 0,
                parallelize_method: Literal["thread", "process"] = "process",
                collate_fn: Callable | None = None,
                batch_sampler_cls: Callable | None = None,
                index_sampler_cls: Callable | None = None,
                map_and_collate_cls: Callable | None = None,
                device=None,
        ):
            
            self.data_map = data_map
            self.weight_dict = weight_dict
            self.shuffle = shuffle
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.parallelize_method = parallelize_method
            self.collate_fn = collate_fn
            self.batch_sampler_cls = batch_sampler_cls
            self.index_sampler_cls = index_sampler_cls
            self.map_and_collate_cls = map_and_collate_cls
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

        def _create_composite_node(self) -> tuple[tn.ParallelMapper, _WeightedSampler]:

            node_dict = {
                key: tn.SamplerWrapper(self.index_sampler_cls(dataset))
                for key, dataset in self.data_map.items()
            }
            
            batcher = self.batch_sampler_cls(
                node_dict, weights=self.weight_dict, batch_size=self.batch_size,
            )

            mapping = self.map_cls(self.data_map, collate_fn=self.collate_fn)
            
            parallel_node = tn.ParallelMapper(
                batcher,
                map_fn=mapping,
                num_workers=self.num_workers,
                method=self.parallelize_method,  # Set this to "thread" for multi-threading
                in_order=True,
            )

            return (parallel_node, batcher)
        
        def __len__(self):
            return sum(len(x) for x in self.data_map.values())
        
        @property
        def num_batches(self):
            if not self._num_batches:
                datasets: list[ParquetDataset] = list(self.data_map.values())
                dataset_names = list(self.data_map.keys())
                max_dataset_idx: int = np.argmax([len(data) for data in datasets])
                max_composition: int = self.batcher._batch_composition[dataset_names[max_dataset_idx]]
                self._num_batches = int(len(datasets[max_dataset_idx]) / max_composition)
            return self._num_batches