from __future__ import annotations

__all__ = [
    "BatchedMultiNodeWeightedSampler",
]

from collections import defaultdict
from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import T, Any

from hbt.ml.torch_utils.utils import reorganize_idx
import copy

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")
ak = maybe_import("awkward")

BatchedMultiNodeWeightedSampler = MockModule("BatchedMultiNodeWeightedSampler")

if not isinstance(torchdata, MockModule):
    from torchdata.nodes import MultiNodeWeightedSampler
    from torchdata.nodes.samplers.stop_criteria import StopCriteria
    from torchdata.nodes.samplers.multi_node_weighted_sampler import _WeightedSampler

    class BatchedMultiNodeWeightedSampler(MultiNodeWeightedSampler):

        def __init__(
            self,
            *args,
            batch_size: int,
            weights: dict[str, float | dict[str, float]],  # type: ignore
            drop_last: bool = False,
            **kwargs,
        ):
            self.batch_size = batch_size
            self.drop_last = drop_last
            super().__init__(*args, weights=weights, **kwargs)
            self._datasets_exhausted = {n: False for n in self.source_nodes}

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
                key: int(
                    weight * self.batch_size // total_weight_sum
                    if isinstance(weight, (int, float))
                    else sum(weight.values()) * self.batch_size // total_weight_sum,
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
            self._weighted_sampler = self._get_new_weighted_sampler()

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
                    f"Invalid {self.stop_criteria=}. stop_criteria must be one of: "
                    "CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED, FIRST_DATASET_EXHAUSTED, ALL_DATASETS_EXHAUSTED",
                )

            if not isinstance(self.batch_size, int) and not self.batch_size >= 1:
                raise ValueError(f"batch_size argument must be >= 1, received {self.batch_size}")

            if not isinstance(self.drop_last, bool):
                raise ValueError(f"drop_last argument must be a boolean, received {self.drop_last}")

            def _weight_check(weight):
                if not isinstance(weight, float) or weight <= 0:
                    raise ValueError(
                        f"""Invalid {self.weights=}. For multi-dataset weighted sampling, weights must be a 1d sequence,
                        non-negative, and non-zero.
                        Weights are used to sample from source nodes. Zero weight means the source node will never be
                        sampled from, and can cause unexpected behavior depending on the stop criteria.
                        Weights are used as inputs to torch.multinomial, please refer to
                        https://pytorch.org/docs/stable/generated/torch.multinomial.html on how to use weights for
                        sampling.
                        """,
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
                    ", ".join(difference),
                )

        def reset(self, initial_state: dict[str, Any] | None = None):
            super().reset(initial_state)
            # the super class uses the weights dict to initialize the exhausted datasets
            # but this class needs to go via the source nodes. Therefore, we need to
            # reinitialize the exhausted datasets
            if not initial_state:
                self._datasets_exhausted = {n: False for n in self.source_nodes.keys()}

        def _next_per_dataset(self, key: str, force: bool = False):
            # print(f"entering _next_per_dataset for node {key}")
            item = None
            try:
                if not (self._datasets_exhausted[key] and self.stop_criteria == StopCriteria.ALL_DATASETS_EXHAUSTED):
                    # Before fetching a new item check if key corresponds to an already
                    # exhaused dataset and StopCriteria is ALL_DATASETS_EXHAUSTED, move to next key
                    item = next(self.source_nodes[key])
            except StopIteration:
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

            batch = defaultdict(list)
            for source_name in self.weights:
                batch_size = self._batch_composition[source_name]
                key = source_name
                sub_batch: dict[str, list[float]] = defaultdict(list)
                for _ in range(batch_size):
                    sampler = self._weighted_sampler.get(source_name, None)
                    if sampler:
                        key = next(sampler)
                    try:
                        item = self._next_per_dataset(key)
                        if item is not None:
                            sub_batch[key].append(item)
                    except StopIteration:
                        if (
                            not self.drop_last and
                            len(sub_batch) > 0 and
                            self.stop_criteria == StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED
                        ):
                            item = self._next_per_dataset(key, force=True)
                            if item is not None:
                                sub_batch[key].append(item)
                        elif self.drop_last:
                            self._check_for_stop_iteration()
                        # in this case, the stop criteria (e.g. ALL_DATASETS_EXHAUSTED)
                        # are met and we can break the loop
                        if not self.stop_criteria == StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED:
                            break

                # if stop criterium is ALL_DATASETS_EXHAUSTED, allow for partial batches
                if (
                    sum(len(x) for x in sub_batch.values()) == batch_size or
                    self.stop_criteria == StopCriteria.ALL_DATASETS_EXHAUSTED
                ):
                    batch.update(sub_batch)

            # if the batch is not completely full, check if we should raise a StopIteration
            if (
                sum((len(x) for x in batch.values())) < self.batch_size and
                not self.stop_criteria == StopCriteria.ALL_DATASETS_EXHAUSTED
            ):
                # at this point
                # StopCriteria.CYCLE_UNTIL_ALL_DATASETS_EXHAUSTED should produce a full batch
                # StopCriteria.FIRST_DATASET_EXHAUSTED should have already raised a StopIteration
                from IPython import embed
                embed(header="DANGERZONE: batch is not full")
                raise StopIteration()

            batch = reorganize_idx(batch)

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
                self.WEIGHTED_SAMPLER_STATE_KEY: {
                    k: self._weighted_sampler[k].state_dict()
                    for k in self._weight_samplers
                },
            }
