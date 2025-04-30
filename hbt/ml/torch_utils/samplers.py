from __future__ import annotations

__all__ = [
    "RowgroupSampler",
    "ListRowgroupSampler",
]

from functools import partial

from collections.abc import Container
from columnflow.util import MockModule, maybe_import
from columnflow.types import Any, Callable

from hbt.ml.torch_utils.datasets import ParquetDataset


torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")

RowgroupSampler = MockModule("RowgroupSampler")  # type: ignore


if not isinstance(torchdata, MockModule):
    from torch.utils.data import RandomSampler, SequentialSampler, Sampler, BatchSampler

    class RowgroupSampler(Sampler):  # noqa: F811
        def __init__(
            self,
            inputs: ParquetDataset | None = None,
            metadata: dict[str, Any] | None = None,
            rowgroup_list: Container[int] | None = None,
            rowgroup_sizes: Container[int] | None = None,
            shuffle_rowgroups: bool = False,
            shuffle_indices: bool = False,
            simultaneous_rowgroups: int = 1,
            replacement: bool = False,
            num_samples: int | None = None,
            generator=None,
        ):
            self.dataset = inputs
            self.metadata = metadata
            self.rowgroup_list = rowgroup_list
            self.rowgroup_sizes = rowgroup_sizes
            self.simultaneous_rowgroups = simultaneous_rowgroups
            self.shuffle_rowgroups = shuffle_rowgroups
            self.shuffle_indices = shuffle_indices

            # arguments for pass through to torch RandomSampler
            random_sampler_args = {
                "replacement": replacement,
                "num_samples": num_samples,
                "generator": generator,
            }
            self.rowgroup_sampler_cls = (
                partial(self.sampler_factory, cls=RandomSampler, **random_sampler_args)
                if self.shuffle_rowgroups else SequentialSampler
            )
            self.index_sampler_cls = (
                partial(self.sampler_factory, cls=RandomSampler, **random_sampler_args)
                if self.shuffle_indices else SequentialSampler
            )

            # if either the rowgroup_list or rowgroup_sizes are missing,
            # try to extrac them from the other information
            if rowgroup_list is None or rowgroup_sizes is None:
                self._extract_rowgroup_information()

            # make sure that the rowgroup list and size list are torch tensors
            # for smart indexing
            self.rowgroup_list = torch.tensor(self.rowgroup_list)
            self.rowgroup_sizes = torch.tensor(self.rowgroup_sizes)
            if self.simultaneous_rowgroups == -1:
                self.simultaneous_rowgroups = len(self.rowgroup_list)

        def sampler_factory(
            self,
            data: Any,
            cls: Callable = RandomSampler,
            replacement: bool = False,
            num_samples: int | None = None,
            generator=None,
        ):
            return cls(data, replacement=replacement, num_samples=num_samples, generator=generator)

        def _extract_rowgroup_information(self):
            if self.metadata is None and self.dataset is None:
                raise ValueError("Cannot extract rowgroup information: both metadata and dataset are missing!")
            # consider the dataset only if the metadata is missing
            if not self.metadata:
                self.metadata = self.dataset.meta_data
                # if the dataset is a FlatRowgroupParquetDataset, it also
                # has information about the allowed rowgroups, so retrieve
                # it in case of a missing list of rowgroups
                if not self.rowgroup_list:
                    self.rowgroup_list = getattr(self.dataset, "_allowed_rowgroups", None)

            if not self.rowgroup_list:
                self.rowgroup_list = list(range(self.metadata["num_row_groups"]))

            # make sure that rowgroup_list is a torch tensor
            self.rowgroup_list = torch.tensor(list(self.rowgroup_list))
            metadata_sizes = torch.tensor(list(self.metadata["col_counts"]))
            self.rowgroup_sizes = metadata_sizes[self.rowgroup_list]

        def __len__(self):
            return sum(self.rowgroup_sizes)

        def __iter__(self):
            rowgroup_idx_sampler = self.rowgroup_sampler_cls(self.rowgroup_list)
            for rg_idx in BatchSampler(rowgroup_idx_sampler, self.simultaneous_rowgroups, drop_last=False):
                rowgroups = self.rowgroup_list[rg_idx]
                sizes = self.rowgroup_sizes[rg_idx]

                size_sampler = self.index_sampler_cls(range(sum(sizes)))
                # from IPython import embed
                # embed(header=f"indices for rowgroup_sampler {rg_idx}")

                for data_idx in size_sampler:
                    yield (tuple(rowgroups.tolist()), data_idx)

    class ListRowgroupSampler(Sampler):

        def __init__(
            self,
            inputs: Container[ParquetDataset] | Container[RowgroupSampler],
            rowgroup_list: Container[int] | None = None,
            rowgroup_sizes: Container[int] | None = None,
            shuffle_rowgroups: bool = False,
            shuffle_indices: bool = False,
            shuffle_list: bool = False,
            simultaneous_rowgroups: int = 1,
            replacement: bool = False,
            num_samples: int | None = None,
            generator=None,
        ):

            self.samplers: list[RowgroupSampler] = list()
            forward_pass_args = {
                "rowgroup_list": rowgroup_list,
                "rowgroup_sizes": rowgroup_sizes,
                "shuffle_rowgroups": shuffle_rowgroups,
                "shuffle_indices": shuffle_indices,
                "simultaneous_rowgroups": simultaneous_rowgroups,
                "replacement": replacement,
                "num_samples": num_samples,
                "generator": generator,
            }
            if all(isinstance(x, ParquetDataset) for x in inputs):
                self.samplers = [RowgroupSampler(x, **forward_pass_args) for x in inputs]
            elif all(isinstance(x, RowgroupSampler) for x in inputs):
                self.samplers = list(inputs)

            self.list_idx_sampler_cls = SequentialSampler if not shuffle_list else RandomSampler

        def __len__(self):
            return sum(len(x) for x in self.samplers)

        def __iter__(self):
            for dataset_idx in self.list_idx_sampler_cls(self.samplers):
                sampler = self.samplers[dataset_idx]
                for output in sampler:
                    yield (dataset_idx, *output)
