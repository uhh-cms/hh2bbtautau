from __future__ import annotations

__all__ = [
    "NodesDataLoader", "CompositeDataLoader",
]

from functools import partialmethod

from collections.abc import Mapping, Container
from columnflow.util import MockModule, maybe_import
from columnflow.types import Any, Callable

from hbt.ml.torch_utils.datasets import ParquetDataset


torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")

RowgroupSampler = MockModule("RowgroupSampler")  # type: ignore


if not isinstance(torchdata, MockModule):
    from torch.utils.data import RandomSampler, SequentialSampler, Sampler, BatchSampler

    class RowgroupSampler(Sampler):
        def __init__(self,
            dataset: ParquetDataset | None = None,
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
            self.dataset = dataset
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
                partialmethod(RandomSampler, **random_sampler_args)
                if self.shuffle_rowgroups else SequentialSampler
            )
            self.index_sampler_cls = (
                partialmethod(RandomSampler, **random_sampler_args)
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
                for data_idx in size_sampler:
                    yield (rowgroups, data_idx)

