from __future__ import annotations

__all__ = [
    "split_pq_dataset_per_path",
]

from collections import Iterable, Mapping, Collection, defaultdict
from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import T, Any, Callable, Sequence
from columnflow.columnar_util import (
    get_ak_routes, Route, remove_ak_column, EMPTY_FLOAT, EMPTY_INT,
    flat_np_view,
)
from hbt.ml.torch_utils.datasets import FlatParquetDataset
import copy
import law

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


def split_pq_dataset_per_path(
    target_path,
    ratio=0.7,
    open_options: dict | None = None,
    columns: Collection[str | Route] | None = None,
    targets: Collection[str | int | Route] | str | Route | int | None = None,
    batch_transformations: torch.nn.Module | None = None,
    global_transformations: torch.nn.Module | None = None,
    categorical_target_transformation: torch.nn.Module | None = None,
    data_type_transform: torch.nn.Module | None = None,
    preshuffle: bool = True,
    dataset_cls: Callable or None = None
):
    meta = ak.metadata_from_parquet(target_path)
    dataset_cls = dataset_cls or FlatParquetDataset
            
    total_row_groups = meta["num_row_groups"]
    rowgroup_indices_func = torch.randperm if preshuffle else np.arange
    rowgroup_indices: list[int] = rowgroup_indices_func(total_row_groups).tolist()
    max_training_group = int( total_row_groups*ratio)
    training_row_groups = None
    if max_training_group == 0:
        logger.warning(
            "Could not split into training and validation data"
            f" number of row groups for '{target_path}' is  {total_row_groups}"
        )
    else:
        training_row_groups = rowgroup_indices[:max_training_group]

    final_options = open_options or dict()
    final_options.update({"row_groups": training_row_groups})

    dataset_kwargs = {
        "columns": columns,
        "target": targets,
        "batch_transform": batch_transformations,
        "global_transform": global_transformations,
        "categorical_target_transform": categorical_target_transformation,
        "data_type_transform": data_type_transform,
    }

    logger.info(f"Constructing training dataset for {target_path} with row_groups {training_row_groups}")

    training = dataset_cls(
        target_path,
        open_options=final_options,
        **dataset_kwargs,
    )

    validation = None
    if training_row_groups is None:
        validation = training
    else:
        validation_row_groups = rowgroup_indices[max_training_group:]
        final_options.update({"row_groups": validation_row_groups})
        logger.info(f"Constructing validation dataset for {target_path} with row_groups {validation_row_groups}")
        validation = dataset_cls(
            target_path,
            open_options=final_options,
            **dataset_kwargs,
        )
    return training, validation
