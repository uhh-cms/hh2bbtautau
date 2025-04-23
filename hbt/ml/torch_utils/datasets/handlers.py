from __future__ import annotations
from functools import partial
from copy import deepcopy
import abc

from columnflow.util import maybe_import
from columnflow.columnar_util import Route
from columnflow.types import Any, Callable
from collections.abc import Collection
from hbt.ml.torch_utils.dataloaders import CompositeDataLoader
from hbt.ml.torch_utils.datasets import (
    ParquetDataset, FlatRowgroupParquetDataset, FlatParquetDataset,
    FlatArrowRowGroupParquetDataset, WeightedFlatRowgroupParquetDataset,
)
from hbt.ml.torch_utils.map_and_collate import (
    NestedListRowgroupMapAndCollate, FlatListRowgroupMapAndCollate,
    NestedDictMapAndCollate,
)

from hbt.ml.torch_utils.samplers import ListRowgroupSampler, RowgroupSampler


torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")
ak = maybe_import("awkward")

tn = maybe_import("torchdata.nodes")
import law
logger = law.logger.get_logger(__name__)


class BaseParquetFileHandler(object):
    dataset_cls: Callable
    training_map_and_collate_cls: Callable
    validation_map_and_collate_cls: Callable
    training_sampler_cls: Callable
    validation_sampler_cls: Callable

    def __init__(
        self,
        *args,
        task: law.Task,
        open_options: dict[str, Any] | None = None,
        columns: Collection[str | Route] | None = None,
        datasets: Collection[str] | None = None,
        batch_transformations: torch.nn.Module | None = None,
        global_transformations: torch.nn.Module | None = None,
        categorical_target_transformation: torch.nn.Module | None = None,
        data_type_transform: torch.nn.Module | None = None,
        preshuffle: bool = True,
        build_categorical_target_fn: Callable | None = None,
        group_datasets: dict[str, list[str]] | None = None,
        device: str | None = None,
        **kwargs,
    ):
        self.open_options = open_options or dict()
        self.task = task
        self.inputs = task.input()
        self.configs = task.configs
        self.datasets = datasets or task.datasets
        self.load_parallel_cores = task.load_parallel_cores
        self.batch_size = task.batch_size
        self.columns = columns
        self.weight_dict = dict()

        self.batch_transformations = batch_transformations
        self.global_transformations = global_transformations
        self.categorical_target_transformation = categorical_target_transformation
        self.data_type_transform = data_type_transform
        self.preshuffle = preshuffle
        self.build_categorical_target_fn = build_categorical_target_fn or self._default_build_categorical_target_fn

        self.group_datasets = group_datasets or dict()
        self.dataset_cls = FlatParquetDataset
        self.device = device

    def _default_build_categorical_target_fn(self, dataset: str) -> int:
        return int(1) if dataset.startswith("hh") else int(0)

    def _split_pq_dataset_per_path(
        self,
        target_path,
        ratio=0.7,
        targets: Collection[str | int | Route] | str | Route | int | None = None,
    ):
        meta = ak.metadata_from_parquet(target_path)

        total_row_groups = meta["num_row_groups"]
        rowgroup_indices_func = torch.randperm if self.preshuffle else np.arange
        rowgroup_indices: list[int] = rowgroup_indices_func(total_row_groups).tolist()
        max_training_group = int(total_row_groups * ratio)
        training_row_groups = None
        if max_training_group == 0:
            logger.warning(
                "Could not split into training and validation data"
                f" number of row groups for '{target_path}' is  {total_row_groups}",
            )
        else:
            training_row_groups = rowgroup_indices[:max_training_group]

        final_options = self.open_options or dict()
        final_options.update({"row_groups": training_row_groups})

        dataset_kwargs = {
            "columns": self.columns,
            "target": targets,
            "batch_transform": self.batch_transformations,
            "global_transform": self.global_transformations,
            "categorical_target_transform": self.categorical_target_transformation,
            "data_type_transform": self.data_type_transform,
        }

        logger.info(f"Constructing training dataset for {target_path} with row_groups {training_row_groups}")

        training = self.dataset_cls(
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
            validation = self.dataset_cls(
                target_path,
                open_options=final_options,
                **dataset_kwargs,
            )
        return training, validation

    def split_training_validation(
        self,
        target_paths,
        ratio=0.7,
        targets: Collection[str | int | Route] | str | Route | int | None = None,
    ):
        training_ds = list()
        validation_ds = list()
        for path in target_paths:
            training, validation = self._split_pq_dataset_per_path(
                target_path=path,
                ratio=ratio,
                targets=targets,
            )
            training_ds.append(training)
            validation_ds.append(validation)
        return training_ds, validation_ds

    def _extract_probability(
        self,
        dataset: str, keyword: str = "sum_mc_weight_selected",
    ):
        expected_events = list()
        sel_stat = self.inputs.selection_stats
        for config in self.configs:
            config_inst = self.task.analysis_inst.get_config(config)
            lumi = config_inst.x.luminosity.nominal

            # following code is used for SiblingFileCollections
            # target = sel_stat[dataset][config].collection[0]["stats"]

            # following code is used for LocalFileTargets
            target = sel_stat[dataset][config]["stats"]
            stats = target.load(formatter="json")
            xs = stats.get(keyword, 0)
            expected_events.append(xs * lumi)
        return sum(expected_events)

    def sampler_factory(
        self,
        datasets,
        cls: Callable = ListRowgroupSampler,
        shuffle_rowgroups=False,
        shuffle_indices=False,
        simultaneous_rowgroups=1,
    ):
        return cls(
            inputs=datasets,
            shuffle_rowgroups=shuffle_rowgroups,
            shuffle_indices=shuffle_indices,
            simultaneous_rowgroups=simultaneous_rowgroups,
        )

    def _get_weights(self):
        self.weight_dict: dict[str, float | dict[str, float]] = {
            d: 1. for d in self.datasets if not any(d in x for x in self.group_datasets.values())
        }
        for key, subspace in self.group_datasets.items():
            subspace_probs = {
                d: self._extract_probability(d) for d in subspace
            }
            prob_sum = sum(subspace_probs.values())
            self.weight_dict[key] = {
                d: val / prob_sum for d, val in subspace_probs.items()
            }

    def _init_training_validation_map(self) -> tuple[dict[str, list[ParquetDataset]], dict[str, list[ParquetDataset]]]:
        # construct datamap
        training_data_map: dict[str, list[ParquetDataset]] = dict()
        validation_data_map: dict[str, list[ParquetDataset]] = dict()

        for dataset in self.datasets:
            # following code is used for SiblingFileCollections
            targets = [self.inputs.events[dataset][c]["collection"] for c in self.configs]
            target_paths = [
                t.abspath
                for collections in targets
                for targets in collections.targets.values()
                for t in targets.values()
            ]

            # following code is used for LocalFileTargets
            # from IPython import embed
            # embed(header=f"creating data loader for target {dataset}")
            # targets = [self.inputs.events[dataset][c]["events"] for c in self.configs]
            # target_paths = [t.abspath for t in targets]

            training, validation = self.split_training_validation(
                target_paths=target_paths,
                targets=self.build_categorical_target_fn(dataset),
            )

            # read this in a gready way
            training_data_map[dataset] = training
            validation_data_map[dataset] = validation
        return training_data_map, validation_data_map

    def _create_validation_dataloader(self, validation_data_map):
        # create merged validation dataset since it's ok to simply evaluate the
        # events one by one
        validation_data: list[ParquetDataset] = list()

        for key, x in validation_data_map.items():
            if not isinstance(x, (list, tuple, set)):
                validation_data.append(x)
            else:
                validation_data.extend(x)

        return CompositeDataLoader(
            validation_data,
            batch_sampler_cls=tn.Batcher,
            shuffle=False,
            batch_size=self.batch_size,
            batcher_options={
                "source": tn.SamplerWrapper(self.sampler_factory(validation_data, cls=self.validation_sampler_cls)),
            },
            map_and_collate_cls=self.validation_map_and_collate_cls,
            device=self.device,
            # collate_fn=lambda x: x,
        )

    def init_datasets(self) -> tuple[CompositeDataLoader, CompositeDataLoader]:
        # construct datamaps
        training_data_map, validation_data_map = self._init_training_validation_map()

        # calculate weights for sub sampling
        self._get_weights()

        # overload factory with suitable settings for training
        sampler_fn = partial(
            self.sampler_factory,
            cls=self.training_sampler_cls,
            shuffle_rowgroups=True,
            shuffle_indices=True,
        )

        # create loader for training data
        training_composite_loader = CompositeDataLoader(
            data_map=training_data_map,
            weight_dict=self.weight_dict,
            map_and_collate_cls=self.training_map_and_collate_cls,
            batch_size=self.batch_size,
            index_sampler_cls=sampler_fn,
            num_workers=self.load_parallel_cores,
            device=self.device,
        )

        # create loader for validation data
        validation_composite_loader = self._create_validation_dataloader(validation_data_map)
        return (training_composite_loader, validation_composite_loader)


class FlatListRowgroupParquetFileHandler(BaseParquetFileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_cls = FlatRowgroupParquetDataset
        self.training_map_and_collate_cls = NestedListRowgroupMapAndCollate
        self.validation_map_and_collate_cls = FlatListRowgroupMapAndCollate
        self.training_sampler_cls = ListRowgroupSampler
        self.validation_sampler_cls = ListRowgroupSampler


class WeightedFlatListRowgroupParquetFileHandler(FlatListRowgroupParquetFileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_cls = WeightedFlatRowgroupParquetDataset
        # self.validation_dataset_cls = WeightedFlatRowgroupParquetDataset
        self.training_map_and_collate_cls = NestedListRowgroupMapAndCollate
        self.validation_map_and_collate_cls = FlatListRowgroupMapAndCollate
        self.training_sampler_cls = ListRowgroupSampler
        self.validation_sampler_cls = ListRowgroupSampler

    def _create_validation_dataloader(self, validation_data_map):
        # calculate final weights
        total_events = 0
        for key, weight in self.weight_dict.items():

            if isinstance(weight, dict):
                total_events = sum(len(x) for k in weight.keys() for x in validation_data_map[k])
                # also account for sum of sub weights such that total sum is 1
                for k, v in weight.items():
                    for d in validation_data_map[k]:
                        d.cls_weight = v / total_events
            else:
                total_events = sum(len(x) for x in validation_data_map[key])
                for d in validation_data_map[key]:
                    d.cls_weight = weight / total_events

        return super()._create_validation_dataloader(validation_data_map)

    def init_datasets(self) -> tuple[CompositeDataLoader, CompositeDataLoader]:

        training_data_map, validation_data_map = self._init_training_validation_map()

        self._get_weights()

        sampler_fn = partial(
            self.sampler_factory,
            cls=self.training_sampler_cls,
            shuffle_rowgroups=True,
            shuffle_indices=True,
        )

        training_composite_loader = CompositeDataLoader(
            data_map=training_data_map,
            weight_dict=self.weight_dict,
            map_and_collate_cls=self.training_map_and_collate_cls,
            batch_size=self.batch_size,
            index_sampler_cls=sampler_fn,
            num_workers=self.load_parallel_cores,
            device=self.device,
        )

        # create merged validation dataset
        train_val_composite_loader = self._create_validation_dataloader(training_data_map)
        validation_composite_loader = self._create_validation_dataloader(validation_data_map)

        return (training_composite_loader, (train_val_composite_loader, validation_composite_loader))


class FlatArrowParquetFileHandler(BaseParquetFileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_cls = FlatArrowRowGroupParquetDataset
        self.training_map_and_collate_cls = NestedDictMapAndCollate
        self.validation_map_and_collate_cls = FlatListRowgroupMapAndCollate
        self.training_sampler_cls = RowgroupSampler
        self.validation_sampler_cls = ListRowgroupSampler

    def split_training_validation(
        self,
        target_paths,
        ratio=0.7,
        targets: Collection[str | int | Route] | str | Route | int | None = None,
    ):

        dataset_kwargs = {
            "columns": self.columns,
            "target": targets,
            "batch_transform": self.batch_transformations,
            "global_transform": self.global_transformations,
            "categorical_target_transform": self.categorical_target_transformation,
            "data_type_transform": self.data_type_transform,
        }
        final_options = self.open_options or dict()

        training = self.dataset_cls(
            target_paths,
            open_options=final_options,
            **dataset_kwargs,
        )

        total_row_groups = training.meta_data["num_row_groups"]
        rowgroup_indices_func = torch.randperm if self.preshuffle else np.arange
        rowgroup_indices: list[int] = rowgroup_indices_func(total_row_groups).tolist()
        max_training_group = int(total_row_groups * ratio)
        training_row_groups = None
        if max_training_group == 0:
            logger.warning(
                "Could not split into training and validation data"
                f" number of row groups for training is  {total_row_groups}",
            )
        else:
            training_row_groups = rowgroup_indices[:max_training_group]

        training._allowed_rowgroups = set(training_row_groups)
        logger.info(f"Constructing training dataset with row_groups {training_row_groups}")

        validation = None
        if training_row_groups is None:
            validation = training
        else:
            validation_row_groups = rowgroup_indices[max_training_group:]
            final_options.update({"row_groups": validation_row_groups})
            logger.info(f"Constructing validation dataset with row_groups {validation_row_groups}")
            validation = deepcopy(training)
            validation._allowed_rowgroups = set(validation_row_groups)
        return training, validation


class DatasetHandlerMixin:
    parameters: dict[str, torch.Tensor]
    dataset_handler: BaseParquetFileHandler
    training_loader: CompositeDataLoader
    validation_loader: CompositeDataLoader

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def _build_categorical_target(self, dataset: str) -> Any:
        pass

    @abc.abstractmethod
    def init_dataset_handler(self, task: law.Task):
        pass

    def init_datasets(self):
        self.training_loader, self.validation_loader = self.dataset_handler.init_datasets()
