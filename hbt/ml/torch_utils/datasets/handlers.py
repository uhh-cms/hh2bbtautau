from __future__ import annotations
from functools import partial
from copy import deepcopy
import abc

from columnflow.util import maybe_import
from columnflow.columnar_util import Route
from columnflow.types import Any, Callable
from collections.abc import Collection, Container
from hbt.ml.torch_utils.dataloaders import CompositeDataLoader
from hbt.ml.torch_utils.datasets import (
    ParquetDataset, FlatRowgroupParquetDataset, FlatParquetDataset,
    FlatArrowRowGroupParquetDataset, WeightedFlatRowgroupParquetDataset,
    WeightedRgTensorParquetDataset, RgTensorParquetDataset,
    WeightedTensorParquetDataset, TensorParquetDataset,
)
from hbt.ml.torch_utils.map_and_collate import (
    NestedListRowgroupMapAndCollate, FlatListRowgroupMapAndCollate,
    NestedDictMapAndCollate, TensorListRowgroupMapAndCollate, NestedTensorListRowgroupMapAndCollate,
    NestedMapAndCollate, FlatMapAndCollate, TensorListMapAndCollate, NestedTensorMapAndCollate,
)

from hbt.ml.torch_utils.samplers import ListRowgroupSampler, RowgroupSampler, ListSampler


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
        inputs: law.FileCollection | None = None,
        open_options: dict[str, Any] | None = None,
        columns: Collection[str | Route] | None = None,
        datasets: Collection[str] | None = None,
        batch_transformations: torch.nn.Module | None = None,
        global_transformations: torch.nn.Module | None = None,
        input_data_transform: torch.nn.Module | None = None,
        categorical_target_transformation: torch.nn.Module | None = None,
        data_type_transform: torch.nn.Module | None = None,
        preshuffle: bool = True,
        build_categorical_target_fn: Callable | None = None,
        group_datasets: dict[str, list[str]] | None = None,
        device: str | None = None,
        extract_dataset_paths_fn: Callable | None = None,
        extract_probability_fn: Callable | None = None,
        **kwargs,
    ):
        self.open_options = open_options or dict()
        self.task = task
        self.inputs = inputs or task.input()
        self.configs = task.configs
        self.datasets = datasets or task.datasets
        self.load_parallel_cores = task.load_parallel_cores
        self.batch_size = task.batch_size
        self.columns = columns
        self.weight_dict = dict()

        self.batch_transformations = batch_transformations
        self.global_transformations = global_transformations
        self.input_data_transform = input_data_transform
        self.categorical_target_transformation = categorical_target_transformation
        self.data_type_transform = data_type_transform
        self.preshuffle = preshuffle
        self.build_categorical_target_fn = build_categorical_target_fn or self._default_build_categorical_target_fn

        self.group_datasets = group_datasets or dict()
        self.dataset_cls = FlatParquetDataset
        self.device = device

        self.extract_dataset_paths: Callable = extract_dataset_paths_fn or self._default_extract_dataset_paths
        self._extract_probability: Callable = extract_probability_fn or self._default_extract_probability

        self.default_dataset_kwargs = {
            "columns": self.columns,
            "batch_transform": self.batch_transformations,
            "global_transform": self.global_transformations,
            "input_data_transform": self.input_data_transform,
            "categorical_target_transform": self.categorical_target_transformation,
            "data_type_transform": self.data_type_transform,
        }

    def _default_build_categorical_target_fn(self, dataset: str) -> int:
        return int(1) if dataset.startswith("hh") else int(0)

    def create_dataset(
        self,
        path: str | ak.Array,
        open_options: dict[str, Any] | None = None,
        **dataset_kwargs
    ):

        return self.dataset_cls(
            path,
            open_options=open_options,
            **dataset_kwargs,
        )

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

        logger.info(f"Constructing training dataset for {target_path} with row_groups {training_row_groups}")

        training = self.create_dataset(
            target_path,
            open_options=final_options,
            target=targets,
            **self.default_dataset_kwargs,
        )

        validation = None
        if training_row_groups is None:
            validation = training
        else:
            validation_row_groups = rowgroup_indices[max_training_group:]
            final_options.update({"row_groups": validation_row_groups})
            logger.info(f"Constructing validation dataset for {target_path} with row_groups {validation_row_groups}")
            validation = self.create_dataset(
                target_path,
                open_options=final_options,
                target=targets,
                **self.default_dataset_kwargs,
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
        for path in sorted(target_paths):
            training, validation = self._split_pq_dataset_per_path(
                target_path=path,
                ratio=ratio,
                targets=targets,
            )
            training_ds.append(training)
            validation_ds.append(validation)
        return training_ds, validation_ds

    def _default_extract_probability(
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
        shuffle_list=False,
        simultaneous_rowgroups=-1,
    ):
        return cls(
            inputs=datasets,
            shuffle_rowgroups=shuffle_rowgroups,
            shuffle_indices=shuffle_indices,
            shuffle_list=shuffle_list,
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

    def _default_extract_dataset_paths(self, inputs, dataset):
        targets = [inputs.events[dataset][c]["collection"] for c in sorted(self.configs)]
        return [
            t.abspath
            for collections in targets
            for targets in collections.targets.values()
            for t in targets.values()
        ]

    def _init_training_validation_map(self) -> tuple[dict[str, list[ParquetDataset]], dict[str, list[ParquetDataset]]]:
        # construct datamap
        training_data_map: dict[str, list[ParquetDataset]] = dict()
        validation_data_map: dict[str, list[ParquetDataset]] = dict()

        for dataset in sorted(self.datasets):
            # following code is used for SiblingFileCollections
            target_paths = self.extract_dataset_paths(self.inputs, dataset)

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

    def _create_validation_dataloader(
        self,
        validation_data_map,
        weights=None,
        **sampler_kwargs,
    ) -> CompositeDataLoader:
        # create merged validation dataset since it's ok to simply evaluate the
        # events one by one
        validation_data: list[ParquetDataset] = list()
        validation_weights = None
        if weights is not None:
            validation_weights: list[float] = list()

        for key, x in validation_data_map.items():
            if not isinstance(x, (list, tuple, set)):
                validation_data.append(x)
                if validation_weights is not None:
                    validation_weights.append(weights[key])
            else:
                validation_data.extend(x)
                if validation_weights is not None:
                    validation_weights.extend([weights[key]] * len(x))

        def wrapped_map_and_collate_cls(data_map, collate_fn):
            return self.validation_map_and_collate_cls(
                data_map,
                collate_fn=collate_fn,
                weights=validation_weights,
            )
        return CompositeDataLoader(
            validation_data,
            batch_sampler_cls=tn.Batcher,
            shuffle=False,
            batch_size=self.batch_size,
            batcher_options={
                "source": tn.SamplerWrapper(
                    self.sampler_factory(
                        validation_data,
                        cls=self.validation_sampler_cls,
                        **sampler_kwargs,
                    ),
                ),
            },
            map_and_collate_cls=wrapped_map_and_collate_cls,
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
            shuffle_list=True,
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

    def _create_validation_dataloader(
        self,
        validation_data_map,
        **sampler_kwargs,
    ) -> CompositeDataLoader:
        # calculate final weights
        total_events = 0
        weights = dict()
        for key, weight in self.weight_dict.items():

            if isinstance(weight, dict):
                # also account for sum of sub weights such that total sum is 1
                for k, v in weight.items():
                    data = validation_data_map[k]
                    total_events = sum(len(x) for x in data) if isinstance(data, (list, tuple, set)) else len(data)
                    weights[k] = v / total_events
            else:
                data = validation_data_map[key]
                total_events = sum(len(x) for x in data) if isinstance(data, (list, tuple, set)) else len(data)
                weights[key] = weight / total_events

        return super()._create_validation_dataloader(
            validation_data_map,
            weights=weights,
            **sampler_kwargs,
        )

    def init_datasets(self) -> tuple[CompositeDataLoader, tuple[CompositeDataLoader, CompositeDataLoader]]:

        training_data_map, validation_data_map = self._init_training_validation_map()

        self._get_weights()

        sampler_fn = partial(
            self.sampler_factory,
            cls=self.training_sampler_cls,
            shuffle_rowgroups=True,
            shuffle_indices=True,
            shuffle_list=True,
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
        train_val_composite_loader = self._create_validation_dataloader(
            training_data_map,
            shuffle_rowgroups=False,
            shuffle_indices=True,
            shuffle_list=True,
        )
        validation_composite_loader = self._create_validation_dataloader(validation_data_map)

        return (training_composite_loader, (train_val_composite_loader, validation_composite_loader))


class RgTensorParquetFileHandler(BaseParquetFileHandler):
    def __init__(
        self,
        *args,
        continuous_features: Container[str] | None = None,
        categorical_features: Container[str] | None = None,
        **kwargs,
    ):
        # overwrite the columns to load
        self.continuous_features = continuous_features or {}
        self.categorical_features = categorical_features or {}
        columns = set(self.continuous_features) | set(self.categorical_features)
        if getattr(kwargs, "columns", None) is not None:
            logger.warning(
                f"{self.__class__.__name__} only supports input feature columns "
                "that are split by categorical and continuous features. Will ignore "
                f"columns={kwargs['columns']} and use columns={columns} instead.",
            )
        kwargs["columns"] = columns
        super().__init__(*args, **kwargs)
        self.dataset_cls = RgTensorParquetDataset
        self.training_map_and_collate_cls = NestedTensorListRowgroupMapAndCollate
        self.validation_map_and_collate_cls = TensorListRowgroupMapAndCollate
        self.training_sampler_cls = ListRowgroupSampler
        self.validation_sampler_cls = ListRowgroupSampler

        self.default_dataset_kwargs = {
            "categorical_features": self.categorical_features,
            "continuous_features": self.continuous_features,
            "batch_transform": self.batch_transformations,
            "global_transform": self.global_transformations,
            "input_data_transform": self.input_data_transform,
            "categorical_target_transform": self.categorical_target_transformation,
            "data_type_transform": self.data_type_transform,
            "padd_value_float": 0,
            "padd_value_int": 15,
        }


class WeightedRgTensorParquetFileHandler(WeightedFlatListRowgroupParquetFileHandler):
    def __init__(
        self,
        *args,
        continuous_features: Container[str] | None = None,
        categorical_features: Container[str] | None = None,
        **kwargs,
    ):
        # overwrite the columns to load
        self.continuous_features = continuous_features or {}
        self.categorical_features = categorical_features or {}
        columns = set(self.continuous_features) | set(self.categorical_features)
        if getattr(kwargs, "columns", None) is not None:
            logger.warning(
                f"{self.__class__.__name__} only supports input feature columns "
                "that are split by categorical and continuous features. Will ignore "
                f"columns={kwargs['columns']} and use columns={columns} instead.",
            )
        kwargs["columns"] = columns
        super().__init__(*args, **kwargs)
        self.dataset_cls = WeightedRgTensorParquetDataset
        # self.validation_dataset_cls = WeightedFlatRowgroupParquetDataset
        self.training_map_and_collate_cls = NestedTensorListRowgroupMapAndCollate
        self.validation_map_and_collate_cls = TensorListRowgroupMapAndCollate
        self.training_sampler_cls = ListRowgroupSampler
        self.validation_sampler_cls = ListRowgroupSampler
        self.default_dataset_kwargs = {
            "categorical_features": self.categorical_features,
            "continuous_features": self.continuous_features,
            "batch_transform": self.batch_transformations,
            "global_transform": self.global_transformations,
            "input_data_transform": self.input_data_transform,
            "categorical_target_transform": self.categorical_target_transformation,
            "data_type_transform": self.data_type_transform,
            "padd_value_float": 0,
            "padd_value_int": 15,
        }


class TensorParquetFileHandler(WeightedRgTensorParquetFileHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_cls = TensorParquetDataset
        # self.validation_dataset_cls = WeightedFlatRowgroupParquetDataset
        self.training_map_and_collate_cls = NestedTensorMapAndCollate
        self.validation_map_and_collate_cls = TensorListMapAndCollate
        self.training_sampler_cls = torch.utils.data.RandomSampler
        self.validation_sampler_cls = ListSampler
        self.default_dataset_kwargs = {
            "categorical_features": self.categorical_features,
            "continuous_features": self.continuous_features,
            "batch_transform": self.batch_transformations,
            "global_transform": self.global_transformations,
            "input_data_transform": self.input_data_transform,
            "categorical_target_transform": self.categorical_target_transformation,
            "data_type_transform": self.data_type_transform,
            "padd_value_float": 0,
            "padd_value_int": 15,
        }

    def split_training_validation(
        self,
        target_paths,
        ratio=0.7,
        targets: Collection[str | int | Route] | str | Route | int | None = None,
    ):
        meta = ak.metadata_from_parquet(target_paths)

        total_rows = meta["num_rows"]
        indices_func = torch.randperm if self.preshuffle else np.arange
        total_indices: list[int] = indices_func(total_rows).tolist()
        max_training_idx = int(total_rows * ratio)
        training_idx = total_indices[:max_training_idx]

        final_options = self.open_options or dict()

        logger.info(f"Constructing training dataset with {max_training_idx} events")

        training = self.create_dataset(
            target_paths,
            idx=training_idx,
            target=targets,
            open_options=final_options,
            **self.default_dataset_kwargs,
        )
        validation_idx = total_indices[max_training_idx:]
        logger.info(f"Constructing validation dataset with {total_rows - max_training_idx} events")
        validation = self.create_dataset(
            target_paths,
            idx=validation_idx,
            target=targets,
            open_options=final_options,
            **self.default_dataset_kwargs,
        )
        return training, validation

    def sampler_factory(
        self,
        datasets,
        cls: Callable = torch.utils.data.Sampler,
        shuffle_indices=False,
        shuffle_list=False,
        **kwargs,
    ):
        if issubclass(cls, ListSampler):
            return cls(datasets, shuffle_indices=shuffle_indices, shuffle_list=shuffle_list)
        else:
            return cls(datasets)

    # def _create_validation_dataloader(
    #     self,
    #     validation_data_map,
    #     weights: dict[str, float] | None = None,
    #     **sampler_kwargs,
    # ) -> CompositeDataLoader:
    #     # create merged validation dataset since it's ok to simply evaluate the
    #     # events one by one
    #     validation_data: list[ParquetDataset] = list()

    #     for key, x in validation_data_map.items():
    #         if not isinstance(x, (list, tuple, set)):
    #             if weights is not None:
    #                 x.cls_weight = weights[key]
    #             validation_data.append(x)

    #         else:
    #             if weights is not None:
    #                 for sub_x in x:
    #                     sub_x.cls_weight = weights[key]
    #             validation_data.extend(x)
    #     from IPython import embed
    #     embed(header="in _create_validation_dataloader with flattend validation_data")
    #     return CompositeDataLoader(
    #         validation_data,
    #         batch_sampler_cls=tn.Batcher,
    #         shuffle=False,
    #         batch_size=self.batch_size,
    #         batcher_options={
    #             "source": tn.SamplerWrapper(
    #                 self.sampler_factory(
    #                     validation_data,
    #                     cls=self.validation_sampler_cls,
    #                     **sampler_kwargs,
    #                 ),
    #             ),
    #         },
    #         map_and_collate_cls=self.validation_map_and_collate_cls,
    #         device=self.device,
    #         collate_fn=torch.cat,
    #     )


class WeightedTensorParquetFileHandler(TensorParquetFileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_cls = WeightedTensorParquetDataset
        self.training_map_and_collate_cls = NestedTensorMapAndCollate
        self.validation_map_and_collate_cls = TensorListMapAndCollate
        self.training_sampler_cls = torch.utils.data.RandomSampler
        self.validation_sampler_cls = ListSampler


class FlatArrowParquetFileHandler(BaseParquetFileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_cls = FlatArrowRowGroupParquetDataset
        self.training_map_and_collate_cls = NestedDictMapAndCollate
        self.validation_map_and_collate_cls = FlatListRowgroupMapAndCollate
        self.training_sampler_cls = RowgroupSampler
        self.validation_sampler_cls = ListRowgroupSampler

        self.default_dataset_kwargs = {
            "columns": self.columns,
            "batch_transform": self.batch_transformations,
            "global_transform": self.global_transformations,
            "input_data_transform": self.input_data_transform,
            "categorical_target_transform": self.categorical_target_transformation,
            "data_type_transform": self.data_type_transform,
        }
    
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
