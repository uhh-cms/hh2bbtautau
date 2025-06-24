from __future__ import annotations

__all__ = [
]

from functools import partial

from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import Callable
from collections.abc import Container

from columnflow.columnar_util import Route, EMPTY_FLOAT, EMPTY_INT

from hbt.ml.torch_utils.functions import (
    get_one_hot, generate_weighted_loss, preprocess_multiclass_outputs,
    WeightedCrossEntropySlice,
)
from hbt.ml.torch_utils.utils import get_standardization_parameter

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")
ak = maybe_import("awkward")
import law

model_clss: DotDict[str, torch.nn.Module] = DotDict()

if not isinstance(torch, MockModule):
    from torch import nn
    from torch.optim import Adam, AdamW
    from torch.utils.tensorboard import SummaryWriter
    from ignite.metrics import Loss, ROC_AUC

    from hbt.ml.torch_models.binary import FeedForwardNet, TensorFeedForwardNet, WeightedTensorFeedForwardNetWithCat
    from hbt.ml.torch_utils.ignite.metrics import (
        WeightedROC_AUC, WeightedLoss,
    )
    from hbt.ml.torch_utils.transforms import AkToTensor, PreProcessFloatValues, MoveToDevice
    from hbt.ml.torch_utils.datasets.handlers import (
        FlatListRowgroupParquetFileHandler, FlatArrowParquetFileHandler,
        WeightedFlatListRowgroupParquetFileHandler,
        RgTensorParquetFileHandler, WeightedRgTensorParquetFileHandler,
        WeightedTensorParquetFileHandler, TensorParquetFileHandler
    )
    from hbt.ml.torch_utils.utils import (
        embedding_expected_inputs, LookUpTable, CategoricalTokenizer,
    )
    from hbt.ml.torch_utils.ignite.mixins import IgniteTrainingMixin, IgniteEarlyStoppingMixin
    from hbt.ml.torch_utils.layers import PaddingLayer, StandardizeLayer, InputLayer

    class FeedForwardMultiCls(TensorFeedForwardNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.categorical_features = {
                "pair_type",
                "decay_mode1",
                "decay_mode2",
                "lepton1.charge",
                "lepton2.charge",
                "has_fatjet",
                "has_jet_pair",
                "year_flag",
            }
            self.continuous_features = self.inputs
            self.categorical_target_map = {
                "hh": 0,
                "tt": 1,
                "dy": 2,
            }
            self._loss_fn = nn.CrossEntropyLoss()
            self.validation_metrics = {
                "loss": Loss(self.loss_fn),
                # "roc_auc": ROC_AUC(),
            }
            self.training_epoch_length_cutoff = 2000
            self.training_weight_cutoff = 0.05

        def setup_preprocessing(self):
            # extract dataset std and mean from dataset
            # extraction happens form no oversampled dataset
            mean, std = [], []
            for _input in sorted(self.inputs, key=str):
                input_statitics = self.dataset_statistics[_input.column]
                mean.append(torch.from_numpy(input_statitics["mean"]))
                std.append(torch.from_numpy(input_statitics["std"]))

            device = next(self.parameters()).device
            mean, std = torch.concat(mean).to(device), torch.concat(std).to(device)

            # set up standardization layer
            self.std_layer.update_buffer(
                mean.float(),
                std.float(),
            )

        def init_layers(self):
            self.std_layer = StandardizeLayer()
            self.input_layer = InputLayer(
                categorical_inputs=sorted(self.categorical_features, key=str),
                continuous_inputs=sorted(self.inputs, key=str),
                expected_categorical_inputs=embedding_expected_inputs,
                embedding_dim=self.embedding_dims,
            )
            self.padding_layer_cat = PaddingLayer(padding_value=self.input_layer.empty, mask_value=EMPTY_INT)
            self.padding_layer_cont = PaddingLayer(padding_value=0, mask_value=EMPTY_FLOAT)

            self.linear_relu_stack = nn.Sequential(
                nn.Linear(self.input_layer.ndim, 512),
                nn.PReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.PReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.PReLU(),
                nn.Linear(512, 3),
            )

        def _build_categorical_target(self, dataset: str):
            for key in self.categorical_target_map.keys():
                if dataset.startswith(key):
                    return self.categorical_target_map[key]
            raise ValueError(f"Dataset {dataset} not in categorical target map")

        def _calculate_max_epoch_length(
            self,
            composite_loader,
            weight_cutoff: float | None = None,
            cutoff: int | None = None,
            # priority_list: list[str] | None = None,
        ):
            global_max = 0
            max_key = None
            weight_cutoff = weight_cutoff or 0.
            batch_comp = getattr(composite_loader.batcher, "_batch_composition", None)
            # composition_length = {}
            if not batch_comp:
                batch_comp = {key: 1. for key in composite_loader.data_map.keys()}
            for key, batchsize in batch_comp.items():
                weights = composite_loader.weight_dict[key]
                if isinstance(weights, float):
                    data = composite_loader.data_map[key]
                    if isinstance(data, (list, tuple, set)):
                        total_length = sum([len(x) for x in data])
                    else:
                        total_length = len(data)
                    local_max = np.ceil(total_length / batchsize)
                    if local_max > global_max:
                        max_key = key
                        global_max = local_max
                    # composition_length[key] = local_max

                elif isinstance(weights, dict):
                    for subkey, weight in weights.items():
                        data = composite_loader.data_map[subkey]
                        if isinstance(data, (list, tuple, set)):
                            total_length = sum([len(x) for x in data])
                        else:
                            total_length = len(data)
                        submax = np.ceil(total_length / batchsize / weight)
                        # filter out datasets with small weight contribution
                        if submax > global_max and weight >= weight_cutoff:
                            global_max = submax
                            max_key = subkey

            if cutoff:
                global_max = np.min([global_max, cutoff])
            self.logger.info(f"epoch dominated by  '{max_key}': expect {global_max} batches/iteration")
            return global_max

        def init_dataset_handler(
            self,
            task: law.Task,
            *args,
            device: str | None = None,
            datasets: list[str] | None = None,
            extract_dataset_paths_fn: Callable | None = None,
            extract_probability_fn: Callable | None = None,
            inputs: law.FileCollection | None = None,
            **kwargs,
        ):
            all_datasets = datasets or getattr(task, "resolved_datasets", task.datasets)
            group_datasets = {
                "ttbar": [d for d in all_datasets if d.startswith("tt_")],
                "dy": [d for d in all_datasets if d.startswith("dy_")],
            }
            device = next(self.parameters()).device

            self.dataset_handler = TensorParquetFileHandler(
                task=task,
                inputs=inputs,
                continuous_features=getattr(self, "continuous_features", self.inputs),
                categorical_features=getattr(self, "categorical_features", None),
                batch_transformations=MoveToDevice(device=device),
                # global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                categorical_target_transformation=partial(get_one_hot, nb_classes=len(self.categorical_target_map)),
                group_datasets=group_datasets,
                device=device,
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_", "dy_"])],
                extract_dataset_paths_fn=extract_dataset_paths_fn,
                extract_probability_fn=extract_probability_fn,
            )
            self.training_loader, self.validation_loader = self.dataset_handler.init_datasets()
            self.max_epoch_length = self._calculate_max_epoch_length(
                self.training_loader,
                cutoff=self.training_epoch_length_cutoff,
                weight_cutoff=self.training_weight_cutoff,
            )
            self.dataset_statistics = get_standardization_parameter(self.train_validation_loader.data_map, self.inputs)

            # self.max_val_epoch_length = self._calculate_max_epoch_length(self.validation_loader)

        def to(self, *args, **kwargs):
            self.std_layer = self.std_layer.to(*args, **kwargs)
            self.input_layer = self.input_layer.to(*args, **kwargs)
            return super().to(*args, **kwargs)

    class WeightedFeedForwardMultiCls(FeedForwardMultiCls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._loss_fn = generate_weighted_loss(torch.nn.CrossEntropyLoss)()
            self.validation_metrics = {
                "loss": WeightedLoss(self.loss_fn),
                # "roc_auc": ROC_AUC(),
            }
            self.training_epoch_length_cutoff = 2000
            self.training_weight_cutoff = 0.05

        def validation_step(self, engine, batch):
            # Set the model to evaluation mode - important for batch normalization and dropout layers
            self.eval()

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors
            # with requires_grad=True
            with torch.no_grad():
                X, y = batch[0], batch[1]
                input_data, weights = X[:-1], X[-1]
                pred = self(input_data)
                target = y.to(torch.float32)
                if target.dim() == 1:
                    target = target.reshape(-1, 1)

                return pred, target, {"weight": weights}

        def init_dataset_handler(
            self,
            task: law.Task,
            device: str | None = None,
            *args,
            datasets: list[str] | None = None,
            extract_dataset_paths_fn: Callable | None = None,
            extract_probability_fn: Callable | None = None,
            inputs: law.FileCollection | None = None,
            **kwargs,
        ):
            all_datasets = datasets or getattr(task, "resolved_datasets", task.datasets)
            group_datasets = {
                "ttbar": [d for d in all_datasets if d.startswith("tt_")],
                "dy": [d for d in all_datasets if d.startswith("dy_")],
            }
            device = next(self.parameters()).device

            self.dataset_handler = WeightedTensorParquetFileHandler(
                task=task,
                inputs=inputs,
                continuous_features=getattr(self, "continuous_features", self.inputs),
                categorical_features=getattr(self, "categorical_features", None),
                batch_transformations=MoveToDevice(device=device),
                build_categorical_target_fn=self._build_categorical_target,
                categorical_target_transformation=partial(get_one_hot, nb_classes=len(self.categorical_target_map)),
                group_datasets=group_datasets,
                device=device,
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_", "dy_"])],
                extract_dataset_paths_fn=extract_dataset_paths_fn,
                extract_probability_fn=extract_probability_fn,
            )
            self.training_loader, (self.train_validation_loader, self.validation_loader) = self.dataset_handler.init_datasets()  # noqa
            self.max_epoch_length = self._calculate_max_epoch_length(
                self.training_loader,
                cutoff=self.training_epoch_length_cutoff,
                weight_cutoff=self.training_weight_cutoff,
            )
            self.dataset_statistics = get_standardization_parameter(self.train_validation_loader.data_map, self.inputs)

    class DeepFeedForwardMultiCls(FeedForwardMultiCls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self._loss_fn = nn.CrossEntropyLoss()
            self.validation_metrics = {
                "loss": Loss(self.loss_fn),
                # "roc_auc": ROC_AUC(),
            }

        def init_layers(self):
            self.std_layer = StandardizeLayer()
            self.input_layer = InputLayer(
                categorical_inputs=sorted(self.categorical_features, key=str),
                continuous_inputs=sorted(self.inputs, key=str),
                expected_categorical_inputs=embedding_expected_inputs,
                embedding_dim=self.embedding_dims,
            )
            self.padding_layer_cat = PaddingLayer(padding_value=self.input_layer.empty, mask_value=EMPTY_INT)
            self.padding_layer_cont = PaddingLayer(padding_value=0, mask_value=EMPTY_FLOAT)

            self.linear_relu_stack1 = nn.Sequential(
                nn.BatchNorm1d(len(self.inputs)),
                nn.Linear(len(self.inputs), 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.ReLU(),
            )
            self.linear_relu_stack2 = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.Linear(512, 2048),
                nn.ReLU(),
                nn.BatchNorm1d(2048),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 3),
            )

        def forward(self, x):
            input_data = self._handle_input(x)
            logits = self.linear_relu_stack1(input_data.to(torch.float32))
            logits = self.linear_relu_stack2(logits)
            return logits
