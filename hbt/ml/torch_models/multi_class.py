from __future__ import annotations

__all__ = [
]

from functools import partial

from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import Any
from collections.abc import Container

from columnflow.columnar_util import Route, EMPTY_FLOAT, EMPTY_INT

from hbt.ml.torch_utils.functions import (
    get_one_hot, generate_weighted_loss, preprocess_multiclass_outputs,
    WeightedCrossEntropySlice,
)

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

    from hbt.ml.torch_models.binary import FeedForwardNet, TensorFeedForwardNet
    from hbt.ml.torch_utils.ignite.metrics import (
        WeightedROC_AUC, WeightedLoss,
    )
    from hbt.ml.torch_utils.transforms import AkToTensor, PreProcessFloatValues, MoveToDevice
    from hbt.ml.torch_utils.datasets.handlers import (
        FlatListRowgroupParquetFileHandler, FlatArrowParquetFileHandler,
        WeightedFlatListRowgroupParquetFileHandler,
        RgTensorParquetFileHandler, WeightedRgTensorParquetFileHandler,
    )
    from hbt.ml.torch_utils.utils import (
        embedding_expected_inputs, LookUpTable, CategoricalTokenizer,
    )
    from hbt.ml.torch_utils.ignite.mixins import IgniteTrainingMixin, IgniteEarlyStoppingMixin
    from hbt.ml.torch_utils.layers import PaddingLayer

    class FeedForwardMultiCls(FeedForwardNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.linear_relu_stack = nn.Sequential(
                nn.BatchNorm1d(len(self.inputs)),
                nn.Linear(len(self.inputs), 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 3),
            )
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

        def validation_step(self, engine, batch):
            # Set the model to evaluation mode - important for batch normalization and dropout layers
            self.eval()
            # if engine.state.iteration > self.max_val_epoch_length * (engine.state.epoch + 1):
            #     engine.terminate_epoch()

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors
            # with requires_grad=True
            with torch.no_grad():
                X, y = batch[0], batch[1]
                pred = self(X)
                target = y["categorical_target"].to(torch.float32)
                if target.dim() == 1:
                    target = target.reshape(-1, 1)
                return pred, target

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

                elif isinstance(weights, dict):
                    for subkey, weight in weights.items():
                        data = composite_loader.data_map[subkey]
                        if isinstance(data, (list, tuple, set)):
                            total_length = sum([len(x) for x in data])
                        else:
                            total_length = len(data)
                        submax = np.ceil(total_length / batchsize / weight)
                        if submax > global_max and weight >= weight_cutoff:
                            global_max = submax
                            max_key = subkey
            if cutoff:
                global_max = np.min([global_max, cutoff])
            self.logger.info(f"epoch dominated by  '{max_key}': expect {global_max} batches/iteration")
            return global_max

        def init_dataset_handler(self, task: law.Task):
            all_datasets = getattr(task, "resolved_datasets", task.datasets)
            group_datasets = {
                "ttbar": [d for d in all_datasets if d.startswith("tt_")],
                "dy": [d for d in all_datasets if d.startswith("dy_")],
            }
            device = next(self.parameters()).device

            self.dataset_handler = FlatListRowgroupParquetFileHandler(
                task=task,
                columns=self.inputs,
                batch_transformations=AkToTensor(device=device),
                # global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                categorical_target_transformation=partial(get_one_hot, nb_classes=3),
                group_datasets=group_datasets,
                device=device,
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_", "dy_"])],
            )
            self.training_loader, self.validation_loader = self.dataset_handler.init_datasets()
            self.max_epoch_length = self._calculate_max_epoch_length(
                self.training_loader,
                cutoff=self.training_epoch_length_cutoff,
                weight_cutoff=self.training_weight_cutoff,
            )
            # self.max_val_epoch_length = self._calculate_max_epoch_length(self.validation_loader)

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
            # if engine.state.iteration > self.max_val_epoch_length * (engine.state.epoch + 1):
            #     engine.terminate_epoch()

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors with
            # requires_grad=True
            with torch.no_grad():
                X, y = batch[0], batch[1]
                pred = self(X)
                target = y["categorical_target"].to(torch.float32)
                if target.dim() == 1:
                    target = target.reshape(-1, 1)
                return pred, target, {"weight": X["weights"]}

        def init_dataset_handler(self, task: law.Task, device: str | None = None):
            all_datasets = getattr(task, "resolved_datasets", task.datasets)
            group_datasets = {
                "ttbar": [d for d in all_datasets if d.startswith("tt_")],
                "dy": [d for d in all_datasets if d.startswith("dy_")],
            }
            device = next(self.parameters()).device

            self.dataset_handler = WeightedRgTensorParquetFileHandler(
                task=task,
                continuous_features=getattr(self, "continuous_features", self.inputs),
                categorical_features=getattr(self, "categorical_features", None),
                batch_transformations=MoveToDevice(device=device),
                categorical_target_transformation=partial(get_one_hot, nb_classes=3),
                group_datasets=group_datasets,
                device=device,
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_", "dy_"])],
            )
            self.training_loader, (self.train_validation_loader, self.validation_loader) = self.dataset_handler.init_datasets()  # noqa
            self.max_epoch_length = self._calculate_max_epoch_length(
                self.training_loader,
                cutoff=self.training_epoch_length_cutoff,
                weight_cutoff=self.training_weight_cutoff,
            )

            dm = self.validation_loader.data_map
            batcher = self.validation_loader.batcher
            self.max_val_epoch_length = np.ceil(sum(map(len, dm)) / batcher.batch_size)
            if self.val_epoch_length_cutoff is not None and self.max_val_epoch_length > self.val_epoch_length_cutoff:
                self.logger.info(f"validation epoch length cutoff: {self.val_epoch_length_cutoff}")
                self.max_val_epoch_length = self.val_epoch_length_cutoff

    class DeepFeedForwardMultiCls(FeedForwardMultiCls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
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

            self._loss_fn = nn.CrossEntropyLoss()
            self.validation_metrics = {
                "loss": Loss(self.loss_fn),
                # "roc_auc": ROC_AUC(),
            }

        def forward(self, x):
            input_data = self._handle_input(x)
            logits = self.linear_relu_stack1(input_data.to(torch.float32))
            logits = self.linear_relu_stack2(logits)
            return logits
