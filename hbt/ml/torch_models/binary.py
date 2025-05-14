from __future__ import annotations

__all__ = [
]

from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import Any
from collections.abc import Container

from columnflow.columnar_util import Route, EMPTY_FLOAT, EMPTY_INT

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

    from hbt.ml.torch_utils.transforms import AkToTensor, PreProcessFloatValues, MoveToDevice
    from hbt.ml.torch_utils.datasets.handlers import (
        FlatListRowgroupParquetFileHandler, FlatArrowParquetFileHandler,
        WeightedFlatListRowgroupParquetFileHandler,
        RgTensorParquetFileHandler, WeightedRgTensorParquetFileHandler,
    )
    from hbt.ml.torch_utils.ignite.metrics import (
        WeightedROC_AUC, WeightedLoss,
    )
    from hbt.ml.torch_utils.utils import (
        embedding_expected_inputs,
    )
    from hbt.ml.torch_utils.functions import generate_weighted_loss
    from hbt.ml.torch_utils.ignite.mixins import IgniteTrainingMixin, IgniteEarlyStoppingMixin
    from hbt.ml.torch_utils.layers import PaddingLayer, InputLayer

    class NetworkBase(nn.Module):
        def __init__(self, *args, tensorboard_path: str | None = None, logger: Any | None = None, **kwargs):
            super().__init__()
            self.writer = None
            self.logger = logger or law.logger.get_logger(__name__)
            if tensorboard_path:
                self.logger.info(f"Creating tensorboard logger at {tensorboard_path}")
                self.writer = SummaryWriter(log_dir=tensorboard_path)
            self.custom_hooks = list()

    class FeedForwardNet(
        IgniteEarlyStoppingMixin,
        IgniteTrainingMixin,
        NetworkBase,
    ):
        def __init__(
            self,
            *args,
            tensorboard_path: str | None = None, logger: Any | None = None,
            task: law.Task,
            **kwargs,
        ):
            super().__init__(*args, tensorboard_path=tensorboard_path, logger=logger, **task.param_kwargs, **kwargs)

            columns = [
                "lepton1.{px,py,pz,energy,mass}",
                "lepton2.{px,py,pz,energy,mass}",
                "bjet1.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
                "bjet2.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
                "fatjet.{px,py,pz,energy,mass}",
            ]
            self.inputs = set()
            self.inputs.update(*list(map(Route, law.util.brace_expand(obj)) for obj in columns))

            self.padding_layer = PaddingLayer(padding_value=0, mask_value=EMPTY_FLOAT)
            self.norm_layer = nn.BatchNorm1d(len(self.inputs))
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(len(self.inputs), 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid(),
            )

            self.logger.info("Constructing loss and optimizer")
            self._loss_fn = nn.BCELoss()
            self.validation_metrics = {
                "loss": Loss(self.loss_fn),
                "roc_auc": ROC_AUC(),
            }
            self.training_epoch_length_cutoff = None
            self.training_weight_cutoff = None
            self.val_epoch_length_cutoff = None
            self.val_weight_cutoff = None
            self.training_logger_interval = 20

        def init_optimizer(self, learning_rate=1e-3, weight_decay=1e-5) -> None:
            self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        def _build_categorical_target(self, dataset: str):
            return int(1) if dataset.startswith("hh") else int(0)

        def train_step(self, engine, batch):
            # Set the model to training mode - important for batch normalization and dropout layers
            self.train()
            # Compute prediction and loss
            X, y = batch[0], batch[1]
            self.optimizer.zero_grad()

            pred = self(X)
            target = y["categorical_target"].to(torch.float32)
            if target.dim() == 1:
                target = target.reshape(-1, 1)

            loss = self.loss_fn(pred, target)
            # Backpropagation
            loss.backward()
            self.optimizer.step()

            return loss.item()

        def validation_step(self, engine, batch):
            # Set the model to evaluation mode - important for batch normalization and dropout layers
            self.eval()

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

        def init_dataset_handler(self, task: law.Task):
            group_datasets = {
                "ttbar": [d for d in task.datasets if d.startswith("tt_")],
            }
            device = next(self.parameters()).device
            all_datasets = getattr(task, "resolved_datasets", task.datasets)

            self.dataset_handler = FlatListRowgroupParquetFileHandler(
                task=task,
                columns=self.inputs,
                batch_transformations=AkToTensor(device=device),
                # global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_"])],
            )
            self.training_loader, self.validation_loader = self.dataset_handler.init_datasets()

        def _handle_input(
            self,
            x: dict[str, torch.Tensor],
            feature_list: Container[str] | None = None,
            mask_value: int | float = EMPTY_FLOAT,
            empty_fill_val: float = 0,
            norm_layer: nn.Module | None = None,
            dtype=torch.float32,
        ):
            if not feature_list:
                feature_list = self.inputs
            input_data = x

            if isinstance(x, dict):
                input_data: torch.Tensor = torch.cat([
                    x[str(key)].reshape(-1, 1) for key in sorted(feature_list)],
                    axis=-1,
                )
            # check for dummy values
            input_data = self.padding_layer(input_data)

            if norm_layer:
                input_data = norm_layer(input_data.to(dtype))
                # from IPython import embed
                # embed(header=f"using norm layer to derive default values")

            return input_data.to(dtype)

        def forward(self, x):
            input_data = self._handle_input(x, norm_layer=getattr(self, "norm_layer", None))
            logits = self.linear_relu_stack(input_data)
            return logits

    class TensorFeedForwardNet(FeedForwardNet):
        def init_dataset_handler(self, task: law.Task):
            group_datasets = {
                "ttbar": [d for d in task.datasets if d.startswith("tt_")],
            }
            device = next(self.parameters()).device
            all_datasets = getattr(task, "resolved_datasets", task.datasets)

            self.dataset_handler = RgTensorParquetFileHandler(
                task=task,
                continuous_features=self.inputs,
                batch_transformations=MoveToDevice(device=device),
                # global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_"])],
            )
            self.training_loader, self.validation_loader = self.dataset_handler.init_datasets()

        def forward(self, x):
            x = self.padding_layer(x)
            x = self.norm_layer(x.to(torch.float32))
            logits = self.linear_relu_stack(x)
            return logits

        def train_step(self, engine, batch):
            # Set the model to training mode - important for batch normalization and dropout layers
            self.train()
            # Compute prediction and loss
            X, y = batch[0], batch[1]
            self.optimizer.zero_grad()
            pred = self(X)

            target = y.to(torch.float32)
            if target.dim() == 1:
                target = target.reshape(-1, 1)

            loss = self.loss_fn(pred, target)
            # Backpropagation
            loss.backward()
            self.optimizer.step()

            return loss.item()

        def validation_step(self, engine, batch):
            # Set the model to evaluation mode - important for batch normalization and dropout layers
            self.eval()

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors
            # with requires_grad=True
            with torch.no_grad():
                X, y = batch[0], batch[1]
                pred = self(X)
                target = y.to(torch.float32)
                if target.dim() == 1:
                    target = target.reshape(-1, 1)
                return pred, target

    class WeightedTensorFeedForwardNet(TensorFeedForwardNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._loss_fn = generate_weighted_loss(nn.BCELoss)()
            self.validation_metrics = {
                "loss": WeightedLoss(self.loss_fn),
                "roc_auc": WeightedROC_AUC(),
            }

        def init_dataset_handler(self, task: law.Task):
            group_datasets = {
                "ttbar": [d for d in task.datasets if d.startswith("tt_")],
            }
            device = next(self.parameters()).device
            all_datasets = getattr(task, "resolved_datasets", task.datasets)

            self.dataset_handler = WeightedRgTensorParquetFileHandler(
                task=task,
                continuous_features=getattr(self, "continuous_features", self.inputs),
                categorical_features=getattr(self, "categorical_features", None),
                batch_transformations=MoveToDevice(device=device),
                # global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_"])],
            )
            self.training_loader, (self.train_validation_loader, self.validation_loader) = self.dataset_handler.init_datasets() #noqa

        def validation_step(self, engine, batch):
            # Set the model to evaluation mode - important for batch normalization and dropout layers
            self.eval()

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors
            # with requires_grad=True
            with torch.no_grad():
                X, y = batch[0], batch[1]
                input_data, weights = X
                pred = self(input_data)
                target = y.to(torch.float32)
                if target.dim() == 1:
                    target = target.reshape(-1, 1)
                return pred, target, {"weight": weights}

    class WeightedTensorFeedForwardNetWithCat(WeightedTensorFeedForwardNet):
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
            self.embedding_dims = 50

            self.init_layers()

        def init_layers(self):
            self.input_layer = InputLayer(
                categorical_inputs=sorted(self.categorical_features, key=str),
                continuous_inputs=sorted(self.inputs, key=str),
                expected_categorical_inputs=embedding_expected_inputs,
                embedding_dim=self.embedding_dims,
            )
            self.norm_layer = nn.BatchNorm1d(len(self.continuous_features))
            self.padding_layer_cat = PaddingLayer(padding_value=self.input_layer.placeholder, mask_value=EMPTY_INT)
            self.padding_layer_cont = PaddingLayer(padding_value=0, mask_value=EMPTY_FLOAT)

            self.linear_relu_stack = nn.Sequential(
                nn.Linear(self.input_layer.ndim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid(),
            )

        def forward(self, X):
            # X is a tuple of (input_data, categorical_features)
            cat_features, cont_features = X
            cont_features = self.padding_layer_cont(cont_features)
            cont_features = self.norm_layer(cont_features.to(torch.float32))
            cat_features = self.padding_layer_cat(cat_features)

            # pass through the embedding layer
            features = self.input_layer(cont_features, cat_features.to(torch.int32))

            # concatenate the continuous and categorical features

            logits = self.linear_relu_stack(features)
            return logits

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

        def to(self, *args, **kwargs):
            self.input_layer.to(*args, **kwargs)
            return super().to(*args, **kwargs)

    class WeightedTensorFeedForwardNetWithCatReducedEmbedding(WeightedTensorFeedForwardNetWithCat):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.embedding_dims = 10
            self.init_layers()
        
        def forward(self, x):
            try:
                output = super().forward(x)
            except Exception as e:
                from IPython import embed
                embed(header=f"Error in forward pass: {e}")
            return output

    class WeightedTensorFeedForwardNetWithCatReducedEmbedding1F(WeightedTensorFeedForwardNetWithCatReducedEmbedding):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.categorical_features = {
                "pair_type",
                # "decay_mode1",
                # "decay_mode2",
                # "lepton1.charge",
                # "lepton2.charge",
                # "has_fatjet",
                # "has_jet_pair",
                "channel_id",
                "year_flag",
            }

            self.embedding_dims = 10
            self.init_layers()

    class TensorFeedForwardNetAdam(TensorFeedForwardNet):
        def init_optimizer(self, learning_rate=0.001, weight_decay=0.00001):
            self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    class DropoutFeedForwardNet(FeedForwardNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.linear_relu_stack = nn.Sequential(

                # nn.Dropout(p=0.2),
                nn.Linear(len(self.inputs), 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                # nn.Dropout(p=0.2),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                # nn.Dropout(p=0.2),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid(),
            )

    class FeedForwardArrow(FeedForwardNet):
        def init_dataset_handler(self, task: law.Task):
            group_datasets = {
                "ttbar": [d for d in task.datasets if d.startswith("tt_")],
            }
            device = next(self.parameters()).device
            self.dataset_handler = FlatArrowParquetFileHandler(
                task=task,
                columns=self.inputs,
                batch_transformations=AkToTensor(device=device),
                global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
            )
