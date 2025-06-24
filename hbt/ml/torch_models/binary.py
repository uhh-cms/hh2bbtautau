from __future__ import annotations

__all__ = [
]

from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import Any, Callable
from collections.abc import Container

from columnflow.columnar_util import Route, EMPTY_FLOAT, EMPTY_INT

from hbt.ml.torch_utils.utils import get_standardization_parameter

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")
ak = maybe_import("awkward")
import law

WeightedTensorFeedForwardNet = MockModule("WeightedTensorFeedForwardNet")
if not isinstance(torch, MockModule):
    from torch import nn
    from torch.optim import Adam, AdamW
    from torch.utils.tensorboard import SummaryWriter
    from ignite.metrics import Loss, ROC_AUC

    from hbt.ml.torch_utils.transforms import (
        AkToTensor, PreProcessFloatValues, MoveToDevice, TokenizeCategories,
    )
    from hbt.ml.torch_utils.datasets.handlers import (
        FlatListRowgroupParquetFileHandler, FlatArrowParquetFileHandler,
        WeightedFlatListRowgroupParquetFileHandler,
        RgTensorParquetFileHandler, WeightedRgTensorParquetFileHandler,
        WeightedTensorParquetFileHandler,
    )
    from hbt.ml.torch_utils.ignite.metrics import (
        WeightedROC_AUC, WeightedLoss,
    )
    from hbt.ml.torch_utils.utils import (
        embedding_expected_inputs,
    )
    from hbt.ml.torch_utils.functions import generate_weighted_loss
    from hbt.ml.torch_utils.ignite.mixins import IgniteTrainingMixin, IgniteEarlyStoppingMixin
    from hbt.ml.torch_utils.layers import PaddingLayer, InputLayer, StandardizeLayer

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
            self.inputs: set[Route] = set()
            self.inputs.update(*list(map(Route, law.util.brace_expand(obj)) for obj in columns))

            self.init_layers()
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

        def init_layers(self):
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

        def init_dataset_handler(
            self,
            task: law.Task,
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
            }
            device = next(self.parameters()).device

            self.dataset_handler = FlatListRowgroupParquetFileHandler(
                task=task,
                inputs=inputs,
                columns=self.inputs,
                batch_transformations=AkToTensor(device=device),
                # global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_"])],
                extract_dataset_paths_fn=extract_dataset_paths_fn,
                extract_probability_fn=extract_probability_fn,
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
        def init_dataset_handler(
            self,
            task: law.Task,
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
            }
            device = next(self.parameters()).device

            self.dataset_handler = RgTensorParquetFileHandler(
                task=task,
                inputs=inputs,
                continuous_features=getattr(self, "continuous_features", self.inputs),
                batch_transformations=MoveToDevice(device=device),
                # global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_"])],
                extract_dataset_paths_fn=extract_dataset_paths_fn,
                extract_probability_fn=extract_probability_fn,
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

        def init_dataset_handler(
            self,
            task: law.Task,
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
            }
            device = next(self.parameters()).device

            self.dataset_handler = WeightedTensorParquetFileHandler(
                task=task,
                inputs=inputs,
                continuous_features=getattr(self, "continuous_features", self.inputs),
                categorical_features=getattr(self, "categorical_features", None),
                batch_transformations=MoveToDevice(device=device),
                # global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_"])],
                extract_dataset_paths_fn=extract_dataset_paths_fn,
                extract_probability_fn=extract_probability_fn,
            )
            self.training_loader, (self.train_validation_loader, self.validation_loader) = self.dataset_handler.init_datasets()  # noqa

            self.dataset_statistics = get_standardization_parameter(self.train_validation_loader.data_map, self.inputs)

        def init_layers(self):
            self.std_layer = StandardizeLayer()
            super().init_layers()

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
                input_data, weights = X[:-1], X[-1]
                if isinstance(input_data, list) and len(input_data) == 1:
                    input_data = input_data[0]
                pred = self(input_data)
                target = y.to(torch.float32)
                if target.dim() == 1:
                    target = target.reshape(-1, 1)
                return pred, target, {"weight": weights}

    class WeightedTensorFeedForwardNetWithCat(WeightedTensorFeedForwardNet):
        def __init__(self, *args, **kwargs):
            self.categorical_features = {
                "pair_type",
                # "decay_mode1",
                "decay_mode2",
                "lepton1.charge",
                "lepton2.charge",
                "has_fatjet",
                "has_jet_pair",
                "year_flag",
            }
            self.embedding_dims = 50

            super().__init__(*args, **kwargs)

            self.continuous_features = self.inputs

            self.init_layers()

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

        def init_optimizer(self, learning_rate=1e-3, weight_decay=1e-5) -> None:
            self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        def forward(self, X):
            # X is a tuple of (input_data, categorical_features)

            cat_features, cont_features = X
            cont_features = self.padding_layer_cont(cont_features)
            cont_features = self.std_layer(cont_features.to(torch.float32))
            cat_features = self.padding_layer_cat(cat_features)

            # pass through the embedding layer
            features = self.input_layer((cat_features.to(torch.int32), cont_features))

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
            self.std_layer = self.std_layer.to(*args, **kwargs)
            self.input_layer = self.input_layer.to(*args, **kwargs)
            return super().to(*args, **kwargs)

    class WeightedTensorFeedForwardNetWithCatOutsourceTokens(WeightedTensorFeedForwardNetWithCat):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def init_layers(self):
            self.std_layer = StandardizeLayer()

            self.tokenizer = TokenizeCategories(
                categories=sorted(self.categorical_features, key=str),
                expected_categorical_inputs=embedding_expected_inputs,
                cat_feature_idx=0,
            )
            self.input_layer = InputLayer(
                continuous_inputs=sorted(self.inputs, key=str),
                categorical_inputs=sorted(self.categorical_features, key=str),
                embedding_dim=self.embedding_dims,
                category_dims=self.tokenizer.num_dim,
            )
            self.padding_layer_cat = PaddingLayer(padding_value=self.input_layer.empty, mask_value=EMPTY_INT)
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

        def init_dataset_handler(
            self,
            task: law.Task,
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
            }
            device = next(self.parameters()).device

            self.dataset_handler = WeightedRgTensorParquetFileHandler(
                task=task,
                inputs=inputs,
                continuous_features=getattr(self, "continuous_features", self.inputs),
                categorical_features=getattr(self, "categorical_features", None),
                batch_transformations=MoveToDevice(device=device),
                input_data_transform=self.tokenizer,
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_"])],
                extract_dataset_paths_fn=extract_dataset_paths_fn,
                extract_probability_fn=extract_probability_fn,
            )
            self.training_loader, (self.train_validation_loader, self.validation_loader) = self.dataset_handler.init_datasets() # noqa
            self.dataset_statistics = get_standardization_parameter(self.train_validation_loader.data_map, self.inputs)

    class WeightedTensorFeedForwardNetWithCatReducedEmbedding(WeightedTensorFeedForwardNetWithCat):

        def __init__(self, *args, **kwargs):

            super().__init__(*args, **kwargs)
            self.embedding_dims = 10

            self.init_layers()

    class WeightedTensorFeedForwardNetWithCatReducedEmbedding1F(WeightedTensorFeedForwardNetWithCatOutsourceTokens):

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
                # "channel_id",
                "year_flag",
            }

            self.embedding_dims = 10
            self.init_layers()
            self.custom_hooks.append("perform_scheduler_step")

        def perform_scheduler_step(self):
            if self.scheduler:
                def do_step(engine, logger=self.logger):
                    logger.info(f"Performing scheduler step")
                    self.scheduler.step()

                self.train_evaluator.add_event_handler(
                    event_name="EPOCH_COMPLETED",
                    handler=do_step,
                )

        def init_optimizer(self, learning_rate=1e-3, weight_decay=1e-5) -> None:
            embedding_params = {x for name, x in self.named_parameters() if "embedding" in name and not "bias" in name}
            other_params = {x for x in self.parameters() if not x in embedding_params}
            self.optimizer = AdamW(
                [
                    {
                        "params": list(other_params),
                    },
                    {
                        "params": list(embedding_params),
                        "lr": learning_rate * 10,
                        "weight_decay": weight_decay * 10,
                    },
                ],
                lr=learning_rate, weight_decay=weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=4,
                gamma=0.9,
            )

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
            # from IPython import embed
            # embed(header=f"check back propagation in class {self.__class__.__name__}")
            # Backpropagation
            loss.backward()
            self.optimizer.step()

            return loss.item()

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
        def init_dataset_handler(
            self,
            task: law.Task,
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
            }
            device = next(self.parameters()).device
            self.dataset_handler = FlatArrowParquetFileHandler(
                task=task,
                inputs=inputs,
                columns=self.inputs,
                batch_transformations=AkToTensor(device=device),
                global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
                extract_dataset_paths_fn=extract_dataset_paths_fn,
                extract_probability_fn=extract_probability_fn,
            )
