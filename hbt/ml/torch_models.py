from __future__ import annotations

__all__ = [
    "model_clss",
]

from functools import partial

from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import Any
from collections.abc import Container

from columnflow.columnar_util import Route, EMPTY_FLOAT, EMPTY_INT

from hbt.ml.torch_utils.functions import get_one_hot, WeightedCrossEntropyLoss

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")
ak = maybe_import("awkward")
import law

model_clss: DotDict[str, torch.nn.Module] = DotDict()

if not isinstance(torch, MockModule):
    from torch import nn
    from torch.optim import Adam
    from torch.utils.tensorboard import SummaryWriter
    from ignite.metrics import Loss, ROC_AUC

    from hbt.ml.torch_utils.transforms import AkToTensor, PreProcessFloatValues
    from hbt.ml.torch_utils.datasets.handlers import (
        FlatListRowgroupParquetFileHandler, FlatArrowParquetFileHandler,
        WeightedFlatListRowgroupParquetFileHandler,
    )
    from hbt.ml.torch_utils.utils import (
        embedding_expected_inputs, LookUpTable, CategoricalTokenizer,
    )
    from hbt.ml.torch_utils.ignite.mixins import IgniteTrainingMixin, IgniteEarlyStoppingMixin

    class NetworkBase(nn.Module):
        def __init__(self, *args, tensorboard_path: str | None = None, logger: Any | None = None, **kwargs):
            super().__init__()
            self.writer = None
            self.logger = logger or law.logger.get_logger(__name__)
            if tensorboard_path:
                self.writer = SummaryWriter(tensorboard_path)
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
                nn.Linear(512, 1),
                nn.Sigmoid(),
            )

            self.logger.info("Constructing loss and optimizer")
            self._loss_fn = nn.BCELoss()
            self.validation_metrics = {
                "loss": Loss(self.loss_fn),
                "roc_auc": ROC_AUC(),
            }

        def init_optimizer(self, learning_rate=1e-3, weight_decay=1e-5) -> None:
            self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
            empty_fill_val: float = -10,
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
            empty_mask = input_data == mask_value
            input_data[empty_mask] = empty_fill_val
            return input_data.to(dtype)

        def forward(self, x):
            input_data = self._handle_input(x)
            logits = self.linear_relu_stack(input_data)
            return logits

    class DropoutFeedForwardNet(FeedForwardNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.linear_relu_stack = nn.Sequential(
                nn.BatchNorm1d(len(self.inputs)),
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

        def _calculate_max_epoch_length(self, composite_loader, weight_cutoff: float = 0.05, cutoff: int = 2000):
            global_max = 0
            max_key = None

            batch_comp = getattr(composite_loader.batcher, "_batch_composition", None)
            if not batch_comp:
                batch_comp = {key: 1. for key in composite_loader.data_map.keys()}
            for key, batchsize in batch_comp.items():
                weights = composite_loader.weight_dict[key]
                if isinstance(weights, float):
                    total_length = sum([len(x) for x in composite_loader.data_map[key]])
                    local_max = np.ceil(total_length / batchsize)
                    if local_max > global_max:
                        max_key = key
                        global_max = local_max

                elif isinstance(weights, dict):
                    for subkey, weight in weights.items():
                        total_length = sum([len(x) for x in composite_loader.data_map[subkey]])
                        submax = np.ceil(total_length / batchsize / weight)
                        if submax > global_max and weight >= weight_cutoff:
                            global_max = submax
                            max_key = subkey
            self.logger.info(f"epoch dominated by  '{max_key}': expect {global_max} batches/iteration")
            return np.min([global_max, cutoff])

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
            self.max_epoch_length = self._calculate_max_epoch_length(self.training_loader)
            # self.max_val_epoch_length = self._calculate_max_epoch_length(self.validation_loader)

    class WeightedFeedForwardMultiCls(FeedForwardMultiCls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._loss_fn = WeightedCrossEntropyLoss()
            self.validation_metrics = {
                "loss": Loss(self.loss_fn),
                # "roc_auc": ROC_AUC(),
            }

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

        def init_dataset_handler(self, task: law.Task):
            all_datasets = getattr(task, "resolved_datasets", task.datasets)
            group_datasets = {
                "ttbar": [d for d in all_datasets if d.startswith("tt_")],
                "dy": [d for d in all_datasets if d.startswith("dy_")],
            }
            device = next(self.parameters()).device

            self.dataset_handler = WeightedFlatListRowgroupParquetFileHandler(
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
            self.training_loader, (self.train_validation_loader, self.validation_loader) = self.dataset_handler.init_datasets()  # noqa
            self.max_epoch_length = self._calculate_max_epoch_length(self.training_loader)

            dm = self.validation_loader.data_map
            batcher = self.validation_loader.batcher
            self.max_val_epoch_length = np.ceil(sum(map(len, dm)) / batcher.batch_size)

    class WeightedResNet(WeightedFeedForwardMultiCls):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.floating_inputs = sorted(self.inputs)
            n_floating_inputs = len(self.inputs)
            self.categorical_inputs = sorted({
                "pair_type",
                "decay_mode1",
                "decay_mode2",
                "lepton1.charge",
                "lepton2.charge",
                "has_fatjet",
                "has_jet_pair",
                "year_flag",
            })

            # update list of inputs
            self.inputs |= set(self.categorical_inputs)
            local_max = max([len(embedding_expected_inputs[x]) for x in self.categorical_inputs])
            self.placeholder = 15
            array = torch.stack(
                [
                    nn.functional.pad(
                        torch.tensor(embedding_expected_inputs[x]),
                        (0, local_max - len(embedding_expected_inputs[x])),
                        mode="constant",
                        # pad with a value that exists in the embedding
                        value=embedding_expected_inputs[x][0],
                    )
                    for x in self.categorical_inputs
                ],
            )
            self.min, self.look_up_table = LookUpTable(array, placeholder=self.placeholder)
            self.tokenizer = CategoricalTokenizer(
                self.look_up_table,
                self.min,
            )

            self.embeddings = torch.nn.Embedding(
                self.tokenizer.num_dim,
                50,
            )

            self.floating_layer = nn.BatchNorm1d(n_floating_inputs)
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(n_floating_inputs + len(self.categorical_inputs) * 50, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 3),
            )

        def forward(self, x):
            floating_inputs = self._handle_input(x, self.floating_inputs)

            categorical_inputs = self._handle_input(
                x,
                self.categorical_inputs,
                dtype=torch.int32,
                empty_fill_val=self.placeholder,
                mask_value=EMPTY_INT,
            )

            normed_floating_inputs = self.floating_layer(floating_inputs)

            # tokenize categorical inputs
            tokenized_inputs = self.tokenizer(categorical_inputs)

            # embed categorical inputs
            cat_inputs = self.embeddings(tokenized_inputs)
            # concatenate with other inputs
            # flatten new embedding space
            flat_cat_inputs = cat_inputs.flatten(start_dim=1)

            input_data = torch.cat([normed_floating_inputs, flat_cat_inputs], axis=-1).to(torch.float32)
            logits = self.linear_relu_stack(input_data)
            return logits

        def to(self, *args, **kwargs):
            self.tokenizer = self.tokenizer.to(*args, **kwargs)
            return super().to(*args, **kwargs)

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

        def __call__(self, x):
            input_data = self._handle_input(x)
            logits = self.linear_relu_stack1(input_data.to(torch.float32))
            logits = self.linear_relu_stack2(logits)
            return logits

    class ResNet(
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

            cat_inputs = []

            columns = [
                "lepton1.{px,py,pz,energy,mass}",
                "lepton2.{px,py,pz,energy,mass}",
                "bjet1.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
                "bjet2.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
                "fatjet.{px,py,pz,energy,mass}",
            ]
            self.inputs = set()
            self.inputs.update(*list(map(Route, law.util.brace_expand(obj)) for obj in columns))

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
                nn.Linear(512, 1),
                nn.Sigmoid(),
            )

            self.logger.info("Constructing loss and optimizer")
            self._loss_fn = nn.BCELoss()
            self.validation_metrics = {
                "loss": Loss(self.loss_fn),
                "roc_auc": ROC_AUC(),
            }




    model_clss["feedforward"] = FeedForwardNet
    model_clss["feedforward_arrow"] = FeedForwardArrow
    model_clss["feedforward_multicls"] = FeedForwardMultiCls
    model_clss["weighted_feedforward_multicls"] = WeightedFeedForwardMultiCls
    model_clss["deepfeedforward_multicls"] = DeepFeedForwardMultiCls
    model_clss["feedforward_dropout"] = DropoutFeedForwardNet
    model_clss["resnet"] = ResNet
    model_clss["weighted_resnet"] = WeightedResNet
