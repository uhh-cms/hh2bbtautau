from __future__ import annotations

__all__ = [
]

from functools import partial

from columnflow.util import MockModule, maybe_import, DotDict, classproperty
from columnflow.types import Callable
from collections.abc import Container

from columnflow.columnar_util import Route, EMPTY_FLOAT, EMPTY_INT

from hbt.ml.torch_utils.functions import (
    get_one_hot, preprocess_multiclass_outputs, WeightedCrossEntropySlice,
)

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")
ak = maybe_import("awkward")
import law

model_clss: DotDict[str, torch.nn.Module] = DotDict()

if not isinstance(torch, MockModule):
    import torch
    # import torch.utils.tensorboard as tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # from ignite.metrics import Loss, ROC_AUC
    import matplotlib.pyplot as plt

    from hbt.ml.torch_models.multi_class import WeightedFeedForwardMultiCls
    from hbt.ml.torch_utils.ignite.metrics import (
        WeightedROC_AUC, WeightedLoss,
        # WeightedBalanced_Acc,
    )
    from hbt.ml.torch_utils.ignite.mixins import IgniteTrainingMixin
    from hbt.ml.torch_utils.transforms import MoveToDevice
    from hbt.ml.torch_utils.datasets.handlers import (
        WeightedTensorParquetFileHandler,
    )
    from hbt.ml.torch_utils.utils import (
        embedding_expected_inputs, expand_columns, get_standardization_parameter,
    )
    from hbt.ml.torch_utils.layers import (
        InputLayer, StandardizeLayer, ResNetPreactivationBlock, DenseBlock, PaddingLayer, RotatePhiLayer,
    )
    from hbt.ml.torch_utils.functions import generate_weighted_loss

    import sklearn
    from ignite.engine import Events

    class BogNet(WeightedFeedForwardMultiCls):
        def __init__(self, *args, **kwargs):

            self.inputs = set(self.categorical_inputs) | set(self.continuous_inputs)

            # targets
            self.categorical_target_map = {
                "hh": 0,
                "tt": 1,
                "dy": 2,
            }

            # build network, get commandline arguments
            self.nodes = kwargs.get("nodes", 128)
            self.activation_functions = kwargs.get("activation_functions", "PReLu")
            self.skip_connection_init = kwargs.get("skip_connection_init", 1)
            self.freeze_skip_connection = kwargs.get("freeze_skip_connection", False)
            self.empty_value = 15
            super().__init__(*args, **kwargs)

            self.padding_layer, self.rotation_layer, self.std_layer, self.input_layer, self.model = self.init_layers()

            # loss function and metrics
            self.label_smoothing_coefficient = 0.05
            self._loss_fn = generate_weighted_loss(
                torch.nn.CrossEntropyLoss,
            )(label_smoothing=self.label_smoothing_coefficient)

            self.validation_metrics["loss"] = WeightedLoss(self.loss_fn)
            self.mode_switch = 0
            self.output_path = "/data/dust/user/wiedersb/"

            self.validation_metrics.update({
                f"loss_cls_{identifier}": WeightedLoss(
                    WeightedCrossEntropySlice(cls_index=idx, label_smoothing=self.label_smoothing_coefficient),
                )
                for identifier, idx in self.categorical_target_map.items()
            })

            self.validation_metrics.update({
                f"roc_auc_cls_{identifier}": WeightedROC_AUC(
                    output_transform=partial(
                        preprocess_multiclass_outputs,
                        multi_class="ovr",
                        average=None,
                    ),
                    target_class_idx=idx,
                )
                for identifier, idx in self.categorical_target_map.items()
            })

            # trainings settings
            self.training_epoch_length_cutoff = 200
            self.training_weight_cutoff = 0.05

            # remove layers that comes due to inheritance
            # TODO clean up, this is only a monkey patch
            if hasattr(self, "linear_relu_stack"):
                del self.linear_relu_stack
            # del self.norm_layer
            self.custom_hooks.append("add_checkpoints")
            self.custom_hooks.append("perform_scheduler_step")

        @classmethod
        def _process_columns(cls, columns: Container[str]) -> list[Route]:
            final_set = set()
            final_set.update(*list(map(Route, law.util.brace_expand(obj)) for obj in columns))
            return sorted(final_set, key=str)

        @classproperty
        def categorical_inputs(cls) -> list[str]:
            columns = {
                "pair_type",
                "decay_mode1",
                "decay_mode2",
                "lepton1.charge",
                "lepton2.charge",
                "has_fatjet",
                "has_jet_pair",
                "year_flag",
            }
            return cls._process_columns(columns)

        @classproperty
        def continuous_inputs(cls) -> list[str]:
            columns = {
                "lepton1.{px,py,pz,energy,mass}",
                "lepton2.{px,py,pz,energy,mass}",
                "bjet1.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
                "bjet2.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
                "fatjet.{px,py,pz,energy,mass}",
            }
            return cls._process_columns(columns)

        def state_dict(self, *args, **kwargs):
            return self.model.state_dict(*args, **kwargs)

        def load_state_dict(self, *args, **kwargs):
            return self.model.load_state_dict(*args, **kwargs)

        def perform_scheduler_step(self):
            if self.scheduler:
                def do_step(engine, logger=self.logger):
                    self.scheduler.step()
                    self.logger.info(f"Current LR: {self.scheduler.get_last_lr()}")

                self.train_evaluator.add_event_handler(
                    event_name="EPOCH_COMPLETED",
                    handler=do_step,
                )

        def add_checkpoints(self):
            def save_checkpoint(engine):
                epoch = engine.state.epoch
                outpath = f"/data/dust/user/wiedersb/model_epoch{epoch}.pt"
                self.logger.info(f"saving checkpoint for epoch {epoch} in {outpath}")
                torch.save(self.model, outpath)

            self.trainer.add_event_handler(
                Events.EPOCH_COMPLETED(every=1),
                save_checkpoint,
            )

        @property
        def num_cat(self):
            return len(self.categorical_inputs)

        @property
        def num_cont(self):
            return len(self.continuous_inputs)

        @property
        def num_inputs(self):
            return self.num_cat + self.num_cont

        def init_layers(self):
            # helper where all layers are defined
            # std layers are filled when statitics are known
            padding_layer = PaddingLayer(padding_value=0, mask_value=EMPTY_FLOAT)

            # only continous_inputs are rotated
            rotation_layer = RotatePhiLayer(
                list(map(lambda x: x.column, self.continous_inputs)),
                rotate_columns=["bjet1", "bjet2", "fatjet", "lepton1", "lepton2", "rotated_PuppiMET"],
                ref_phi_columns=["lepton1", "lepton2"],
                activate=False,
            )

            std_layer = StandardizeLayer(
                None,
                None,
            )

            input_layer = InputLayer(
                continuous_inputs=self.continuous_inputs,
                categorical_inputs=self.categorical_inputs,
                embedding_dim=3,
                expected_categorical_inputs=embedding_expected_inputs,
                empty=self.empty_value,
                std_layer=std_layer,
                rotation_layer=rotation_layer,
                padding_continous_layer=padding_layer,
            )

            model = torch.nn.Sequential(
                input_layer,
                DenseBlock(input_nodes = input_layer.ndim, output_nodes = self.nodes, activation_functions=self.activation_functions), # noqa
                ResNetPreactivationBlock(self.nodes, self.activation_functions, self.skip_connection_init, self.freeze_skip_connection), # noqa
                ResNetPreactivationBlock(self.nodes, self.activation_functions, self.skip_connection_init, self.freeze_skip_connection), # noqa
                ResNetPreactivationBlock(self.nodes, self.activation_functions, self.skip_connection_init, self.freeze_skip_connection), # noqa
                ResNetPreactivationBlock(self.nodes, self.activation_functions, self.skip_connection_init, self.freeze_skip_connection), # noqa
                torch.nn.Linear(self.nodes, len(self.categorical_target_map)),
                # no softmax since this is already part of loss
            )
            return padding_layer, rotation_layer, std_layer, input_layer, model

        def train_step(self, engine, batch):
            self.train()

            # Compute prediction and loss
            (categorical_x, continous_x), y = batch
            self.optimizer.zero_grad()

            # extra step normalize embedding vectors to have unit norm of 1, increases stability
            self.input_layer.embedding_layer.normalize_embeddings()

            pred = self(categorical_x, continous_x)
            target = y.to(torch.float32)
            if target.dim() == 1:
                target = target.reshape(-1, 1)

            loss = self.loss_fn(pred, target)
            # Backpropagation
            loss.backward()
            self.optimizer.step()

            return loss.item()

        def validation_step(self, engine, batch):
            self.eval()
            # Set the model to evaluation mode - important for batch normalization and dropout layers

            # if engine.state.iteration > self.max_val_epoch_length * (engine.state.epoch + 1):
            #     engine.terminate_epoch()

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors with
            # requires_grad=True
            with torch.no_grad():
                (categorical_x, continous_x, weights), y = batch
                pred = self(categorical_x, continous_x)

                if y.dim() == 1:
                    y = y.reshape(-1, 1)
                y = y.to(torch.float32)
                return pred, y, {"weight": weights.reshape(-1, 1)}

        def to(self, *args, **kwargs):
            # helper to move all customlayers to given device
            self.std_layer = self.std_layer.to(*args, **kwargs)
            self.input_layer = self.input_layer.to(*args, **kwargs)
            self.model = self.model.to(*args, **kwargs)
            return super().to(*args, **kwargs)

        def setup_preprocessing(self):
            # extract dataset std and mean from dataset
            # extraction happens form no oversampled dataset
            mean, std = [], []
            for _input in self.continuous_inputs:
                input_statitics = self.dataset_statistics[_input.column]
                mean.append(torch.from_numpy(input_statitics["mean"]))
                std.append(torch.from_numpy(input_statitics["std"]))

            mean, std = torch.concat(mean), torch.concat(std)
            # set up standardization layer
            self.std_layer.update_buffer(
                mean.float(),
                std.float(),
            )

        def logging(self, *args, **kwargs):
            # output histogram
            for target, index in self.categorical_target_map.items():
                # apply softmax to prediction
                logit = kwargs["prediction"]

                self.writer.add_histogram(
                    f"output_logit_{target}",
                    logit[:, index],
                    self.trainer.state.iteration,
                )

        def init_dataset_handler(self, task: law.Task, device: str = "cpu") -> None:
            all_datasets = getattr(task, "resolved_datasets", task.datasets)
            group_datasets = {
                "ttbar": [d for d in all_datasets if d.startswith("tt_")],
                "dy": [d for d in all_datasets if d.startswith("dy_")],
            }

            self.dataset_handler = WeightedTensorParquetFileHandler(
                task=task,
                continuous_features=getattr(self, "continuous_features", self.continuous_inputs),
                categorical_features=getattr(self, "categorical_features", self.categorical_inputs),
                batch_transformations=MoveToDevice(device=device),
                # global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
                categorical_target_transformation=partial(get_one_hot, nb_classes=3),
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_", "dy_"])],
            )

            self.training_loader, (self.train_validation_loader, self.validation_loader) = self.dataset_handler.init_datasets() # noqa

            # define lenght of training epoch
            self.max_epoch_length = self._calculate_max_epoch_length(
                self.training_loader,
                cutoff=self.training_epoch_length_cutoff,
                weight_cutoff=self.training_weight_cutoff,
            )

            # get statistics for standardization from training dataset without oversampling
            self.dataset_statistics = get_standardization_parameter(
                self.train_validation_loader.data_map, self.continous_inputs,
            )

        def control_plot_1d(self):
            import matplotlib.pyplot as plt
            d = {}
            for dataset, file_handler in self.training_loader.data_map.items():
                d[dataset] = ak.concatenate(list(map(lambda x: x.data, file_handler)))

            for cat in self.dataset_handler.categorical_features:
                plt.clf()
                data = []
                labels = []
                for dataset, arrays in d.items():
                    data.append(Route(cat).apply(arrays))
                    labels.append(dataset)
                plt.hist(data, histtype="barstacked", alpha=0.5, label=labels)
                plt.xlabel(cat)
                plt.legend()
                plt.savefig(f"{cat}_all.png")

        def init_optimizer(self, learning_rate=1e-2, weight_decay=1e-5) -> None:
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=3, gamma=0.5)

        def forward(self, *inputs):
            return self.model(inputs)

    class UpdatedBogNet(BogNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.training_epoch_length_cutoff = 10000
            if "perform_scheduler_step" not in self.custom_hooks:
                self.custom_hooks.append("perform_scheduler_step")

            # disable checkpoints for now since they point to Bogdan specifically
            if "add_checkpoints" in self.custom_hooks:
                self.custom_hooks.remove("add_checkpoints")

        def log_graph(self, engine):
            input_data = engine.state.batch[0]
            self.writer.add_graph(self, input_data)

        def perform_scheduler_step(self):
            if hasattr(self, "scheduler") and self.scheduler:
                def do_step(engine, logger=self.logger):
                    self.scheduler.step()
                    logger.info(f"Performing scheduler step, last lr: {self.scheduler.get_last_lr()}")

                self.train_evaluator.add_event_handler(
                    event_name="EPOCH_COMPLETED",
                    handler=do_step,
                )

        def init_layers(self):
            # helper where all layers are defined
            # std layers are filled when statitics are known
            std_layer = StandardizeLayer()

            continuous_padding = PaddingLayer(padding_value=0, mask_value=EMPTY_FLOAT)
            categorical_padding = PaddingLayer(padding_value=self.empty_value, mask_value=EMPTY_INT)
            rotation_layer = RotatePhiLayer(columns=list(map(str, self.continuous_inputs)))

            input_layer = InputLayer(
                continuous_inputs=self.continuous_inputs,
                categorical_inputs=self.categorical_inputs,
                embedding_dim=3,
                expected_categorical_inputs=embedding_expected_inputs,
                empty=self.empty_value,
                std_layer=std_layer,
                rotation_layer=rotation_layer,
                padding_categorical_layer=categorical_padding,
                padding_continous_layer=continuous_padding,
            )

            model = torch.nn.Sequential(
                input_layer,
                DenseBlock(input_nodes = input_layer.ndim, output_nodes = self.nodes, activation_functions=self.activation_functions), # noqa
                ResNetPreactivationBlock(self.nodes, self.activation_functions, self.skip_connection_init, self.freeze_skip_connection), # noqa
                ResNetPreactivationBlock(self.nodes, self.activation_functions, self.skip_connection_init, self.freeze_skip_connection), # noqa
                ResNetPreactivationBlock(self.nodes, self.activation_functions, self.skip_connection_init, self.freeze_skip_connection), # noqa
                torch.nn.Linear(self.nodes, len(self.categorical_target_map)),
                # no softmax since this is already part of loss
            )
            return continuous_padding, rotation_layer, std_layer, input_layer, model

        def init_optimizer(self, learning_rate=1e-2, weight_decay=1e-5) -> None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1, gamma=0.9)

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
        ) -> None:
            all_datasets = datasets or getattr(task, "resolved_datasets", task.datasets)
            group_datasets = {
                "ttbar": [d for d in all_datasets if d.startswith("tt_")],
                "dy": [d for d in all_datasets if d.startswith("dy_")],
            }

            self.dataset_handler = WeightedTensorParquetFileHandler(
                task=task,
                inputs=inputs,
                continuous_features=getattr(self, "continuous_features", self.continuous_inputs),
                categorical_features=getattr(self, "categorical_features", self.categorical_inputs),
                batch_transformations=MoveToDevice(device=device),
                # global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
                categorical_target_transformation=partial(get_one_hot, nb_classes=3),
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_", "dy_"])],
                extract_dataset_paths_fn=extract_dataset_paths_fn,
                extract_probability_fn=extract_probability_fn,
                # categorical_target_transformation=,
                # data_type_transformation=AkToTensor,
            )

            self.training_loader, (self.train_validation_loader, self.validation_loader) = self.dataset_handler.init_datasets()  # noqa

            # define lenght of training epoch
            self.max_epoch_length = self._calculate_max_epoch_length(
                self.training_loader,
                cutoff=self.training_epoch_length_cutoff,
                weight_cutoff=self.training_weight_cutoff,
            )

            # get statistics for standardization from training dataset without oversampling
            self.dataset_statistics = get_standardization_parameter(self.train_validation_loader.data_map, self.continuous_inputs)

        def setup_preprocessing(self, device: str | None = None):
            # extract dataset std and mean from dataset
            # extraction happens form no oversampled dataset
            mean, std = [], []
            if not device:
                device = next(self.model.parameters()).device
            for _input in self.continuous_inputs:
                input_statistics = self.dataset_statistics[_input.column]
                mean.append(torch.from_numpy(input_statistics["mean"]))
                std.append(torch.from_numpy(input_statistics["std"]))

            mean, std = torch.concat(mean).to(device), torch.concat(std).to(device)
            # set up standardization layer
            self.std_layer.update_buffer(
                mean.float(),
                std.float(),
            )

        def validation_step(self, engine, batch):
            self.eval()
            # Set the model to evaluation mode - important for batch normalization and dropout layers

            # if engine.state.iteration > self.max_val_epoch_length * (engine.state.epoch + 1):
            #     engine.terminate_epoch()

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors with
            # requires_grad=True
            with torch.no_grad():
                (categorical_x, continous_x, weights), y = batch

                pred = self(categorical_x, continous_x)

                if y.dim() == 1:
                    y = y.reshape(-1, 1)
                y = y.to(torch.float32)

                return pred, y, {"weight": weights}

        def train_step(self, engine, batch):
            # from IPython import embed; embed(header="string - 149 in bognet.py ")
            self.train()

            # Compute prediction and loss
            (categorical_x, continous_x), y = batch
            self.optimizer.zero_grad()

            # replace missing values with empty_fill, convert to expected type

            pred = self(categorical_x, continous_x)
            target = y.to(torch.float32)
            if target.dim() == 1:
                target = target.reshape(-1, 1)

            loss = self.loss_fn(pred, target)
            # Backpropagation
            loss.backward()
            self.optimizer.step()

            return loss.item()

    class ShapModel(torch.nn.Module):
        # dummy Model class to give interface for single tensor inputs, since SHAP expect this kind of input tensor
        def __init__(self, model):
            super().__init__()
            self.model = model.model
            self.num_cont = len(model.continuous_inputs)
            self.num_cat = len(model.categorical_inputs)

        def forward(self, x):
            cont, cat = (
                torch.as_tensor(x[:, self.num_cat:], dtype=torch.float32),
                torch.as_tensor(x[:, :self.num_cat], dtype=torch.int32),
            )
            return self.model((cat, cont))
