from __future__ import annotations

__all__ = [
]

from functools import partial

from columnflow.util import MockModule, maybe_import, DotDict
from columnflow.types import Any
from collections.abc import Container

from columnflow.columnar_util import Route, EMPTY_FLOAT, EMPTY_INT

from hbt.ml.torch_utils.functions import (
    get_one_hot, preprocess_multiclass_outputs,
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

    from hbt.ml.torch_models.binary import NetworkBase
    from hbt.ml.torch_models.multi_class import WeightedFeedForwardMultiCls, FeedForwardMultiCls
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
        embedding_expected_inputs, LookUpTable, CategoricalTokenizer, expand_columns, get_standardization_parameter,
    )
    from hbt.ml.torch_utils.ignite.mixins import IgniteTrainingMixin, IgniteEarlyStoppingMixin
    from hbt.ml.torch_utils.layers import PaddingLayer, InputLayer, StandardizeLayer, ResNetBlock

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

    class WeightedResnetNoDropout(WeightedResNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.floating_inputs = set(self.floating_inputs)
            self.floating_inputs |= {
                f"{field}.{prop}"
                for field in ("htt", "hbb", "htthbb")
                for prop in ("px", "py", "pz", "energy", "mass")
            }
            self.inputs |= self.floating_inputs
            self.floating_inputs = sorted([str(x) for x in self.floating_inputs])

            self.floating_layer = nn.BatchNorm1d(len(self.floating_inputs))
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(len(self.floating_inputs) + len(self.categorical_inputs) * 50, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 3),
            )

            self.training_epoch_length_cutoff = 5000
            self.training_weight_cutoff = 0.1

    class WeightedResnetTest(WeightedResnetNoDropout):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.validation_metrics["loss"] = WeightedLoss(self.loss_fn)

            self.validation_metrics.update({
                f"loss_cls_{identifier}": WeightedLoss(
                    WeightedCrossEntropySlice(cls_index=idx),
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

            self.training_epoch_length_cutoff = 2000
            # self.val_epoch_length_cutoff = 100

        def init_optimizer(self, learning_rate=0.001, weight_decay=0.00001):
            self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    class WeightedResnetTest2(WeightedResnetTest):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.training_epoch_length_cutoff = 100
            self.training_logger_interval = 1
            # self.val_epoch_length_cutoff = 100

        def init_dataset_handler(self, task):
            super().init_dataset_handler(task)

            # from IPython import embed
            # embed(header=f"initialized datasets for class {self.__class__.__name__}")

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
            # from IPython import embed
            # embed(header=f"in training step of class {self.__class__.__name__}")
            # Backpropagation
            loss.backward()
            self.optimizer.step()

            return loss.item()

        def validation_step(self, engine, batch):
            output = super().validation_step(engine=engine, batch=batch)
            return output

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

    class BogNet(WeightedFeedForwardMultiCls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # inputs
            # categories

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

            self.categorical_target_map = {
                "hh": 0,
                "tt": 1,
                "dy": 2,
            }

            # continuous inputs
            self.continous_inputs = expand_columns(
                "lepton1.{px,py,pz,energy,mass}",
                "lepton2.{px,py,pz,energy,mass}",
                "bjet1.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
                "bjet2.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
                "fatjet.{px,py,pz,energy,mass}",
            )
            self.inputs = set(self.categorical_inputs) | set(self.continous_inputs)

            self.nodes = kwargs.get("nodes", 256)
            self.activation_functions = kwargs.get("activation_functions", "LeakyReLu")
            self.skip_connection_init = kwargs.get("skip_connection_init", 0)
            self.freeze_skip_connection = kwargs.get("freeze_skip_connection", True)

            # layer layout
            self.training_epoch_length_cutoff = 2000
            self.training_weight_cutoff = 0.05
            self.placeholder = 15
            self.std_layer, self.input_layer, self.model = self._build_network()
            self.linear_relu_stack = None
            # loss,
            self._loss_fn = nn.CrossEntropyLoss()
            self.validation_metrics = {
                # "unweighted_loss": Loss(self.loss_fn),
                "loss": WeightedLoss(self.loss_fn),
            }

            # self.cemb = CombinedEmbeddings(
            #     self.categorical_inputs,
            #     embedding_expected_inputs,
            #     [10 for i in self.categorical_inputs],
            # )

        def _build_network(self):
            std_layer = StandardizeLayer(
                None,
                None,
            )

            input_layer = InputLayer(
                self.continous_inputs,
                self.categorical_inputs,
                embedding_dim=1,
                expected_categorical_inputs=embedding_expected_inputs,
                placeholder=self.placeholder,
            )

            model = nn.Sequential(
                torch.nn.Linear(input_layer.ndim, self.nodes),
                torch.nn.BatchNorm1d(self.nodes),
                torch.nn.LeakyReLU(),
                ResNetBlock(self.nodes, self.activation_functions, self.skip_connection_init, self.freeze_skip_connection), # noqa
                ResNetBlock(self.nodes, self.activation_functions, self.skip_connection_init, self.freeze_skip_connection), # noqa
                ResNetBlock(self.nodes, self.activation_functions, self.skip_connection_init, self.freeze_skip_connection), # noqa
                torch.nn.Linear(self.nodes, len(self.categorical_target_map)),
            )
            return std_layer, input_layer, model

        def to(self, *args, **kwargs):
            self.std_layer = self.std_layer.to(*args, **kwargs)
            self.input_layer = self.input_layer.to(*args, **kwargs)
            self.model = self.model.to(*args, **kwargs)
            return super().to(*args, **kwargs)

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

        def setup_preprocessing(self):
            # extract dataset std and mean from dataset
            # extraction happens form no oversampled dataset
            mean, std = [], []
            for _input in self.continous_inputs:
                input_statitics = self.dataset_statitics[_input.column]
                mean.append(torch.from_numpy(input_statitics["mean"]))
                std.append(torch.from_numpy(input_statitics["std"]))

            mean, std = torch.concat(mean), torch.concat(std)
            # set up standardization layer
            self.std_layer.set_mean_std(
                mean.float(),
                std.float(),
            )

        def logging(self, *args, **kwargs):
            # output histogram
            for target, index in self.categorical_target_map.items():
                # apply softmax to prediction
                logit = kwargs["prediction"]
                pred_prob = torch.softmax(logit, dim=1)

                self.writer.add_histogram(
                    f"output_prob_{target}",
                    pred_prob[:, index],
                    self.trainer.state.iteration,
                )
                self.writer.add_histogram(
                    f"output_logit_{target}",
                    logit[:, index],
                    self.trainer.state.iteration,
                )

        def init_dataset_handler(self, task: law.Task, device: str = "cpu") -> None:
            group_datasets = {
                "ttbar": [d for d in task.datasets if d.startswith("tt_")],
            }
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

            # get statistics for standardization from training dataset without oversampling
            train_valid_dataset_map = self.train_validation_loader.data_map
            self.dataset_statitics = get_standardization_parameter(self.train_validation_loader.data_map, self.continous_inputs)


        def init_optimizer(self, learning_rate=1e-2, weight_decay=1e-5) -> None:
            self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1, gamma=0.9)

        def forward(self, x, *args, **kwargs):
            from IPython import embed; embed(header="string - 890 in RESNET FORWARD ")
            cat_inp = self._handle_input(
                x,
                self.categorical_inputs,
                dtype=torch.int32,
                empty_fill_val=self.placeholder,
                mask_value=EMPTY_INT,
            )
            self.cemb = self.cemb(cat_inp)
            con_inp = self._handle_input(x, self.continous_inputs)
            # standardize inputs
            con_inp = self.std_layer(con_inp)
            x = self.input_layer(con_inp, cat_inp)
            return self.model(x)
