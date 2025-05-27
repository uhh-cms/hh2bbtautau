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
    WeightedCrossEntropySlice, generate_weighted_loss,
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
    from hbt.ml.torch_utils.functions import generate_weighted_loss
    from hbt.ml.torch_utils.ignite.mixins import IgniteTrainingMixin, IgniteEarlyStoppingMixin
    from hbt.ml.torch_utils.layers import PaddingLayer, InputLayer, StandardizeLayer, ResNetBlock
    import matplotlib.pyplot as plt
    import shap

    class WeightedResNet(WeightedFeedForwardMultiCls):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            self.categorical_features = sorted({
                "pair_type",
                "decay_mode1",
                "decay_mode2",
                "lepton1.charge",
                "lepton2.charge",
                "has_fatjet",
                "has_jet_pair",
                "year_flag",
            })
            self.embedding_dims = 50
            super().__init__(*args, **kwargs)
            self.continuous_features = sorted(self.inputs)
            n_floating_inputs = len(self.inputs)
            self.categorical_features = sorted({
                "pair_type",
                "decay_mode1",
                "decay_mode2",
                "lepton1.charge",
                "lepton2.charge",
                "has_fatjet",
                "has_jet_pair",
                "year_flag",
            })

            self.init_layers()
            self._loss_fn = generate_weighted_loss(nn.CrossEntropyLoss)()

            self.validation_metrics = dict()
            # update list of inputs
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
                nn.BatchNorm1d(512),
                nn.Linear(512, 64),
                nn.PReLU(),
                nn.BatchNorm1d(64),
                nn.Linear(64, 16),
                nn.PReLU(),
                nn.BatchNorm1d(16),
                nn.Linear(16, 64),
                nn.PReLU(),
                nn.BatchNorm1d(64),
                nn.Linear(64, 512),
                nn.PReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.PReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.PReLU(),
                nn.Linear(512, 3),
            )

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

        def to(self, *args, **kwargs):
            self.std_layer = self.std_layer.to(*args, **kwargs)
            self.input_layer = self.input_layer.to(*args, **kwargs)
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
