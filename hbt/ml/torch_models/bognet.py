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
                # "dy": 2,
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

            self._loss_fn = nn.BCELoss()
            # self._loss_fn = torch.nn.functional.binary_cross_entropy

            # layer layout
            self.training_epoch_length_cutoff = 2000
            self.training_weight_cutoff = 0.05
            self.placeholder = 15
            self.std_layer, self.input_layer, self.model = self._build_network()
            # del linear_relu_stack
            # loss,
            # self._loss_fn = nn.CrossEntropyLoss()
            self.validation_metrics = {
                # "unweighted_loss": Loss(self.loss_fn),
                "loss": WeightedLoss(self.loss_fn),
            }

        def _build_network(self):
            std_layer = StandardizeLayer(
                None,
                None,
            )

            input_layer = InputLayer(
                self.continous_inputs,
                self.categorical_inputs,
                embedding_dim=2,
                expected_categorical_inputs=embedding_expected_inputs,
                empty_value=self.placeholder,
            )

            model = nn.Sequential(
                input_layer,
                torch.nn.Linear(input_layer.ndim, self.nodes),
                torch.nn.BatchNorm1d(self.nodes),
                torch.nn.LeakyReLU(),
                ResNetBlock(self.nodes, self.activation_functions, self.skip_connection_init, self.freeze_skip_connection), # noqa
                ResNetBlock(self.nodes, self.activation_functions, self.skip_connection_init, self.freeze_skip_connection), # noqa
                ResNetBlock(self.nodes, self.activation_functions, self.skip_connection_init, self.freeze_skip_connection), # noqa
                # torch.nn.Linear(self.nodes, len(self.categorical_target_map)),

                torch.nn.Linear(self.nodes, 1),
                torch.nn.Sigmoid(),
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
            (categorical_x, continous_x), y = batch
            self.optimizer.zero_grad()

            categorical_x = self._handle_input(
                categorical_x,
                self.categorical_inputs,
                dtype=torch.int32,
                empty_fill_val=self.placeholder,
                mask_value=EMPTY_INT,
            )

            continous_x = self._handle_input(
                continous_x,
                self.continous_inputs,
                dtype=torch.float32,
                empty_fill_val=-10,
                mask_value=EMPTY_FLOAT,
                norm_layer=self.std_layer,
            )

            # from IPython import embed; embed(header="string - 397 in resnet.py ")
            # from IPython import embed; embed(header="string - 397 in resnet.py ")
            self.input_layer.embedding_layer.normalize_embeddings()

            pred = self(categorical_x, continous_x)
            target = y.to(torch.float32)
            if target.dim() == 1:
                target = target.reshape(-1, 1)

            loss = self.loss_fn(pred, target)
            # Backpropagation
            loss.backward()
            self.optimizer.step()

            if engine.state.iteration % 100 == 0:
                # from IPython import embed; embed(header="training_step - 414 in resnet.py ")
                # for ind, cat in enumerate(self.categorical_inputs):
                #     plt.clf()
                #     data = []
                #     labels = []
                #     # for dataset, arrays in d.items():
                #     data.append(arrays[ind])
                #     plt.hist(data, histtype="barstacked", alpha=0.5, label=labels)
                #     plt.xlabel(cat)
                #     plt.legend()
                #     plt.savefig(f"{cat}_all.png")



            #     # self.produce_shap(cat=categorical_x, cont=continous_x, prediction=pred.detach().cpu().numpy().flatten())
                detach_pred = pred.detach().cpu().numpy().flatten()
                detach_target = target.detach().cpu().numpy().flatten()
                detach_loss = loss.detach().cpu().numpy().flatten()
                self.plot_2D(
                    x = detach_target,
                    y = detach_pred,
                    bins=10,
                    xlabel="pred",
                    ylabel="target",
                    title=f"iteration {engine.state.iteration}",
                    savepath=f"2D_pred_target_{engine.state.iteration}.png",
                )

                self.plot_1D(
                    (detach_pred, detach_target),
                    annotation={
                        "loss": detach_loss,
                        "pred": detach_pred,
                        "target": detach_target,
                        "iteration": engine.state.iteration,
                    },
                    bins=20,
                    label=["pred", "target"],
                    density=True,
                    alpha=0.3,
                    color=["blue", "red"],
                    histtype="stepfilled",
                    savepath=f"1D_pred_target_{engine.state.iteration}.png",
                )

        # @staticmethod
        # def plot_batch(self, data, input, target, target_map={0:"signal"}):

        #         # no target_map assumes binary
        #     for idx_target, target in target_map.items():

        #         for idx, name in enumerate(input):
        #             plt.hist()


            return loss.item()

        @staticmethod
        def plot_2D(x,y, bins=10, **kwargs):
            fig, ax = plt.subplots()
            hist = ax.hist2d(
                x,
                y,
                bins=bins,
                cmap=kwargs.get("cmap"),
            )
            ax.set_xlabel(kwargs.get("xlabel"))
            ax.set_xlabel(kwargs.get("ylabel"))
            fig.colorbar(hist[3], ax=ax)
            ax.set_title(kwargs.get("title"))
            if kwargs.get("savepath"):
                fig.savefig(kwargs.get("savepath"))
            return fig, ax

        @staticmethod
        def plot_1D(x, annotations=None, **kwargs):
            fig, ax = plt.subplots()
            hist = ax.hist(
                x,
                bins=kwargs.get("bins"),
                histtype="stepfilled",
                alpha=0.7,
                color=kwargs.get("color"),
                label=kwargs.get("label"),
                density=kwargs.get("density",
                )
            )
            ax.set_xlabel(kwargs.get("xlabel"))
            ax.set_xlabel(kwargs.get("ylabel"))
            ax.set_title(kwargs.get("title"))
            if kwargs.get("savepath"):
                fig.savefig(kwargs.get("savepath"))
            ax.legend()

            if annotations:
                loss = annotations.get("loss")
                target = annotations.get("target")
                pred = annotations.get("pred")
                num_0s = np.sum(target == 0)
                num_1s = np.sum(target == 1)

                ax.annotate(f"num: 0s: {num_0s:.2f}", (0.5, 0.70), xycoords="figure fraction")
                ax.annotate(f"num: 1s: {num_1s:.2f}", (0.5, 0.65), xycoords="figure fraction")
                ax.annotate(f"loss: {loss:.2f}", (0.5, 0.60), xycoords="figure fraction")
                ax.annotate(f"loss: {loss:.2f}", (0.5, 0.60), xycoords="figure fraction")
                iteration = annotations.get("iteration")
                ax.annotate(f"iteration {iteration}", (0.5, 0.55))

                tp = np.sum((target == 1) & (pred > 0.5))
                tn = np.sum((target == 0) & (pred < 0.5))
                fp = np.sum((target == 0) & (pred > 0.5))
                fn = np.sum((target == 1) & (pred < 0.5))
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                sensitivity = tp / (tp + fn)

                ax.annotate(f"accuracy: {accuracy:.2f}", (0.5, 0.50), xycoords="figure fraction")
                ax.annotate(f"sensitivity: {sensitivity:.2f}", (0.5, 0.45), xycoords="figure fraction")

            return fig, ax


        def produce_shap(self, cat, cont, prediction):
            # from IPython import embed; embed(header="string - 397 in resnet.py ")
            self.eval()
            input_tensor = torch.cat([cat , cont], dim=1)

            samples = 3000
            background, test = input_tensor[:samples], input_tensor[samples: samples + 100]
            trans_model = ShapModel(self)
            explainer = shap.DeepExplainer(trans_model, background)
            from IPython import embed; embed(header="string - 483 in resnet.py ")
            shap_values = explainer.shap_values(test)

        def validation_step(self, engine, batch):
            self.eval()
            # Set the model to evaluation mode - important for batch normalization and dropout layers

            # if engine.state.iteration > self.max_val_epoch_length * (engine.state.epoch + 1):
            #     engine.terminate_epoch()

            # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
            # also serves to reduce unnecessary gradient computations and memory usage for tensors with
            # requires_grad=True
            with torch.no_grad():
                # from IPython import embed; embed(header="validation - 533 in resnet.py ")
                (categorical_x, continous_x, weights), y = batch

                categorical_x = self._handle_input(
                    categorical_x,
                    self.categorical_inputs,
                    dtype=torch.int32,
                    empty_fill_val=self.placeholder,
                    mask_value=EMPTY_INT,
                )

                continous_x = self._handle_input(
                    continous_x,
                    self.continous_inputs,
                    dtype=torch.float32,
                    empty_fill_val=-10,
                    mask_value=EMPTY_FLOAT,
                    norm_layer=self.std_layer,
                )

                # from IPython import embed; embed(header="string - 397 in resnet.py ")
                pred = self(categorical_x, continous_x)

                if y.dim() == 1:
                    y = y.reshape(-1, 1)
                return pred, y, {"weight": weights.reshape(-1, 1)}


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
            from IPython import embed; embed(header="logging - 451 in resnet.py ")
            for target, index in self.categorical_target_map.items():
                # apply softmax to prediction
                logit = kwargs["prediction"]

                # pred_prob = torch.softmax(logit, dim=1)

                # self.writer.add_histogram(
                #     f"output_prob_{target}",
                #     pred_prob[:, index],
                #     self.trainer.state.iteration,
                # )
                self.writer.add_histogram(
                    f"output_logit_{target}",
                    logit[:, index],
                    self.trainer.state.iteration,
                )

        def init_dataset_handler(self, task: law.Task, device: str = "cpu") -> None:
            all_datasets = getattr(task, "resolved_datasets", task.datasets)
            group_datasets = {
                "ttbar": [d for d in all_datasets if d.startswith("tt_")],
                # "dy": [d for d in all_datasets if d.startswith("dy_")],y
            }

            # group_datasets = {
            #     "ttbar": [d for d in task.datasets if d.startswith("tt_")],
            # }


            self.dataset_handler = WeightedRgTensorParquetFileHandler(
                task=task,
                continuous_features=getattr(self, "continuous_features", self.continous_inputs),
                categorical_features=getattr(self, "categorical_features", self.categorical_inputs),
                batch_transformations=MoveToDevice(device=device),
                # global_transformations=PreProcessFloatValues(),
                build_categorical_target_fn=self._build_categorical_target,
                group_datasets=group_datasets,
                device=device,
                # categorical_target_transformation=partial(get_one_hot, nb_classes=2),
                datasets=[d for d in all_datasets if any(d.startswith(x) for x in ["tt_", "hh_"])],
                # categorical_target_transformation=,
                # data_type_transformation=AkToTensor,
            )
            self.training_loader, (self.train_validation_loader, self.validation_loader) = self.dataset_handler.init_datasets() #noqa

            # get statistics for standardization from training dataset without oversampling
            self.dataset_statitics = get_standardization_parameter(self.train_validation_loader.data_map, self.continous_inputs)

            # delete
            # all_data = ak.concatenate(list(map(lambda x: x.data, self.train_validation_loader.data_map)))

            # from IPython import embed; embed(header="dataset handling - 496 in resnet.py ")


        def control_plot_1d(self):
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
            self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1, gamma=0.9)

        def log_graph(self, engine):
            (categorical_x, continous_x) = engine.state.batch[0]

            categorical_x = self._handle_input(
                categorical_x,
                self.categorical_inputs,
                dtype=torch.int32,
                empty_fill_val=self.placeholder,
                mask_value=EMPTY_INT,
            )

            continous_x = self._handle_input(
                continous_x,
                self.continous_inputs,
                dtype=torch.float32,
                empty_fill_val=-10,
                mask_value=EMPTY_FLOAT,
                norm_layer=self.std_layer,
            )

            self.writer.add_graph(self, (categorical_x, continous_x))


        def forward(self, *inputs):
            return self.model(inputs)

    class ShapModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model
            self.num_cont = len(model.continous_inputs)
            self.num_cat = len(model.categorical_inputs)

        def forward(self, x):
            cont, cat = torch.as_tensor(x[:, self.num_cat:], dtype=torch.float32), torch.as_tensor(x[:, :self.num_cat], dtype=torch.int32)
            return self.model((cat, cont))
