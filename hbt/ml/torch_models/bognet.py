from __future__ import annotations

__all__ = [
]

from functools import partial

from columnflow.util import MockModule, maybe_import, DotDict, classproperty
from columnflow.types import Callable
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
    import torch
    import torch.utils.tensorboard as tensorboard
    from torch.utils.tensorboard import SummaryWriter
    from ignite.metrics import Loss, ROC_AUC
    import matplotlib.pyplot as plt

    from hbt.ml.torch_models.multi_class import WeightedFeedForwardMultiCls, FeedForwardMultiCls
    from hbt.ml.torch_utils.ignite.metrics import (
        WeightedROC_AUC, WeightedLoss,
        # WeightedBalanced_Acc,
    )
    from hbt.ml.torch_utils.ignite.mixins import IgniteTrainingMixin
    from hbt.ml.torch_utils.transforms import AkToTensor, PreProcessFloatValues, MoveToDevice
    from hbt.ml.torch_utils.datasets.handlers import (
        WeightedTensorParquetFileHandler,
    )
    from hbt.ml.torch_utils.utils import (
        embedding_expected_inputs, expand_columns, get_standardization_parameter,
    )
    from hbt.ml.torch_utils.layers import (
        InputLayer, StandardizeLayer, ResNetBlock, ResNetPreactivationBlock, DenseBlock,
        PaddingLayer, RotatePhiLayer,
    )
    from hbt.ml.torch_utils.functions import generate_weighted_loss

    import sklearn

    class BogNet(WeightedFeedForwardMultiCls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # inputs

            # categories in fixed order alphabetical
            # self.categorical_inputs = sorted({
            #     "pair_type",
            #     "decay_mode1",
            #     "decay_mode2",
            #     "lepton1.charge",
            #     "lepton2.charge",
            #     "has_fatjet",
            #     "has_jet_pair",
            #     "year_flag",
            # })

            # # continuous
            # self.continuous_inputs = expand_columns(
            #     "lepton1.{px,py,pz,energy,mass}",
            #     "lepton2.{px,py,pz,energy,mass}",
            #     "bjet1.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
            #     "bjet2.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
            #     "fatjet.{px,py,pz,energy,mass}",
            # )

            self.inputs = set(self.categorical_inputs) | set(self.continuous_inputs)

            # targets
            self.categorical_target_map = {
                "hh": 0,
                "tt": 1,
                "dy": 2,
            }

            # build network, get commandline arguments
            self.nodes = kwargs.get("nodes", 256)
            self.activation_functions = kwargs.get("activation_functions", "LeakyReLu")
            self.skip_connection_init = kwargs.get("skip_connection_init", 1)
            self.freeze_skip_connection = kwargs.get("freeze_skip_connection", False)
            self.empty_value = 15

            self.std_layer, self.input_layer, self.model = self._build_network()

            # loss function and metrics
            self._loss_fn = generate_weighted_loss(torch.nn.CrossEntropyLoss)()

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

            # self.validation_metrics["balanced_accuracy"] = WeightedBalanced_Acc(
            #     output_transform=torch.nn.functional.softmax,
            # )

            # trainings settings
            self.training_epoch_length_cutoff = 2000
            self.training_weight_cutoff = 0.05

            # remove layers that comes due to inheritance
            # TODO clean up, this is only a monkey patch
            if hasattr(self, "linear_relu_stack"):
                del self.linear_relu_stack
            # del self.norm_layer

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

        def init_layers(self):
            return None

        def state_dict(self, *args, **kwargs):
            return self.model.state_dict(*args, **kwargs)

        def load_state_dict(self, *args, **kwargs):
            return self.model.load_state_dict(*args, **kwargs)

        def _build_network(self):
            # helper where all layers are defined
            # std layers are filled when statitics are known
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
            return std_layer, input_layer, model

        def to(self, *args, **kwargs):
            # helper to move all customlayers to given device
            self.std_layer = self.std_layer.to(*args, **kwargs)
            self.input_layer = self.input_layer.to(*args, **kwargs)
            self.model = self.model.to(*args, **kwargs)
            return super().to(*args, **kwargs)

        def train_step(self, engine, batch):
            # from IPython import embed; embed(header="string - 149 in bognet.py ")
            self.train()

            # Compute prediction and loss
            (categorical_x, continous_x), y = batch
            self.optimizer.zero_grad()

            # replace missing values with empty_fill, convert to expected type
            categorical_x = self._handle_input(
                categorical_x,
                self.categorical_inputs,
                dtype=torch.int32,
                empty_fill_val=self.empty_value,
                mask_value=EMPTY_INT,
            )

            continous_x = self._handle_input(
                continous_x,
                self.continuous_inputs,
                dtype=torch.float32,
                empty_fill_val=-10,
                mask_value=EMPTY_FLOAT,
                norm_layer=self.std_layer,
            )

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

            # plotting routines doing every % iterations
            iteration_condition = engine.state.iteration % 400 == 0
            if iteration_condition:

                detach_logit = pred.detach().cpu().numpy()
                detach_target = target.detach().cpu().numpy()
                detach_loss = loss.detach().cpu().numpy()
                prob = torch.nn.functional.softmax(pred).detach().cpu().numpy()

                plot_pred_fig, _ = self.plot_prediction(
                    y_true=detach_target,
                    y_pred=prob,
                    target_map=self.categorical_target_map,
                    savepath=f"1D_pred_target_{engine.state.iteration}.png",
                )

                self.writer.add_figure(
                    "Prediction Probabilities",
                    plot_pred_fig,
                    engine.state.iteration,
                )

                plot_logit_fig, _ = self.plot_prediction(
                    y_true=detach_target,
                    y_pred=detach_logit,
                    target_map=self.categorical_target_map,
                    savepath=f"1D_pred_logit_{engine.state.iteration}.png",
                )

                self.writer.add_figure(
                    "Prediction Logits",
                    plot_logit_fig,
                    engine.state.iteration,
                )

                # self.plot_1D(
                #     (prob, detach_target),
                #     annotation={
                #         "loss": detach_loss,
                #         "pred": target,
                #         "target": detach_target,
                #         "iteration": engine.state.iteration,
                #     },
                #     bins=20,
                #     label=["pred", "target"],
                #     density=True,
                #     alpha=0.3,
                #     color=["blue", "red"],
                #     histtype="stepfilled",
                #     savepath=f"1D_pred_target_{engine.state.iteration}.png",
                # )

                # confusion matrix
                cm_fig, _, c_matrix = self.plot_confusion_matrix(
                    np.argmax(detach_target, axis=1),
                    np.argmax(prob, axis=1),
                    list(self.categorical_target_map.values()),
                    cmap="Blues",
                    savepath=f"confusion_matrix_{engine.state.iteration}.png",
                    sample_weight=None, # if you want to weight the confusion matrix by the target
                )

                self.writer.add_figure(
                    "Confusion Matrix",
                    cm_fig,
                    engine.state.iteration,
                )


                # balanced accuracy
                # from IPython import embed; embed(header="string - 296 in bognet.py ")





                # @staticmethod
                # def plot_batch(self, data, input, target, target_map={0:"signal"}):

                #         # no target_map assumes binary
                #     for idx_target, target in target_map.items():

                #         for idx, name in enumerate(input):
                #             plt.hist()


            return loss.item()

        @staticmethod
        def plot_prediction(y_true, y_pred, target_map, **kwargs):
            # swap axes to get (num_samples, cls) shape
            fig, ax = plt.subplots()
            save_path = kwargs.pop("savepath", None)
            ax.set_xlabel(kwargs.pop("xlabel", "score"))
            ax.set_ylabel(kwargs.pop("ylabel", "frequency"))
            fig.suptitle(kwargs.pop("title", None))

            # get events that are predicted correctly for each class
            y_pred_cls, labels = [], []
            for _cls, idx in target_map.items():
                # get events of cls
                correct_cls_mask = y_true[:, idx] == 1
                # filter predictions for cls
                y_pred_cls.append(y_pred[correct_cls_mask][:,0])
                labels.append(f"{_cls}")

            hist = ax.hist(
                y_pred_cls,
                bins=kwargs.get("bins", 20),
                histtype=kwargs.get("histtype", "step"),
                alpha=kwargs.get("alpha", 0.7),
                label=labels,
                **kwargs,
            )

            fig.legend(title="predicted class")

            if save_path:
                fig.savefig(save_path)

            return fig, ax


        @staticmethod
        def plot_confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, cmap="Blues", savepath=None):
            confusion_matrix = sklearn.metrics.confusion_matrix(
                y_true,
                y_pred,
                labels=labels,
                sample_weight=sample_weight,
                normalize="all",  # normalize to get probabilities
            )

            disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
            disp.plot(cmap=cmap)

            if savepath:
                disp.figure_.savefig(savepath)
            return disp.figure_, disp.ax_, confusion_matrix



        def plot_batch(self, input, target, loss, iteration, target_map=None, labels=None):
            input_per_feature = input.to("cpu").transpose(0,1).detach().numpy()
            input, target = input.to("cpu").detach().numpy(), target.to("cpu").detach().numpy()



            fig, ax = plt.subplots(1,len(self.categorical_inputs), figsize=(8 * len(self.categorical_inputs), 8))
            fig.tight_layout()

            for ind, cat in enumerate(self.categorical_inputs):
                signal_target = target[:, self.categorical_target_map["hh"]]
                background_mask, signal_mask = signal_target.flatten() == 0, signal_target.flatten() == 1
                # background_prediction = detach_pred[zero_s_mask]
                # signal_prediction = detach_pred[zero_s_mask]
                _input = input_per_feature[ind]
                background_input = _input[background_mask]
                signal_input = _input[signal_mask]
                cax = ax[ind]

                cax.hist(
                    [background_input, signal_input],
                    bins=10,
                    histtype="barstacked",
                    alpha=0.5,
                    label=["tt & dy","hh"],
                    density=True,
                )
                cax.set_xlabel(cat)
                cax.annotate(f"loss: {loss:.2f}", (0.8, 0.60), xycoords="figure fraction")
                cax.annotate(f"iteration {iteration}", (0.75, 0.55))
                cax.legend()
            fig.savefig(f"1d_cats_{iteration}.png")


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

                ax.annotate(f"num: 0s: {num_0s:.2f}", (0.5, 0.70), xycoords="axes fraction")
                ax.annotate(f"num: 1s: {num_1s:.2f}", (0.5, 0.65), xycoords="axes fraction")
                ax.annotate(f"loss: {loss:.2f}", (0.5, 0.60), xycoords="axes fraction")
                iteration = annotations.get("iteration")
                ax.annotate(f"iteration {iteration}", (0.5, 0.55))

                tp = np.sum((target == 1) & (pred > 0.5))
                tn = np.sum((target == 0) & (pred < 0.5))
                fp = np.sum((target == 0) & (pred > 0.5))
                fn = np.sum((target == 1) & (pred < 0.5))
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                sensitivity = tp / (tp + fn)

                ax.annotate(f"accuracy: {accuracy:.2f}", (0.5, 0.50), xycoords="axes fraction")
                ax.annotate(f"sensitivity: {sensitivity:.2f}", (0.5, 0.45), xycoords="axes fraction")

            return fig, ax


        def produce_shap(self, cat, cont, prediction):
            # from IPython import embed; embed(header="string - 397 in resnet.py ")
            self.eval()
            input_tensor = torch.cat([cat , cont], dim=1)

            samples = 3000
            background, test = input_tensor[:samples], input_tensor[samples: samples + 100]
            trans_model = ShapModel(self)
            explainer = shap.DeepExplainer(trans_model, background)
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
                    empty_fill_val=self.empty_value,
                    mask_value=EMPTY_INT,
                )

                continous_x = self._handle_input(
                    continous_x,
                    self.continuous_inputs,
                    dtype=torch.float32,
                    empty_fill_val=-10,
                    mask_value=EMPTY_FLOAT,
                    norm_layer=self.std_layer,
                )

                # from IPython import embed; embed(header="string - 397 in resnet.py ")
                pred = self(categorical_x, continous_x)

                if y.dim() == 1:
                    y = y.reshape(-1, 1)
                y = y.to(torch.float32)

                from IPython import embed; embed(header="string - 490 in bognet.py ")
                iteration_condition = engine.state.iteration % 400 == 0
                if iteration_condition:

                    detach_logit = pred.detach().cpu().numpy()
                    detach_target = y.detach().cpu().numpy()
                    # detach_loss = loss.detach().cpu().numpy()
                    prob = torch.nn.functional.softmax(pred).detach().cpu().numpy()

                    plot_pred_fig, _ = self.plot_prediction(
                        y_true=detach_target,
                        y_pred=prob,
                        target_map=self.categorical_target_map,
                        savepath=f"1D_pred_target_{engine.state.iteration}.png",
                    )

                    self.writer.add_figure(
                        "Prediction Probabilities Validation",
                        plot_pred_fig,
                        engine.state.iteration,
                    )

                    plot_logit_fig, _ = self.plot_prediction(
                        y_true=detach_target,
                        y_pred=detach_logit,
                        target_map=self.categorical_target_map,
                        savepath=f"1D_pred_logit_{engine.state.iteration}.png",
                    )

                    self.writer.add_figure(
                        "Prediction Logits",
                        plot_logit_fig,
                        engine.state.iteration,
                    )

                    # confusion matrix
                    cm_fig, _, c_matrix = self.plot_confusion_matrix(
                        np.argmax(detach_target, axis=1),
                        np.argmax(prob, axis=1),
                        list(self.categorical_target_map.values()),
                        cmap="Blues",
                        savepath=f"confusion_matrix_{engine.state.iteration}.png",
                        sample_weight=None, # if you want to weight the confusion matrix by the target
                    )

                    self.writer.add_figure(
                        "Confusion Matrix",
                        cm_fig,
                        engine.state.iteration,
                    )

                return pred, y, {"weight": weights.reshape(-1, 1)}

        def setup_preprocessing(self):
            # extract dataset std and mean from dataset
            # extraction happens form no oversampled dataset
            mean, std = [], []
            for _input in self.continuous_inputs:
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
                # categorical_target_transformation=,
                # data_type_transformation=AkToTensor,
            )

            self.training_loader, (self.train_validation_loader, self.validation_loader) = self.dataset_handler.init_datasets() #noqa

            # define lenght of training epoch
            self.max_epoch_length = self._calculate_max_epoch_length(
                self.training_loader,
                cutoff=self.training_epoch_length_cutoff,
                weight_cutoff=self.training_weight_cutoff,
            )

            # get statistics for standardization from training dataset without oversampling
            self.dataset_statitics = get_standardization_parameter(self.train_validation_loader.data_map, self.continuous_inputs)

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
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1, gamma=0.9)

        def log_graph(self, engine):
            (categorical_x, continous_x) = engine.state.batch[0]

            categorical_x = self._handle_input(
                categorical_x,
                self.categorical_inputs,
                dtype=torch.int32,
                empty_fill_val=self.empty_value,
                mask_value=EMPTY_INT,
            )

            continous_x = self._handle_input(
                continous_x,
                self.continuous_inputs,
                dtype=torch.float32,
                empty_fill_val=-10,
                mask_value=EMPTY_FLOAT,
                norm_layer=self.std_layer,
            )

            self.writer.add_graph(self, (categorical_x, continous_x))

        def forward(self, *inputs):
            return self.model(inputs)

    class UpdatedBogNet(BogNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.custom_hooks.append("perform_scheduler_step")

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

        def _build_network(self):
            # helper where all layers are defined
            # std layers are filled when statitics are known
            std_layer = StandardizeLayer(
                None,
                None,
            )

            continuous_padding = PaddingLayer(padding_value=0, mask_value=EMPTY_FLOAT)
            categorical_padding = PaddingLayer(padding_value=self.empty_value, mask_value=EMPTY_INT)
            rotation_layer = RotatePhiLayer(columns=self.continuous_inputs)

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
            return std_layer, input_layer, model

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
            **kwargs,
        ) -> None:
            all_datasets = datasets or getattr(task, "resolved_datasets", task.datasets)
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

        def setup_preprocessing(self):
            # extract dataset std and mean from dataset
            # extraction happens form no oversampled dataset
            mean, std = [], []
            for _input in self.continuous_inputs:
                input_statistics = self.dataset_statistics[_input.column]
                mean.append(torch.from_numpy(input_statistics["mean"]))
                std.append(torch.from_numpy(input_statistics["std"]))

            mean, std = torch.concat(mean), torch.concat(std)
            # set up standardization layer
            self.std_layer.set_mean_std(
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
            cont, cat = torch.as_tensor(x[:, self.num_cat:], dtype=torch.float32), torch.as_tensor(x[:, :self.num_cat], dtype=torch.int32)
            return self.model((cat, cont))
