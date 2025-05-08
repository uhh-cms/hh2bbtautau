from columnflow.util import MockModule, maybe_import
from columnflow.types import Callable
from collections import defaultdict
from hbt.ml.torch_utils.datasets.handlers import DatasetHandlerMixin

tqdm = maybe_import("tqdm")
torch = maybe_import("torch")

class PlainBaseMixin:
    """
    A mixin class for plain base models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_hooks = getattr(self, "custom_hooks", list())
        self.validation_metrics = getattr(self, "validation_metrics", None)
        self.run_name = getattr(self, "run_name", None)
        self.writer = getattr(self, "writer", None)

class PlainTrainingMixin(DatasetHandlerMixin, PlainBaseMixin):
    """
    A mixin class for plain training models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_fn = None
        self.max_epoch_length = None
        self.max_val_epoch_length = None
        self.training_logger_interval = 20
        self.training_printout_interval = 100
        self.global_step = 0

    def train_step(self, batch: tuple) -> tuple:
        """Override this method to define the training step."""
        pass

    def validation_step(self, batch: tuple) -> tuple:
        """Override this method to define the validation step."""
        pass

    @property
    def loss_fn(self) -> Callable:
        """Override this method to define the loss function."""
        if not self._loss_fn:
            raise NotImplementedError("Please implement the loss function.")
        return self._loss_fn

    def run_training_loop(self, max_epoch_lenght: int):
        training_bar = tqdm(self.training_loader.data_loader, total=max_epoch_lenght, desc="Training")

        for ibatch, batch in enumerate(training_bar):
            if ibatch > max_epoch_lenght:
                break
            self.global_step += 1
            # Perform the training step
            pred, target = self.train_step(batch)
            loss = self.loss_fn(pred, target)

            if int(ibatch) % int(self.training_printout_interval) == 0:
                loss = loss.item()
                self.log_training_loss(loss, self.global_step)
                update = f"loss: {loss:>7f} "
                training_bar.set_description(update)

    def _log_attributes(self, metrics, epoch, mode="training"):

        for name, value in metrics.items():
            self.writer.add_scalars(
                f"{self.run_name}_{name}",
                {mode: value},
                epoch,
            )
        # save model weights as histogram
        for name, values in self.state_dict().items():
            weights = values
            flattened_weights = weights.flatten()
            tag = f"{name}"
            self.writer.add_histogram(
                tag,
                flattened_weights,
                global_step=epoch,
                bins="tensorflow",
            )

    def log_graph(self, engine):
        self.writer.add_graph(self, engine.state.batch[0])

    def log_training_loss(self, loss, iteration):

        if self.writer:
            self.writer.add_scalars(
                f"{self.run_name}_per_batch_training",
                {"loss": loss},
                iteration,
            )

            # save model weights as histogram
            for name, values in self.state_dict().items():
                weights = values
                flattened_weights = weights.flatten()
                tag = f"{name}_per_batch_training"
                self.writer.add_histogram(
                    tag,
                    flattened_weights,
                    global_step=iteration,
                    bins="tensorflow",
                )

    def run_validation_loop(self, dataloader, max_epoch_length: int, epoch: int = 0, mode: str = "training"):
        validation_bar = tqdm(dataloader, total=max_epoch_length, desc="Validation")

        metric_values = defaultdict(float)
        n_points = 0
        for ibatch, batch in enumerate(validation_bar):
            if ibatch > max_epoch_length:
                break
            # Perform the validation step
            output = self.validation_step(batch)
            kwargs = dict()
            if len(output) == 3:
                pred, target, kwargs = output
            else:
                pred, target = output
            for self.metric_name, metric in self.validation_metrics.items():
                metric_values[self.metric_name] += metric(pred, target, **kwargs)

            if "weights" in kwargs:
                n_points += torch.sum(kwargs["weights"])
            else:
                n_points += pred.shape[0]

        metric_values = {k: v / n_points for k, v in metric_values.items()}
        self._log_attributes(
            metrics=metric_values,
            epoch=epoch,
            mode=mode,
        )

    def start_training(self, run_name: str, max_epochs: int):
        self.run_name = run_name
        for t in max_epochs:
            self.run_training_loop(self.max_epoch_length)

            self.run_validation_loop(
                dataloader=getattr(self, "train_validation_loader", self.training_loader).data_loader,
                max_epoch_length=self.max_val_epoch_length,
                epoch=t,
                mode="training",
            )

            self.run_validation_loop(
                dataloader=self.validation_loader.data_loader,
                max_epoch_length=self.max_val_epoch_length,
                epoch=t,
                mode="validation",
            )

        if self.writer:
            self.writer.close()
