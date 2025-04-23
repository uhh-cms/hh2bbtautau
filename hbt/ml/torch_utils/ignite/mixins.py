from __future__ import annotations
import abc

from functools import partial
from columnflow.util import MockModule, maybe_import
from columnflow.types import Callable
from hbt.ml.torch_utils.datasets.handlers import DatasetHandlerMixin
from hbt.ml.torch_utils.utils import CustomEarlyStopping as EarlyStopping


ignite = maybe_import("ignite")

if not isinstance(ignite, MockModule):
    from ignite.engine import Engine, Events
    from ignite.metrics import Metric
    from torch.utils.tensorboard import SummaryWriter
    import torchdata.nodes as tn

    class IgniteMixinBase:
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.custom_hooks = []
            self.trainer = None
            self.train_evaluator = None
            self.val_evaluator = None
            self.validation_metrics = {}
            self.writer = None
            self.run_name = None

    class IgniteTrainingMixin(DatasetHandlerMixin, IgniteMixinBase):
        trainer: Engine
        train_evaluator: Engine
        val_evaluator: Engine
        validation_metrics: dict[str, Metric]
        writer: SummaryWriter | None
        run_name: str
        logger: object
        custom_hooks: list[str]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trainer = None
            self._loss_fn = None
            self.max_epoch_length = None
            self.max_val_epoch_length = None

        @abc.abstractmethod
        def train_step(self, engine: Engine, batch: tuple) -> tuple:
            """Override this method to define the training step."""
            pass

        @abc.abstractmethod
        def validation_step(self, engine: Engine, batch: tuple) -> tuple:
            """Override this method to define the validation step."""
            pass

        @property
        def loss_fn(self) -> Callable:
            """Override this method to define the loss function."""
            if not self._loss_fn:
                raise NotImplementedError("Please implement the loss function.")
            return self._loss_fn

        def create_engines(self) -> None:
            self.trainer = Engine(self.train_step)
            self.train_evaluator = Engine(self.validation_step)
            self.val_evaluator = Engine(self.validation_step)

            self.init_metrics()

            self.trainer.add_event_handler(
                Events.ITERATION_COMPLETED(every=100),
                self.log_training_loss,
            )
            self.trainer.add_event_handler(
                Events.EPOCH_STARTED,
                self.reset_dataloaders,
            )
            self.trainer.add_event_handler(
                Events.EPOCH_COMPLETED,
                partial(
                    self.log_results,
                    evaluator=self.train_evaluator,
                    data_loader=self.training_loader.data_loader,
                    max_epoch_length=self.max_epoch_length,
                    mode="training",
                ),
            )
            self.trainer.add_event_handler(
                Events.EPOCH_COMPLETED,
                partial(
                    self.log_results,
                    evaluator=self.val_evaluator,
                    data_loader=self.validation_loader.data_loader,
                    max_epoch_length=self.max_val_epoch_length,
                    mode="validation",
                ),
            )

            for custom_hook in self.custom_hooks:
                fn = getattr(self, custom_hook, None)
                if not fn or not callable(fn):
                    self.logger.warning(f"Could not find custom hook '{custom_hook}', skipping")
                else:
                    fn()

        def init_metrics(self) -> None:

            # Attach metrics to the evaluators
            for name, metric in self.validation_metrics.items():
                metric.attach(self.train_evaluator, name)

            for name, metric in self.validation_metrics.items():
                metric.attach(self.val_evaluator, name)

        def log_training_loss(self, engine):
            if self.writer:
                self.writer.add_scalars(
                    f"{self.run_name}_per_batch_training",
                    {"loss": engine.state.output},
                    engine.state.iteration,
                )
            self.logger.info(
                f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}",
            )

        def reset_dataloaders(self, trainer):
            self.training_loader.data_loader.reset()

        def log_results(
            self,
            trainer: Engine,
            evaluator: Engine,
            data_loader: tn.ParallelMapper,
            max_epoch_length: int | None = None,
            mode: str = "training",
        ):
            data_loader.reset()
            evaluator.run(data_loader, epoch_length=max_epoch_length)
            metrics = evaluator.state.metrics
            infos = " | ".join([f"Avg {name}: {value:.2f}" for name, value in metrics.items()])
            for name, value in metrics.items():
                if self.writer:
                    self.writer.add_scalars(
                        f"{self.run_name}_{name}",
                        {mode: value},
                        trainer.state.epoch,
                    )
            self.logger.info(f"Results ({mode}) - Epoch[{trainer.state.epoch}] {infos}")

        def start_training(self, run_name: str, max_epochs: int):
            if not self.trainer:
                self.create_engines()
            self.trainer.run(self.training_loader.data_loader, max_epochs=max_epochs, epoch_length=self.max_epoch_length)
            if self.writer:
                self.writer.close()

    class IgniteEarlyStoppingMixin(IgniteMixinBase):
        early_stopping_patience: int
        early_stopping_min_epochs: int
        early_stopping_min_diff: float
        trainer: Engine
        val_evaluator: Engine
        custom_hooks: list[str]

        def __init__(
            self,
            *args,
            early_stopping_patience: int = 10,
            early_stopping_min_epochs: int = 1,
            early_stopping_min_diff: float = 0.0,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.custom_hooks.append("create_early_stopping")
            self.early_stopping_patience = early_stopping_patience
            self.early_stopping_min_epochs = early_stopping_min_epochs
            self.early_stopping_min_diff = early_stopping_min_diff

        def create_early_stopping(self):
            def score_function(engine):
                val_loss = engine.state.metrics["loss"]
                return -val_loss

            handler = EarlyStopping(
                patience=self.early_stopping_patience,
                min_epochs=self.early_stopping_min_epochs,
                model=self,
                score_function=score_function,
                trainer=self.trainer,
                min_delta=self.early_stopping_min_diff,
            )
            # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
            self.val_evaluator.add_event_handler(Events.COMPLETED, handler)
