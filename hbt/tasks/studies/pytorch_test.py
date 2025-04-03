from __future__ import annotations
import law.decorator
from collections import Collection
from functools import partial

from columnflow.tasks.union import UniteColumns, UniteColumnsWrapper
from columnflow.util import dev_sandbox, DotDict, maybe_import
from columnflow.columnar_util import Route
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.selection import MergeSelectionStats
from columnflow.types import Callable

from hbt.tasks.base import HBTTask
# from hbt.ml.pytorch_util import ListDataset, MapAndCollate
import law
import luigi

logger = law.logger.get_logger(__name__)

dd = maybe_import("dask.dataframe")
torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
dak = maybe_import("dask_awkward")
ak = maybe_import("awkward")
np = maybe_import("numpy")
tqdm = maybe_import("tqdm")


def run_task(datasets=["hh_ggf_hbb_htt_kl1_kt1_powheg", "tt_sl_powheg"]):
    task = UniteColumnsWrapper(
        **dict((
            ("configs", "run3_2022_postEE_limited"),
            ("version", "pytorch_test"),
            ("datasets", ",".join(datasets)),
        )),
    )

    task.law_run(
        _global=[
            "--workers", 4,
            "--cf.UniteColumns-check-finite-output", "False",
        ],
    )

    sub_targets = {
        d: UniteColumns(
            config = "run3_2022_postEE_limited",
            version = "pytorch_test",
            dataset = d,
        ).target()
        for d in datasets
    }

    from IPython import embed; embed(header="finished running task")
    return task.target()

def main():
    
    output_paths = run_task()
    from IPython import embed; embed(header=f"output_paths: {output_paths}")


class HBTPytorchTask(
    HBTTask,
    UniteColumnsWrapper,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    """
    Base task for trigger related studies.
    """

    sandbox = dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_torch.sh")

    reqs = Requirements(
        RemoteWorkflow.reqs,
        UniteColumnsWrapper=UniteColumnsWrapper,
        UniteColumns=UniteColumns,
        MergeSelectionStats=MergeSelectionStats,
    )

    batch_size = luigi.IntParameter(
        default=1024,
        significant=False,
        description="Batch size to use in training. Default: 1024",
    )

    max_epochs = luigi.IntParameter(
        default=300,
        significant=False,
        description="Maximum training epochs to use. Default: 300",
    )

    load_parallel_cores = luigi.IntParameter(
        default=0,
        significant=False,
        description="Number of sub processes to load for data loading. Default: 0 (only run in main thread)",
    )

    models = law.CSVParameter(
        description="comma-separated names of ml models to train;",
        significant=True,
        brace_expand=True,
    )

    version = law.Parameter(
        description="Version of the task",
        significant=True,
    )

    def create_branch_map(self):
        return dict(enumerate(self.models))

    def workflow_requires(self):
        reqs = super().workflow_requires()

        # require the full merge forest
        reqs["unite_columns"] = self.reqs.UniteColumnsWrapper.req(self)
        # from IPython import embed; embed(header="workflow_requires")
        return reqs

    def requires(self):
        reqs = DotDict()
        reqs["events"] = DotDict.wrap({
            d: DotDict.wrap({
                config:  self.reqs.UniteColumns.req(
                    self,
                    dataset=d,
                    config=config,
                    branch=-1,
                    _exclude=["datasets", "configs", "branch", "branches"])
                for config in self.configs
            })
            for d in self.datasets
        })
        # also require selection stats

        reqs["selection_stats"] = DotDict.wrap({
            d: DotDict.wrap({
                config: self.reqs.MergeSelectionStats.req_different_branching(self, dataset=d, config=config, _exclude=["datasets", "configs"])
                for config in self.configs
            })
            for d in self.datasets
        })

        # from IPython import embed; embed(header="requires")
        return reqs
    
    def output(self):
        return {
            "model": self.target(f"torch_model_{self.branch_data}.pt"),
            "tensorboard": self.local_target(f"tb_logger_{self.branch_data}", dir=True, optional=True),   
        }
    
    @property
    def _parameter_repr(self) -> str:
        additional_params = []
        param_repr = f"bs_{self.batch_size}__max_epochs_{self.max_epochs}"
        if additional_params:
            param_repr += "__{}".format(law.util.create_hash([str(p) for p in additional_params]))
        return param_repr

    def store_parts(self) -> law.util.InsertableDict:
        """
        :return: Dictionary with parts that will be translated into an output directory path.
        """
        parts = super().store_parts()
        dataset_repr = law.util.create_hash([str(d) for d in sorted(self.datasets)])
        config_repr = law.util.create_hash([str(c) for c in sorted(self.configs)])

        parts.insert_after("task_family", "configs", f"configs_{len(self.configs)}__{config_repr}")

        parts.insert_after("task_family", "datasets", f"datasets_{len(self.datasets)}__{dataset_repr}")
        # parts.insert_after("task_family", "model", f"model_{self.branch_data}")
        parts.insert_after("datasets", "parameters", self._parameter_repr)
        return parts
    
    def complete(self):
        from law.util import flatten
        # create a flat list of all outputs
        outputs = flatten(self.output())

        if len(outputs) == 0:
            logger.warning("task {!r} has no outputs or no custom complete() method".format(self))
            return True

        return all(t.complete() for t in outputs)

    @law.decorator.log
    @law.decorator.safe_output
    @law.decorator.localize(input=True, output=True)
    def run(self):
        from hbt.ml.torch_utils.dataloaders import (
            CompositeDataLoader, 
        )
        from hbt.ml.torch_utils.datasets import ParquetDataset, FlatParquetDataset, FlatRowgroupParquetDataset
        from torch.utils.data import default_collate
        import torch.nn.functional as F
        from hbt.ml.torch_utils.map_and_collate import NestedDictMapAndCollate

        from hbt.ml.torch_utils.transforms import (
            AkToTensor, PreProcessFloatValues, PreProssesAndCast,
        )
        from hbt.ml.torch_models import FeedForwardNet

        from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
        from ignite.metrics import Accuracy, Loss, ROC_AUC
        from ignite.handlers import ModelCheckpoint
        from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"Running pytorch on {device}")
        # if device.type == "cuda":                
        #     torch.multiprocessing.set_start_method('spawn')
        open_options = {}

        ### read via dask dataframe ######################################################
        # signal_dfs = dd.read_parquet(
        #     [
        #         t.path
        #         for collections in signal_targets
        #         for targets in collections.targets.values()
        #         for t in targets.values() 
        #     ]
        # )
        # background_dfs = dd.read_parquet(
        #     [
        #         t.path
        #         for collections in backgrounds_targets
        #         for targets in collections.targets.values()
        #         for t in targets.values() 
        #     ]
        # )

        ### read via dask awkward ###################################################
        # signal_daks = dak.from_parquet(
        #     signal_target_paths,
        #     split_row_groups=True,
        # )

        # pro:
        # - can read multiple parquet files w/o loading to memory
        # - can read only the columns we need
        # - option to split between row groups
        # con:
        # - when accessing single elements, there seems to be a lot of overhead/
        #   leaked memory
        # - each compute step takes time


        #### read via awkward array #####################################################
        # signal_daks = ak.from_parquet(
        #     signal_target_paths,
        # )

        # pro:
        # - can read multiple parquet files
        # - can read only the columns we need
        # - fast
        # - can also read individual partitions (not implemented now)
        # con:
        # - eager loading of all data (problem?)

        #### test case
        from hbt.ml.torch_models import model_clss
        model_cls = model_clss.get(self.branch_data, None)
        if not model_cls:
            raise ValueError(f"Unable to load model {self.branch_data}, available list: {model_clss}")
        model = model_cls()
        model = model.to(device)

        model.init_dataset_handler(task=self)

        training_composite_loader, validation_composite_loader = model.init_datasets()        

        logger.info("Constructing loss and optimizer")
        loss_fn = getattr(model, "loss_fn", None)
        if not loss_fn:
            raise ValueError(f"Unable to load attribute 'loss_fn' from model '{self.branch_data}")
        
        optimizer = model.init_optimizer()

        def lr_decay(epoch):
            return 0.9**epoch if 1e-1*0.9**epoch > 1e-9 else 1e-1
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_decay)
        val_metrics = {
            "loss": Loss(loss_fn),
            "roc_auc": ROC_AUC(),
        }
        from hbt.ml.torch_utils.functions import ignite_train_step, ignite_validation_step

        trainer = Engine(partial(ignite_train_step, model=model, loss_fn=loss_fn, optimizer=optimizer))
        train_evaluator = Engine(partial(ignite_validation_step, model=model))
        val_evaluator = Engine(partial(ignite_validation_step, model=model))

        # Attach metrics to the evaluators
        for name, metric in val_metrics.items():
            metric.attach(train_evaluator, name)

        for name, metric in val_metrics.items():
            metric.attach(val_evaluator, name)

        from torch.utils.tensorboard import SummaryWriter
        logger_path = self.output()["tensorboard"].abspath
        logger.info(f"Creating tensorboard logger at {logger_path}")
        writer = SummaryWriter(logger_path)
        run_name = f"{self._parameter_repr}_{self.version}"
        # How many batches to wait before logging training status
        log_interval = 20
        @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
        def log_training_loss(engine):
            writer.add_scalars(f"{run_name}_per_batch_training", {"loss": engine.state.output}, engine.state.iteration)
            logger.info(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

        @trainer.on(Events.EPOCH_STARTED)
        def reset_dataloaders(trainer):
            training_composite_loader.data_loader.reset()

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            training_composite_loader.data_loader.reset()
            train_evaluator.run(training_composite_loader.data_loader)
            metrics = train_evaluator.state.metrics
            infos = " | ".join([f"Avg {name}: {value:.2f}" for name, value in metrics.items()])
            for name, value in metrics.items():
                writer.add_scalars(f"{run_name}_{name}", {f"training": value }, trainer.state.epoch)
            logger.info(f"Training Results - Epoch[{trainer.state.epoch}] {infos}")

        from ignite.handlers import EarlyStopping

        def score_function(engine):
            val_loss = engine.state.metrics['loss']
            return -val_loss

        handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
        # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
        val_evaluator.add_event_handler(Events.COMPLETED, handler)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            validation_composite_loader.data_loader.reset()
            val_evaluator.run(validation_composite_loader.data_loader)
            # validation_composite_loader.data_loader.reset()
            metrics = val_evaluator.state.metrics
            for name, value in metrics.items():
                writer.add_scalars(f"{run_name}_{name}", {f"validation": value }, trainer.state.epoch)
            infos = " | ".join([f"Avg {name}: {value:.2f}" for name, value in metrics.items()])
            logger.info(f"Validation Results - Epoch[{trainer.state.epoch}] {infos}")

        
        trainer.run(training_composite_loader.data_loader, max_epochs=self.max_epochs)
        writer.close()
        logger.info(f"Saving model to {self.output()['model'].abspath}")
        torch.save(model.state_dict(), self.output()["model"].abspath)

        # good source: https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
        # https://discuss.pytorch.org/t/proper-way-of-using-weightedrandomsampler/73147
        # https://pytorch.org/data/0.10/stateful_dataloader_tutorial.html
        