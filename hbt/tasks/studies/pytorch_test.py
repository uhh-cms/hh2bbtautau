from __future__ import annotations
import law.decorator

from columnflow.tasks.union import UniteColumns, UniteColumnsWrapper
from columnflow.util import dev_sandbox, DotDict, maybe_import
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.selection import MergeSelectionStats

from hbt.tasks.base import HBTTask
# from hbt.ml.pytorch_util import ListDataset, MapAndCollate
import law
import luigi

logger = law.logger.get_logger(__name__)

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
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

    sub_targets = {  # noqa: F841
        d: UniteColumns(
            config="run3_2022_postEE_limited",
            version="pytorch_test",
            dataset=d,
        ).target()
        for d in datasets
    }

    from IPython import embed
    embed(header="finished running task")
    return task.target()


def main():
    output_paths = run_task()
    from IPython import embed
    embed(header=f"output_paths: {output_paths}")


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

    learning_rate = luigi.FloatParameter(
        default=1e-3,
        significant=False,
        description="Learning rate to use in training. Default: 1e-3",
    )

    weight_decay = luigi.FloatParameter(
        default=1e-5,
        significant=False,
        description="Weight decay to use in training. Default: 1e-5",
    )
    early_stopping_patience = luigi.IntParameter(
        default=10,
        significant=False,
        description="Patience for early stopping. Default: 10",
    )
    early_stopping_min_epochs = luigi.IntParameter(
        default=1,
        significant=False,
        description="Minimum epochs before early stopping kicks in. Default: 1",
    )
    early_stopping_min_diff = luigi.FloatParameter(
        default=0,
        significant=False,
        description="Minimum difference for validation loss between epochs for early stopping. Default: 0",
    )

    use_tensorboard_logger = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Track training process with tensorboard. Default: True",
    )

    deterministic_seed = luigi.IntParameter(
        default=-1,
        significant=False,
        description="If set to >= 0, will be set as random seed for torch, python and numpy. Default: -1",
    )

    additional_params: list[str] = list()

    def create_branch_map(self):
        return dict(enumerate(self.models))

    def workflow_requires(self):
        reqs = super().workflow_requires()

        self.resolved_configs = set()
        self.resolved_datasets = set()
        self.resolved_shifts = set()
        for config, shift, dataset in self.wrapper_parameters:
            self.resolved_configs.add(config)
            self.resolved_datasets.add(dataset)
            self.resolved_shifts.add(shift)

        reqs["events"] = DotDict.wrap({
            d: DotDict.wrap({
                config: self.reqs.UniteColumns.req(
                    self,
                    dataset=d,
                    config=config,
                    branch=-1,
                    _exclude=["datasets", "configs", "branch", "branches"])
                for config in self.resolved_configs
            })
            for d in self.resolved_datasets
        })
        # also require selection stats

        reqs["selection_stats"] = DotDict.wrap({
            d: DotDict.wrap({
                config: self.reqs.MergeSelectionStats.req_different_branching(
                    self,
                    dataset=d,
                    config=config,
                    _exclude=["datasets", "configs"],
                )
                for config in self.resolved_configs
            })
            for d in self.resolved_datasets
        })

        # from IPython import embed; embed(header="requires")
        return reqs

    def requires(self):
        reqs = DotDict()
        self.resolved_configs = set()
        self.resolved_datasets = set()
        self.resolved_shifts = set()
        for config, shift, dataset in self.wrapper_parameters:
            self.resolved_configs.add(config)
            self.resolved_datasets.add(dataset)
            self.resolved_shifts.add(shift)

        reqs["events"] = DotDict.wrap({
            d: DotDict.wrap({
                config: self.reqs.UniteColumns.req(
                    self,
                    dataset=d,
                    config=config,
                    branch=-1,
                    _exclude=["datasets", "configs", "branch", "branches"])
                for config in self.resolved_configs
            })
            for d in self.resolved_datasets
        })
        # also require selection stats

        reqs["selection_stats"] = DotDict.wrap({
            d: DotDict.wrap({
                config: self.reqs.MergeSelectionStats.req_different_branching(
                    self,
                    dataset=d,
                    config=config,
                    _exclude=["datasets", "configs"],
                )
                for config in self.resolved_configs
            })
            for d in self.resolved_datasets
        })

        # from IPython import embed; embed(header="requires")
        return reqs

    def output(self):
        return {
            "model": self.target(f"torch_model_{self.branch_data}.pt"),
            "parameter_summary": self.target(f"parameter_summary_{self.branch_data}.json"),
            "tensorboard": self.local_target(f"tb_logger_{self.branch_data}", dir=True, optional=True),
        }

    @property
    def _parameter_repr(self) -> str:
        self.additional_params = [
            "learning_rate",
            "weight_decay",
            "early_stopping_patience",
            "early_stopping_min_epochs",
            "early_stopping_min_diff",
            "deterministic_seed",
        ]
        param_repr = f"bs_{self.batch_size}__max_epochs_{self.max_epochs}"
        if self.additional_params:
            param_parts = [f"{p}_{getattr(self, p)}" for p in sorted(self.additional_params)]
            param_repr += "__{}".format(law.util.create_hash(param_parts))
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
    @law.decorator.localize(input=True, output=False)
    def run(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"Running pytorch on {device}")
        outputs = self.output()
        with outputs["model"].localize() as model_output:

            from hbt.ml.torch_models import model_clss
            model_cls = model_clss.get(self.branch_data, None)
            if not model_cls:
                raise ValueError(f"Unable to load model {self.branch_data}, available list: {model_clss}")
            logger_path = self.output()["tensorboard"].abspath

            if self.deterministic_seed >= 0:
                # set seed for reproducibility
                torch.manual_seed(self.deterministic_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(self.deterministic_seed)
                    torch.cuda.manual_seed_all(self.deterministic_seed)
                np.random.seed(self.deterministic_seed)
                import random
                random.seed(self.deterministic_seed)
                torch.use_deterministic_algorithms(True)

            model = model_cls(tensorboard_path=logger_path, logger=logger, task=self)

            model.init_dataset_handler(task=self, device=device)
            model.init_optimizer(learning_rate=self.learning_rate, weight_decay=self.weight_decay)
            run_name = f"{self._parameter_repr}_{self.version}"

            # How many batches to wait before logging training status

            # run only when datastitics exists
            # set statitical modes for preprocessing
            if hasattr(model, "dataset_statitics"):
                model.setup_preprocessing()
            # move all model parameters to device
            model = model.to(device)

            model.start_training(run_name=run_name, max_epochs=self.max_epochs)
            logger.info(f"Saving model to {model_output.abspath}")
            torch.save(model.state_dict(), model_output.abspath)

        # save model parameter summary
        params = {
            "model": self.branch_data,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "best_epoch": getattr(model.trainer, "best_epoch", model.trainer.state.epoch),
        }
        params.update({
            k: getattr(self, k) for k in self.additional_params
        })
        self.output()["parameter_summary"].dump(params, indent=4, formatter="json")

        # good source: https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
        # https://discuss.pytorch.org/t/proper-way-of-using-weightedrandomsampler/73147
        # https://pytorch.org/data/0.10/stateful_dataloader_tutorial.html
