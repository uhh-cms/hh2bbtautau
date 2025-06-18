# coding: utf-8

"""
Base implementation for DNN models for the HH->bbtautau analysis.
This work is completely based on the work from the HH->bbWW analysis teaam,
specificly Mathis Frahm and Lara Markus.
"""

from __future__ import annotations

from abc import abstractmethod
from functools import partial
import yaml

import law
import order as od

from columnflow.types import Sequence
from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox, DotDict, DerivableMeta
from columnflow.columnar_util import Route, set_ak_column, EMPTY_FLOAT, flat_np_view
from columnflow.config_util import get_datasets_from_process
from columnflow.types import Any, Callable

from hbt.ml.torch_utils.ignite.mixins import IgniteTrainingMixin, IgniteEarlyStoppingMixin
from hbt.ml.torch_utils.ignite.metrics import (
    WeightedROC_AUC, WeightedLoss,
)
from hbt.ml.torch_utils.functions import generate_weighted_loss
from hbt.ml.torch_utils.utils import get_standardization_parameter

from hbt.ml.torch_utils.datasets.handlers import (
    FlatListRowgroupParquetFileHandler, FlatArrowParquetFileHandler,
    WeightedFlatListRowgroupParquetFileHandler,
    RgTensorParquetFileHandler, WeightedRgTensorParquetFileHandler,
    WeightedTensorParquetFileHandler,
)
from hbt.ml.torch_utils.transforms import MoveToDevice
from hbt.ml.torch_utils.datasets import ParquetDataset


# from hbw.util import log_memory
# from hbw.ml.data_loader import MLDatasetLoader, MLProcessData, input_features_sanity_checks
# from hbw.config.processes import prepare_ml_processes

# from hbw.tasks.ml import MLPreTraining

np = maybe_import("numpy")
ak = maybe_import("awkward")
pickle = maybe_import("pickle")
torch = maybe_import("torch")

logger = law.logger.get_logger(__name__)


class MLClassifierBase(MLModel):
    """
    Provides a base structure to implement Multiclass Classifier in Columnflow
    """
    # flag denoting whether the preparation_producer is invoked before evaluate()
    preparation_producer_in_ml_evaluation: bool = False

    # set some defaults, can be overwritten by subclasses or via cls_dict
    # NOTE: the order of processes is crucial! Do not change after training
    _default__processes: tuple[str] = ("tt", "hh_ggf_hbb_htt_kl1_kt1")
    # configs_to_use: tuple[str] = ("22pre_v14_larger_limited", "22post_v14_larger_limited")
    categorical_target_map: dict[str, int] = {
        "tt": 0,
        "hh": 1,
    }

    # identifier of the PrepareMLEvents and MergeMLEvents outputs. Needs to be changed when producing new input features
    store_name: str = "inputs_base"

    # Class for data loading and it's dependencies.
    data_loader = None
    # NOTE: we might want to use the data_loader.hyperparameter_deps instead
    preml_params: set[str] = {"data_loader", "categorical_features", "continuous_features", "train_val_test_split"}

    # NOTE: we split each fold into train, val, test + do k-folding, so we have a 4-way split in total
    # TODO: test whether setting "test" to 0 is working
    train_val_test_split: tuple = (0.75, 0.15, 0.10)
    folds: int = 5

    # training-specific parameters. Only need to re-run training when changing these
    _default__class_factors: dict = {"st": 1, "tt": 1}
    _default__sub_process_class_factors: dict = {"st": 2, "tt": 1}
    _default__negative_weights: str = "handle"
    _default__epochs: int = 50
    _default__batchsize: int = 2 ** 10
    _default__learning_rate: float = 1e-1
    _default__weight_decay: float = 1e-3
    _default__early_stopping_patience: int = 10
    _default__early_stopping_min_epochs: int = 4
    _default__early_stopping_min_diff: float = 0.0
    _default__deterministic_seeds: list[int] | int | None = None
    _default__configs_to_use: tuple[str] = ("22{pre,post}_v14_larger_limited",)

    # parameters to add into the `parameters` attribute to determine the 'parameters_repr' and to store in a yaml file
    bookkeep_params: set[str] = {
        "data_loader", "categorical_features", "continuous_features", "train_val_test_split",
        "processes", "categorical_target_map", "class_factors", "sub_process_class_factors",
        "negative_weights", "epochs", "batchsize", "folds",
        "learning_rate",
        "weight_decay",
        "early_stopping_patience",
        "early_stopping_min_epochs",
        "early_stopping_min_diff",
        "configs_to_use",
        # "deterministic_seeds",
        # "configs_to_use",
    }

    # parameters that can be overwritten via command line
    settings_parameters: set[str] = {
        "processes", "class_factors", "sub_process_class_factors",
        "negative_weights", "epochs", "batchsize",
        "learning_rate",
        "weight_decay",
        "early_stopping_patience",
        "early_stopping_min_epochs",
        "early_stopping_min_diff",
        "deterministic_seeds",
        "configs_to_use",
    }

    settings_list_delimiter = ":"

    _model = None

    @classmethod
    def derive(
        cls,
        cls_name: str,
        bases: tuple = (),
        cls_dict: dict[str, Any] | None = None,
        module: str | None = None,
    ):
        """
        derive but rename classattributes included in settings_parameters to "_default__{attr}"
        """
        if cls_dict:
            for attr, value in cls_dict.copy().items():
                if attr in cls.settings_parameters:
                    cls_dict[f"_default__{attr}"] = cls_dict.pop(attr)

        return DerivableMeta.derive(cls, cls_name, bases, cls_dict, module)

    def __init__(
            self,
            *args,
            folds: int | None = None,
            **kwargs,
    ):
        """
        Initialization function of the MLModel. We first set properties using values from the
        *self.parameters* dictionary that is obtained via the `--ml-model-settings` parameter. If
        the parameter is not set via command line,cthe "_default__{attr}" classattribute is used as
        fallback. Then we cast the parameters to the correct types and store them as individual
        class attributes. Finally, we store the parameters in the `self.parameters` attribute,
        which is used both to create a hash for the output path and to store the parameters in a yaml file.

        Only the parameters in the `settings_parameters` attribute can be overwritten via the command line.
        Only the parameters in the `bookkeep_params` attribute are stored in the `self.parameters` attribute.
        Parameters defined in the `settings_parameters` must be named "_default__{attr}" in the main class definition.
        When deriving, the "_default__" is automatically added.
        Similarly, a parameter starting with "_default__" must be part of the `settings_parameters`.
        """
        super().__init__(*args, **kwargs)

        # logger.warning("Running MLModel init")
        # checks
        if diff := self.settings_parameters.difference(self.bookkeep_params):
            if any(x != "deterministic_seeds" for x in diff):
                # deterministic_seeds is a special case, so we allow it to be in settings_parameters
                # but not in bookkeep_params
                raise Exception(
                    f"settings_parameters {diff} not in bookkeep_params; all customizable settings should"
                    "be bookkept in the parameters.yaml file and the self.parameters_repr to ensure reproducibility",
                )
        if diff := self.preml_params.difference(self.bookkeep_params):
            raise Exception(
                f"preml_params {diff} not in bookkeep_params; all parameters that change the preml_store_name"
                "should be bookkept via the 'self.bookkeep_params' attribute",
            )
        if unknown_params := set(self.parameters.keys()).difference(self.settings_parameters):
            raise Exception(
                f"unknown parameters {unknown_params} passed to the MLModel; only the following "
                f"parameters are allowed: {', '.join(self.settings_parameters)}",
            )

        for param in self.settings_parameters:
            # param is not allowed to exist on class level
            if hasattr(self, param):
                raise ValueError(
                    f"{self.cls_name} has classatribute {param} (value: {getattr(self, param)}) on class level "
                    "but also requests it as configurable via settings_parameters. Maybe you have to rename the",
                    "classattribute in some base class to '_default__{param}'?",
                )
            # set to requested value, fallback on "__default_{param}"
            value = self.parameters.get(param, getattr(self, f"_default__{param}"))
            if isinstance(value, str) and self.settings_list_delimiter in value:
                value = value.split(self.settings_list_delimiter)
            setattr(self, param, value)

        # special case for deterministic_seeds: shouldn't be part of the parameters,
        # so remove the when necessary
        if "deterministic_seeds" in self.parameters:
            del self.parameters["deterministic_seeds"]

        # check that all _default__ attributes are taken care of
        for attr in dir(self):
            if not attr.startswith("_default__"):
                continue
            if not hasattr(self, attr.replace("_default__", "", 1)):
                raise ValueError(
                    f"{self.cls_name} has classatribute {attr} but never sets corresponding property",
                )

        # cast the ml parameters to the correct types if necessary
        self.cast_ml_param_values()

        # overwrite self.parameters with the typecasted values
        for param in self.bookkeep_params:
            self.parameters[param] = getattr(self, param)
            if isinstance(self.parameters[param], set):
                # sets are not hashable, so convert them to sorted tuple
                self.parameters[param] = tuple(sorted(self.parameters[param]))

        # sort the self.settings_parameters
        self.parameters = DotDict(sorted(self.parameters.items()))

        self._model = None
        # sanity check: for each process in "train_nodes", we need to have 1 process with "ml_id" in config

    # @property
    # def used_datasets(self) -> dict[od.Config, set[od.Dataset]]:
    #     configs_to_use = getattr(self, "configs_to_use", self.config_insts)
    #     # expand braces
    #     configs_to_use = law.util.flatten(list(map(law.util.brace_expand, configs_to_use)))
    #     config_insts = list(map(self.analysis_inst.get_config, configs_to_use))
    #     used_datasets = DotDict.wrap({config: self.datasets(config) for config in config_insts})
    #     return used_datasets

    def training_configs(self, requested_configs: Sequence[str]) -> list[str]:
        configs_to_use = getattr(self, "configs_to_use", requested_configs)
    #     # expand braces
        configs_to_use = law.util.flatten(list(map(law.util.brace_expand, configs_to_use)))
        return configs_to_use

    # @property
    # def config_insts(self) -> list[od.Config]:

    # def _setup_configs(self, configs: list[str | od.Config]):
    #     # call setup for specific set, overriding the configs provided by the
    #     # task instance
    #     configs_to_use = getattr(self, "configs_to_use", configs)

    #     config_insts = super()._setup_configs(configs_to_use)
    #     from IPython import embed
    #     embed(header=f"in {self.__class__.__name__}._setup_configs, {config_insts=}")
    #     return config_insts

    def cast_ml_param_values(self):
        """
        Resolve the values of the parameters that are used in the MLModel
        """
        self.processes = tuple(self.processes)
        self.train_val_test_split = tuple(self.train_val_test_split)
        if not isinstance(self.sub_process_class_factors, dict):
            # cast tuple to dict
            self.sub_process_class_factor = {
                proc: weight for proc, weight in [s.split(":") for s in self.sub_process_class_factor]
            }
        # cast weights to int and remove processes not used in training
        self.ml_model_weights = {
            proc: int(weight)
            for proc, weight in self.sub_process_class_factors.items()
            if proc in self.processes
        }
        self.negative_weights = str(self.negative_weights)
        self.epochs = int(self.epochs)
        self.batchsize = int(self.batchsize)
        self.folds = int(self.folds)

        configs_to_use = getattr(self, "configs_to_use", self.config_insts)
        self.configs_to_use = law.util.flatten(list(map(law.util.brace_expand, configs_to_use)))
        # checks
        if self.negative_weights not in ("ignore", "abs", "handle"):
            raise Exception(
                f"negative_weights {self.negative_weights} not in ('ignore', 'abs', 'handle')",
            )

    @property
    def preml_store_name(self):
        """
        Create a hash of the parameters that are used in the MLModel to determine the 'preml_store_name'.
        The preml_store_name is cached to ensure that it does not change during the lifetime of the object.
        """
        preml_params = {param: self.parameters[param] for param in self.preml_params}
        preml_store_name = law.util.create_hash(sorted(preml_params.items()))
        if hasattr(self, "_preml_store_name") and self._preml_store_name != preml_store_name:
            raise Exception(
                f"preml_store_name changed from {self._preml_store_name} to {preml_store_name};"
                "this should not happen",
            )
        self._preml_store_name = preml_store_name
        return self._preml_store_name

    @property
    def parameters_repr(self):
        """
        Create a hash of the parameters to store as part of the output path.
        The repr is cached to ensure that it does not change during the lifetime of the object.
        """
        if not self.parameters:
            return ""
        internal_parameters_repr = law.util.create_hash(sorted([
            (name, val) for name, val in self.parameters.items()
            # deterministic seed might be part of branch data of task, so exclude it from
            # the representation
            if not any(name == x for x in ("deterministic_seed", "deterministic_seeds"))
        ]))

        if hasattr(self, "_parameters_repr") and self._parameters_repr != internal_parameters_repr:
            raise Exception(
                f"parameters_repr changed from {self._parameters_repr} to {internal_parameters_repr};"
                "this should not happen",
            )
        self._parameters_repr = internal_parameters_repr
        return self._parameters_repr

    def valid_ml_id_sanity_check(self):
        """
        ml_ids must include 0 and each following integer up to the number of requested train_nodes
        """
        for p in self.process_insts:
            sub_process_class_factor = p.x("sub_process_class_factor", None)
            if sub_process_class_factor is None:
                logger.warning(f"Process {p.name} has no 'sub_process_class_factor' aux; will be set to 1.")
                p.x.sub_process_class_factor = 1
            ml_id = p.x("ml_id", None)
            if ml_id is None:
                logger.warning(f"Process {p.name} has no 'ml_id' aux; will be set to -1.")
                p.x.ml_id = -1

        ml_ids = sorted(set(p.x.ml_id for p in self.process_insts) - {-1})

        if len(ml_ids) != len(self.train_nodes.keys()):
            raise Exception(f"ml_ids {ml_ids} does not match number of requested train_nodes {self.train_nodes.keys()}")

        expected_id = 0
        while ml_ids:
            _id = ml_ids.pop(0)
            if _id == expected_id:
                # next id should be previous value + 1
                expected_id += 1
                continue
            else:
                raise ValueError(f"Invalid combination of ml ids {set(p.x.ml_id for p in self.process_insts)}")

        logger.debug("ml_id_sanity_check passed")

    def _build_categorical_target(self, dataset: str):
        for key in self.categorical_target_map.keys():
            if dataset.startswith(key):
                return self.categorical_target_map[key]
        raise ValueError(f"Dataset {dataset} not in categorical target map")

    def setup(self) -> None:
        """ function that is run as part of the setup phase. Most likely overwritten by subclasses """
        if self.config_inst.has_tag(f"{self.cls_name}_called"):
            # call this function only once per config
            return
        logger.debug(
            f"Setting up MLModel {self.cls_name} (parameter hash: {self.parameters_repr}), "
            f"parameters: \n{self.parameters}",
        )
        # dynamically add processes and variables for the quantities produced by this model
        # NOTE: this function might not be called for all configs when the requested configs
        # between MLTraining and the requested task are different

        # setup processes for training

        # setup variables
        # for proc in self.processes:
        for proc, node_config in self.categorical_target_map.items():
            for config_inst in self.config_insts:
                if f"mlscore.{proc}" not in config_inst.variables:
                    proc_inst = config_inst.get_process(proc, default=None)
                    config_inst.add_variable(
                        name=f"mlscore.{proc}",
                        expression=f"mlscore.{proc}",
                        null_value=EMPTY_FLOAT,
                        binning=(1000, 0., 1.),
                        x_title=f"DNN output score {proc_inst.x('ml_label', proc) if proc_inst else proc}",
                        aux={
                            "rebin": 25,
                            "rebin_config": {
                                "processes": [proc],
                                "n_bins": 4,
                            },
                        },  # automatically rebin to 40 bins for plotting tasks
                    )

                    config_inst.add_variable(
                        name=f"best_mlscore.{proc}",
                        expression=f"best_mlscore.{proc}",
                        null_value=EMPTY_FLOAT,
                        binning=(1000, 0., 1.),
                        x_title=f"Best DNN output score {proc_inst.x('ml_label', proc) if proc_inst else proc}",
                        aux={
                            "rebin": 25,
                            "rebin_config": {
                                "processes": [proc],
                                "n_bins": 4,
                            },
                        },  # automatically rebin to 40 bins for plotting tasks
                    )

        # add tag to allow running this function just once
        self.config_inst.add_tag(f"{self.cls_name}_called")

    @property
    def process_insts(self):
        if hasattr(self, "_process_insts"):
            return self._process_insts
        return [self.config_inst.get_process(proc) for proc in self.processes]

    def preparation_producer(self: MLModel, analysis_inst: od.Analysis):
        """ producer that is run as part of PrepareMLEvents and MLEvaluation (before `evaluate`) """
        return "ml_preparation"

    def training_calibrators(self, analysis_inst: od.Analysis, requested_calibrators: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        # NOTE: since automatic resolving is not working here, we do it ourselves
        return requested_calibrators or [analysis_inst.x.default_calibrator]

    def training_producers(self, analysis_inst: od.Analysis, requested_producers: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        # NOTE: might be nice to keep the "pre_ml_cats" for consistency, but running two
        # categorization Producers in the same workflow is messy, so we skip it for now
        # return requested_producers or ["event_weights", "pre_ml_cats", analysis_inst.x.ml_inputs_producer]
        # return requested_producers or ["event_weights", analysis_inst.x.ml_inputs_producer]
        return ["default", analysis_inst.x.ml_inputs_producer]

    def evaluation_producers(self, analysis_inst: od.Analysis, requested_producers: Sequence[str]) -> list[str]:
        return self.training_producers(analysis_inst, requested_producers)

    def sandbox(self, task: law.Task) -> str:
        # venv_ml_tf sandbox but with scikit-learn and restricted to tf 2.11.0
        return dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_torch.sh")

    def init_optimizer(self, learning_rate=1e-3, weight_decay=1e-5, model=None) -> None:
        from torch.optim import AdamW
        if not model:
            model = self
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def datasets(self, config_inst: od.Config) -> set[od.Dataset]:
        used_datasets = set()
        for i, proc in enumerate(self.processes):
            if not config_inst.has_process(proc):
                raise Exception(f"Process {proc} not included in the config {config_inst.name}")

            proc_inst = config_inst.get_process(proc)
            # NOTE: this info is accessible during training but probably not afterwards in other tasks
            # --> move to setup? or store in some intermediate output file?
            # proc_inst.x.ml_id = i
            # proc_inst.x.sub_process_class_factor = self.sub_process_class_factors.get(proc, 1)

            # get datasets corresponding to this process
            dataset_insts = [
                dataset_inst for dataset_inst in
                get_datasets_from_process(config_inst, proc, strategy="all")
            ]

            # store assignment of datasets and processes in the instances
            for dataset_inst in dataset_insts:
                dataset_inst.x.ml_process = proc
            proc_inst.x.ml_datasets = [dataset_inst.name for dataset_inst in dataset_insts]

            # check that no dataset is used multiple times
            if datasets_already_used := used_datasets.intersection(dataset_insts):
                raise Exception(f"{datasets_already_used} datasets are used for multiple processes")
            used_datasets |= set(dataset_insts)

        return used_datasets

    @property
    @abstractmethod
    def continuous_features(self):
        pass

    @property
    @abstractmethod
    def categorical_features(self):
        pass

    def uses(self, config_inst: od.Config) -> set[Route | str]:
        columns = set(self.categorical_features) | set(self.continuous_features)

        # TODO: switch to full event weight
        # TODO: this might not work with data, to be checked
        columns.add("process_id")
        columns.add("normalization_weight")
        columns.add("stitched_normalization_weight")
        columns.add("event_weight")
        return columns

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        produced = set()

        for proc in self.categorical_target_map.keys():
            produced.add(f"mlscore.{proc}")
            produced.add(f"best_mlscore.{proc}")
            produced.add(f"mlscore_std.{proc}")
            produced.add(f"best_mlscore_std.{proc}")

        return produced

    def output(self, task: law.Task) -> dict[str, law.FileSystemTarget]:
        branch_data = task.branch_data
        fold = None
        suffix = ""

        if isinstance(branch_data, dict) and (seed := branch_data.get("deterministic_seed", None)) is not None:
            fold = branch_data.get("fold")
            suffix = f"_seed_{seed}"
        else:
            fold = task.branch

        target = task.target(f"mlmodel_f{fold}of{self.folds}", dir=True)
        # declare the main target

        # from IPython import embed
        # embed(header=f"in {self.__class__.__name__}.output")
        ml_name = self.__class__.__name__
        # from IPython import embed
        # embed(header=f"check output path for local_target")
        output = {
            "model": target.child(f"torch_model_{ml_name}_f{fold}of{self.folds}{suffix}.pt", type="f"),
            "tensorboard": task.local_target(
                f"tb_logger_{ml_name}_f{fold}of{self.folds}{suffix}",
                dir=True,
                optional=True,
            ),
            "model_architecture": task.target("model_architecture", type="f", dir=False),
        }
        output["aux_files"] = {
            "input_features": target.child("input_features.pkl", type="f"),
            "parameter_summary": target.child(f"parameter_summary{suffix}.yaml", type="f"),
        }
        return DotDict.wrap(output)

    def load_data(
        self,
        task: law.Task,
        input: Any,
        output: DotDict[str, law.LocalDirectoryTarget],
        device: torch.device | str = "cpu",
    ):
        datasets = set(x.name for config in self.used_datasets for x in self.used_datasets[config])

        # the task is expected to know a couple of details of data loading, so add this information
        task.batch_size = self.batchsize
        task.load_parallel_cores = 0

        extract_probabilities_fn = partial(
            self.extract_probabilities,
            input=input,
        )

        self.model.init_dataset_handler(
            task=task,
            inputs=input,
            extract_dataset_paths_fn=self.extract_dataset_paths_fn,
            datasets=list(datasets),
            extract_probability_fn=extract_probabilities_fn,
            device=device,
        )

        def sort_set(set_name: str, key=str) -> list[str]:
            return sorted(getattr(self, set_name, set()), key=key)

        out_dict = {k: sort_set(k) for k in ("continuous_features", "categorical_features")}
        # store input features as an output
        output.aux_files.input_features.dump(out_dict, formatter="pickle")

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @model.setter
    def model(self, input) -> None:
        if not isinstance(input, torch.nn.Module):
            raise ValueError(f"Input must be of type torch.nn.Module, got {type(input)}")
        self._model = input

    def train(
        self,
        task: law.Task,
        input: Any,
        output: DotDict[str, law.LocalDirectoryTarget],
    ) -> ak.Array:
        """ Training function that is called during the MLTraining task """
        import torch
        # np.random.seed(1337)  # for reproducibility
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"Running pytorch on {device}")
        # deterministic seed may be part of the branch data of the task,
        # so look for it
        branch_data = task.branch_data
        deterministic_seed = None
        if isinstance(branch_data, dict):
            deterministic_seed = branch_data.get("deterministic_seed", None)

        if deterministic_seed is not None and deterministic_seed >= 0:
            # set seed for reproducibility
            logger.info(f"Setting deterministic seed to {deterministic_seed}")
            # self.parameters["deterministic_seed"] = deterministic_seed
            torch.manual_seed(deterministic_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(deterministic_seed)
                torch.cuda.manual_seed_all(deterministic_seed)
            np.random.seed(deterministic_seed)
            import random
            random.seed(deterministic_seed)
            torch.use_deterministic_algorithms(True)
        # input preparation
        self.model = self.prepare_ml_model(task)
        self.model = self.model.to(device)

        with law.localize_file_targets(input, mode="r") as inp:

            self.load_data(task, inp, output, device=device)

            self.model.init_optimizer(learning_rate=self.learning_rate, weight_decay=self.weight_decay)

            # hyperparameter bookkeeping
            output.aux_files.parameter_summary.dump(dict(self.parameters), formatter="yaml")
            logger.info(f"Training will be run with the following parameters: \n{self.parameters}")
            #
            # model preparation
            #

            run_name = f"{self.parameters_repr}_{task.version}"

            # How many batches to wait before logging training status

            # run only when datastitics exists
            # set statitical modes for preprocessing
            if hasattr(self.model, "dataset_statistics"):
                self.model.setup_preprocessing()
            # move all model parameters to device

            self.model.start_training(run_name=run_name, max_epochs=self.epochs)
            logger.info(f"Saving model to {output['model'].abspath}")
            torch.save(self.model.state_dict(), output["model"].abspath)
            torch.save(self.model, output["model_architecture"].abspath)
        return

    @abstractmethod
    def prepare_ml_model(
        self,
        task: law.Task,
    ) -> torch.nn.Module:
        """ Function to define the ml model. Needs to be implemented in daughter class """
        return

    @abstractmethod
    def start_training(
        self,
        run_name,
        max_epochs,
    ) -> None:
        """ Function to run the ml training loop. Needs to be implemented in daughter class """
        return

    def requires(self, task: law.Task) -> dict[str, Any]:
        """
        Requirements for the MLModel. This is used to determine the dependencies of the MLModel.
        """
        reqs = dict()
        reqs["stats"] = {
            config_inst.name: {
                dataset_inst.name: task.reqs.MergeMLStats.req_different_branching(
                    task,
                    config=config_inst.name,
                    dataset=dataset_inst.name,
                )
                for dataset_inst in dataset_insts
            }
            for config_inst, dataset_insts in self.used_datasets.items()
        }
        return reqs

    @property
    @abstractmethod
    def eval_activation(self) -> Callable | str:
        pass

    def evaluate_training_events(
        self,
        models: list[torch.nn.Module],
        fold_indices: ak.Array,
        data: ParquetDataset,
        outputs_view: np.ndarray,
        outputs_std_view: np.ndarray,
        template: ak.Array,
    ):
        from hbt.ml.torch_utils.utils import MLEnsembleWrapper

        step_size = 1
        deterministic_seeds = getattr(self, "deterministic_seeds", None)
        if isinstance(deterministic_seeds, list):
            step_size = len(deterministic_seeds)

        for fold_idx, model_range_start in enumerate(range(0, len(models), step_size)):
            fold_models = models[model_range_start : model_range_start + step_size]
            logger.info(f"evaluating fold {fold_idx} with {len(fold_models)} models")
            eval_wrapper = MLEnsembleWrapper(fold_models, final_activation=self.eval_activation)
            fold_mask = fold_indices == fold_idx
            flat_bc_mask = ak.flatten(ak.broadcast_arrays(fold_mask[..., None], template)[0])
            fold_data = data[fold_mask]

            with torch.no_grad():
                output_mean, output_std = eval_wrapper(fold_data)

            outputs_view[flat_bc_mask] = ak.flatten(output_mean)
            if step_size > 1:
                outputs_std_view[flat_bc_mask] = ak.flatten(output_std)

    def evaluate_orthogonal_events(
        self,
        models: list[torch.nn.Module],
        data: ParquetDataset,
        outputs_view: np.ndarray,
        outputs_std_view: np.ndarray,
    ):
        from hbt.ml.torch_utils.utils import MLEnsembleWrapper
        eval_wrapper = MLEnsembleWrapper(models, final_activation=self.eval_activation)

        with torch.no_grad():
            output_mean, output_std = eval_wrapper(data[:])

        outputs_view[:] = ak.flatten(output_mean)
        outputs_std_view[:] = ak.flatten(output_std)

    def evaluate(
        self,
        task: law.Task,
        events: ak.Array,
        models: list(torch.nn.Module),
        fold_indices: ak.Array,
        events_used_in_training: bool = True,
    ) -> None:
        """
        Evaluation function that is run as part of the MLEvaluation task
        """
        from hbt.ml.torch_utils.datasets import TensorParquetDataset

        # create place holders for the mean and std of the model predictions
        n_classes = len(self.categorical_target_map)
        template = np.array([EMPTY_FLOAT] * n_classes, ndmin=2)

        outputs: ak.Array = ak.broadcast_arrays(fold_indices[..., None], template)[1]
        outputs_std: ak.Array = ak.broadcast_arrays(fold_indices[..., None], template)[1]

        # create view for outputs for later filling
        outputs_view = flat_np_view(outputs)
        outputs_std_view = flat_np_view(outputs_std)

        data = TensorParquetDataset(
            categorical_features=self.categorical_features,
            continuous_features=self.continuous_features,
            input=events,
        )

        if events_used_in_training:
            self.evaluate_training_events(
                models=models,
                fold_indices=fold_indices,
                data=data,
                outputs_view=outputs_view,
                outputs_std_view=outputs_std_view,
                template=template,
            )
        else:
            self.evaluate_orthogonal_events(
                models=models,
                data=data,
                outputs_view=outputs_view,
                outputs_std_view=outputs_std_view,
            )

        pred_cls = ak.argmax(outputs, axis=-1)
        for proc, proc_idx in self.categorical_target_map.items():

            events = set_ak_column(events, f"mlscore.{proc}", outputs[:, proc_idx])
            events = set_ak_column(events, f"mlscore_std.{proc}", outputs_std[:, proc_idx])

            mask = pred_cls == proc_idx

            events = set_ak_column(events, f"best_mlscore.{proc}", ak.where(mask, outputs[:, proc_idx], EMPTY_FLOAT))
            events = set_ak_column(
                events,
                f"best_mlscore_std.{proc}",
                ak.where(mask, outputs_std[:, proc_idx], EMPTY_FLOAT),
            )

        return events

    def extract_dataset_paths_fn(self, input, dataset):
        events = input["events"]
        return [x["mlevents"].abspath for c in events for x in events[c][dataset]]

    def extract_probabilities(self, dataset, input=None, keyword="sum_norm_weights_per_process"):
        input = DotDict.wrap(input)
        stats = input.model.stats
        allowed_process_ids = [
            x[0].id
            for proc in self.process_insts
            for x in proc.walk_processes(include_self=True)
        ]
        expected_events = list()
        for config in self.config_insts:
            lumi = config.x.luminosity.nominal
            target = stats[config.name][dataset]["stats"]
            dataset_stats = target.load(formatter="json")
            sel_stats = dataset_stats.get(keyword, dict())
            allowed = list(filter(lambda x: int(x) in allowed_process_ids, sel_stats.keys()))
            xs = sum([val for x, val in sel_stats.items() if x in allowed])
            expected_events.append(xs * lumi)
        return sum(expected_events)


class BinaryMLBase(MLClassifierBase):
    """ Example class how to implement a DNN from the MLClassifierBase """

    # optionally overwrite input parameters
    _default__epochs: int = 10
    ml_cls = None

    @property
    def continuous_features(self) -> list[Route | str]:
        if not self._continuous_features:
            columns = {
                "lepton1.{px,py,pz,energy,mass}",
                "lepton2.{px,py,pz,energy,mass}",
                "bjet1.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
                "bjet2.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
                "fatjet.{px,py,pz,energy,mass}",
            }
            self._continuous_features = set()
            self._continuous_features.update(*list(map(Route, law.util.brace_expand(obj)) for obj in columns))
            self._continuous_features = sorted(self._continuous_features, key=str)
        return self._continuous_features

    @property
    def categorical_features(self) -> set[Route | str]:
        return set()

    @property
    def eval_activation(self) -> Callable | str:
        return lambda x: x

    def prepare_ml_model(
        self,
        task: law.Task | None = None,
    ):
        """
        Minimal implementation of a ML model
        """
        from hbt.ml.torch_models.binary import WeightedTensorFeedForwardNet

        logger_path = None
        if task and not isinstance(task, DotDict):
            logger_target = self.output(task).get("tensorboard", None)
            logger_path = logger_target.abspath if logger_target else None

        # fake the task to propagate to the model and initialize the model instance
        # in case there is no task instance provided
        dummy_task = DotDict.wrap({"param_kwargs": dict()})
        model = WeightedTensorFeedForwardNet(tensorboard_path=logger_path, logger=logger, task=(task or dummy_task))
        model.categorical_target_map = self.categorical_target_map
        model.continuous_features = self.continuous_features
        model.categorical_features = self.categorical_features

        return model

    def open_model(self, task, *args, **kwargs):
        model = self.prepare_ml_model(task)

        from IPython import embed
        embed(header=f"in {self.__class__.__name__}.open_model")

    def start_training(self, run_name, max_epochs) -> None:
        return self.model.start_training(run_name, max_epochs)


class MultiClsMLBase(MLClassifierBase):
    _default__epochs: int = 10
    ml_cls = None

    _default__processes: tuple[str] = ("tt", "hh_ggf_hbb_htt_kl1_kt1", "dy")
    categorical_target_map: dict[str, int] = {
        "hh": 0,
        "tt": 1,
        "dy": 2,
    }

    @property
    def continuous_features(self) -> set[Route | str]:
        columns = {
            "lepton1.{px,py,pz,energy,mass}",
            "lepton2.{px,py,pz,energy,mass}",
            "bjet1.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
            "bjet2.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
            "fatjet.{px,py,pz,energy,mass}",
        }
        final_set = set()
        final_set.update(*list(map(str, law.util.brace_expand(obj)) for obj in columns))
        return final_set

    @property
    def categorical_features(self) -> list[Route | str]:
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
        self._categorical_features = set()
        self._categorical_features.update(*list(map(str, law.util.brace_expand(obj)) for obj in columns))
        self._categorical_features = sorted(self._categorical_features, key=str)
        return self._categorical_features

    @property
    def eval_activation(self) -> Callable | str:
        return torch.nn.functional.softmax

    def prepare_ml_model(
        self,
        task: law.Task | None = None,
    ):
        """
        Minimal implementation of a ML model
        """
        from hbt.ml.torch_models.resnet import WeightedResnetTest
        logger_path = None
        if task and not isinstance(task, DotDict):
            logger_target = self.output(task).get("tensorboard", None)
            logger_path = logger_target.abspath if logger_target else None

        # fake the task to propagate to the model and initialize the model instance
        # in case there is no task instance provided
        dummy_task = DotDict.wrap({"param_kwargs": dict()})

        model = WeightedResnetTest(tensorboard_path=logger_path, logger=logger, task=(task or dummy_task))
        model.categorical_target_map = self.categorical_target_map
        model.continuous_features = self.continuous_features
        model.categorical_features = self.categorical_features

        return model

    def open_model(self, inputs, *args, **kwargs):

        # from IPython import embed
        # embed(header=f"in {self.__class__.__name__}.open_model")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.prepare_ml_model()

        # load paths to the model state dicts and load the weights
        model_path = inputs.model.abspath
        logger.info(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    def start_training(self, run_name, max_epochs) -> None:
        return self.model.start_training(run_name, max_epochs)


class BogNetBase(MultiClsMLBase):

    def training_producers(self, analysis_inst: od.Analysis, requested_producers: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        # NOTE: might be nice to keep the "pre_ml_cats" for consistency, but running two
        # categorization Producers in the same workflow is messy, so we skip it for now
        # return requested_producers or ["event_weights", "pre_ml_cats", analysis_inst.x.ml_inputs_producer]
        # return requested_producers or ["event_weights", analysis_inst.x.ml_inputs_producer]
        return ["default", f"{analysis_inst.x.ml_inputs_producer}_no_rotation"]

    def prepare_ml_model(self, task: law.Task | None = None):
        from hbt.ml.torch_models.bognet import UpdatedBogNet

        logger_path = None
        if task and not isinstance(task, DotDict):
            logger_target = self.output(task).get("tensorboard", None)
            logger_path = logger_target.abspath if logger_target else None

        # fake the task to propagate to the model and initialize the model instance
        # in case there is no task instance provided
        dummy_task = DotDict.wrap({"param_kwargs": dict()})
        model = UpdatedBogNet(tensorboard_path=logger_path, logger=logger, task=(task or dummy_task))
        model.categorical_target_map = self.categorical_target_map
        # check if input feature set is set consistently
        def compare_features(feature_set_name):
            ml_inst_set = sorted(map(str, self.continuous_features), key=str)
            model_inst_set = sorted(map(str, model.continuous_features), key=str)
            if not all(x == y for x, y in zip(ml_inst_set, model_inst_set)):
                raise ValueError(
                    f"Input feature set {feature_set_name} is not consistent between MLModel and BogNet model. "
                    f"{self.__class__.__name__}: {ml_inst_set}, {model.__class__.__name__}: {model_inst_set}",
                )

        compare_features("continuous_features")
        compare_features("categorical_features")

        return model

    def _process_columns(self, columns: Container[str]) -> list[str]:
        final_set = set()
        final_set.update(*list(map(str, law.util.brace_expand(obj)) for obj in columns))
        return sorted(final_set, key=str)

    @property
    def categorical_features(self) -> list[str]:
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
        return self._process_columns(columns)

    @property
    def continuous_features(self) -> list[str]:
        columns = {
            "lepton1.{px,py,pz,energy,mass}",
            "lepton2.{px,py,pz,energy,mass}",
            "bjet1.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
            "bjet2.{px,py,pz,energy,mass,btagDeepFlavB,btagDeepFlavCvB,btagDeepFlavCvL,hhbtag}",
            "fatjet.{px,py,pz,energy,mass}",
        }
        return self._process_columns(columns)

    # @property
    # def model(self):
    #     if not self._model:
    #         self.model = self.prepare_ml_model(self.task)
    #     return self._model


# dervive another model from the ExampleDNN class with different class attributes
# from hbt.ml.torch_models.binary import WeightedTensorFeedForwardNet
example_test = BinaryMLBase.derive("example_test", cls_dict={
    "epochs": 5,
    # "ml_cls": WeightedTensorFeedForwardNet,
})

resnet_test = MultiClsMLBase.derive("resnet_test", cls_dict={
    "epochs": 10,
    # "ml_cls": WeightedResnetTest,
})

bognet_test = BogNetBase.derive("bognet_test", cls_dict={
    "epochs": 10,
    # "ml_cls": UpdatedBogNet,
})

bognet_ensemble_test = BogNetBase.derive("bognet_ensemble_test", cls_dict={
    "epochs": 10,
    "deterministic_seeds": [0, 1, 2, 3, 4],
    # "ml_cls": UpdatedBogNet,
})

bognet_ensemble_full_stats = BogNetBase.derive("bognet_ensemble_full_stats", cls_dict={
    "epochs": 100,
    "deterministic_seeds": [0, 1, 2, 3, 4],
    "configs_to_use": ("{22,23}{pre,post}_v14",),
    # "ml_cls": UpdatedBogNet,
})

# load all ml modules here
if law.config.has_option("analysis", "ml_modules"):
    for m in law.config.get_expanded("analysis", "ml_modules", [], split_csv=True):
        logger.debug(f"loading ml module '{m}'")
        maybe_import(m.strip())
