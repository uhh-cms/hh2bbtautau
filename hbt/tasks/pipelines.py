# coding: utf-8

"""
Different pipelines and triggers.
"""

from __future__ import annotations

import os
import re
import hashlib

import luigi
import law
import order as od

from columnflow.tasks.framework.base import ConfigTask
from columnflow.tasks.cms.inference import CreateDatacards
from columnflow.production import Producer
from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.util import expand_path, maybe_import

from hbt.tasks.base import HBTTask
from hbt.inference.default import default as default_inference_model

np = maybe_import("numpy")


class TorchModelToDatacards(HBTTask, ConfigTask, law.WrapperTask):
    """
    Example usage:

    .. code-block:: bash

        law run hbt.TorchModelToDatacards \
            --configs 22pre_v14 \
            --torch-model model_name:/path/to/model.pt2 \
            --version dev_model_name
    """

    single_config = False

    torch_model = luigi.Parameter(
        description="name and path to the torch model file to load in the format 'name:path'; no default",
    )
    custom_torch_model_hash = luigi.Parameter(
        default=law.NO_STR,
        description="custom hash for the torch model file; when NO_STR, it is computed from the model file contents; "
        "default: NO_STR",
    )
    base_inference_model = luigi.Parameter(
        default="default_no_shifts",
        description="name of the inference model to use for writing datacards; default: 'default_no_shifts'",
    )
    logit_variable = luigi.BoolParameter(
        default=False,
        description="whether to use transorm the model output back to logit space; default: False",
    )
    fine_variable = luigi.BoolParameter(
        default=False,
        description="whether to use extra fine binning; default: False",
    )
    base_producer = luigi.Parameter(
        default="_external_dnn",
        description="base class for the dynamically created producer; default: '_external_dnn'",
    )
    hist_hooks = CreateDatacards.hist_hooks

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # check torch model
        if not (m := re.match(r"^([^:]+):(.+)$", self.torch_model)):
            raise ValueError(f"invalid torch model format: {self.torch_model}")
        self.torch_model_name = m.group(1)
        self.torch_model_path = expand_path(m.group(2))
        if not os.path.exists(self.torch_model_path):
            raise FileNotFoundError(f"torch model file does not exist: {self.torch_model_path}")
        self.torch_model_hash = (
            self.custom_torch_model_hash
            if self.custom_torch_model_hash not in {law.NO_STR, "", None}
            else self.compute_file_hash(self.torch_model_path)
        )

        # check variable flags
        if self.fine_variable and not self.hist_hooks:
            raise ValueError("--fine-variable is defined, but --hist-hooks (for flat-s binning!) is empty")

        # state attributes
        self.producer_name = f"torch_{self.torch_model_name}_{self.torch_model_hash}"
        self.variable_name = f"torch_{self.torch_model_name}_hh"
        if self.logit_variable:
            self.variable_name += "_logit"
        if self.fine_variable:
            self.variable_name += "_fine"
        self.variable_expr = f"{self.producer_name}_hh"
        self.inference_model_name = f"{self.base_inference_model}_{self.variable_name}"

        # register objects
        self._register_objects()

    @classmethod
    def compute_file_hash(cls, file_path: str) -> str:
        with open(expand_path(file_path), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:8]

    def _register_objects(self) -> None:
        # config dependent objects
        for config_inst in self.config_insts:
            self._register_variable(config_inst)

        # config independent objects
        self._register_producer()
        self._register_inference_model()

    def _register_variable(self, config_inst: od.Config) -> None:
        if self.variable_name in config_inst.variables:
            return

        # aux fields
        aux = {"underflow": True, "overflow": True}
        if self.fine_variable:
            aux |= {"x_transformations": "equal_distance_with_indices"}

        # binning
        if self.fine_variable and self.logit_variable:
            binning = np.linspace(-15.0, 15.0, 2001).tolist()
        elif self.fine_variable:
            binning = np.linspace(0.0, 0.8, 801).tolist() + np.linspace(0.8, 1.0, 1001)[1:].tolist()
        elif self.logit_variable:
            binning = (30, -15, 15)
        else:
            binning = (10, 0.0, 1.0)

        config_inst.add_variable(
            name=self.variable_name,
            expression=self.variable_expr,
            binning=binning,
            null_value=EMPTY_FLOAT,
            aux=aux,
        )

    def _register_producer(self) -> None:
        if Producer.has_cls(self.producer_name):
            return

        base_cls = Producer.get_cls(self.base_producer)
        base_cls.derive(self.producer_name, cls_dict={
            "exposed": True,
            "local_model_path": self.torch_model_path,
        })

    def _register_inference_model(self) -> None:
        if default_inference_model.has_cls(self.inference_model_name):
            return

        base_cls = default_inference_model.get_cls(self.base_inference_model)
        base_cls.derive(self.inference_model_name, cls_dict={
            "variable": self.variable_name,
            "add_qcd": any(hook.startswith("qcd") for hook in self.hist_hooks),
        })

    def requires(self):
        return CreateDatacards.req(
            self,
            inference_model=self.inference_model_name,
            producers=("default", self.producer_name),
        )
