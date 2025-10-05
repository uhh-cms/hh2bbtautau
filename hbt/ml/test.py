# coding: utf-8

"""
Test model definition.
"""

from __future__ import annotations

from typing import Any

import law
import order as od

from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import Route, set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")

law.contrib.load("tensorflow")


class TestModel(MLModel):

    def setup(self):
        # dynamically add variables for the quantities produced by this model
        if f"{self.cls_name}.kl" not in self.config_inst.variables:
            self.config_inst.add_variable(
                name=f"{self.cls_name}_kl",
                null_value=-1,
                binning=(20, -10, 10),
                x_title="Predicted kappa-lambda",
            )

    def sandbox(self, task: law.Task) -> str:
        return dev_sandbox("bash::$HBT_BASE/sandboxes/venv_hbt.sh")

    def datasets(self, config_inst: od.Config) -> set[od.Dataset]:
        return {
            config_inst.get_dataset("hh_ggf_hbb_htt_kl1_kt1_powheg"),
            config_inst.get_dataset("hh_ggf_hbb_htt_kl0_kt1_powheg"),
        }

    def uses(self, config_inst: od.Config) -> set[Route | str]:
        return {"Jet.{pt,eta,phi}"}

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        return {f"{self.cls_name}.kl"}

    def training_calibrators(
        self,
        config_inst: od.Config,
        requested_calibrators: list[str],
    ) -> list[str]:
        return ["default"]

    def training_selectors(
        self,
        config_inst: od.Config,
        requested_selectors: list[str],
    ) -> list[str]:
        return ["default"]

    def training_producers(
        self,
        config_inst: od.Config,
        requested_producers: list[str],
    ) -> list[str]:
        return ["default"]

    def output(self, task: law.Task) -> law.FileSystemDirectoryTarget:
        return task.target(f"mlmodel_f{task.fold}of{self.folds}.keras")

    def open_model(self, target: law.FileSystemDirectoryTarget):
        return target.load(formatter="tf_keras_model")

    def train(
        self,
        task: law.Task,
        input: dict[str, list[law.FileSystemFileTarget]],
        output: law.FileSystemDirectoryTarget,
    ) -> None:
        tf = maybe_import("tensorflow")

        # define a dummy NN
        x = tf.keras.Input(shape=(2,))
        a1 = tf.keras.layers.Dense(10, activation="elu")(x)
        y = tf.keras.layers.Dense(2, activation="softmax")(a1)
        model = tf.keras.Model(inputs=x, outputs=y)

        # the output is just a single directory target
        output.dump(model, formatter="tf_keras_model")

    def evaluate(
        self,
        task: law.Task,
        events: ak.Array,
        models: list[Any],
        fold_indices: ak.Array,
        events_used_in_training: bool = False,
    ) -> ak.Array:
        # fake evaluation
        events = set_ak_column(events, f"{self.cls_name}.kl", 0.5, value_type=np.float32)

        return events


# usable derivations
test_model = TestModel.derive("test_model", cls_dict={"folds": 2})
