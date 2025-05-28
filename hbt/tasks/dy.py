# coding: utf-8

"""
Tasks to study and create weights for DY events.
"""

from __future__ import annotations

import re
import gzip

import law

from columnflow.tasks.framework.base import ConfigTask
from columnflow.tasks.framework.histograms import HistogramsUserSingleShiftBase
from columnflow.util import maybe_import, dev_sandbox

from hbt.tasks.base import HBTTask

hist = maybe_import("hist")


class ComuteDYWeights(HBTTask, HistogramsUserSingleShiftBase):
    """
    Example command:

        > law run hbt.ComuteDYWeights \
            --config 22pre_v14 \
            --datasets tt_sl_powheg \
            --version prod8_dy \
            --hist-producer no_dy_weight \
            --categories mumu__dy__os \
            --variables njets-dilep_pt
    """

    single_config = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # only one category is allowed right now
        if len(self.categories) != 1:
            raise ValueError(f"{self.task_family} requires exactly one category, got {self.categories}")
        # ensure that the category matches a specific pattern: starting with "ee"/"mumu" and ending in "os"
        if not re.match(r"^(ee|mumu)__.*__os$", self.categories[0]):
            raise ValueError(f"category must start with '{{ee,mumu}}__' and end in '__os', got {self.categories[0]}")
        self.category_inst = self.config_inst.get_category(self.categories[0])

        # only one variable is allowed
        if len(self.variables) != 1:
            raise ValueError(f"{self.task_family} requires exactly one variable, got {self.variables}")
        self.variable = self.variables[0]
        # for now, variable must be "njets-dilep_pt"
        if self.variable != "njets-dilep_pt":
            raise ValueError(f"variable must be 'njets-dilep_pt', got {self.variable}")

    def output(self):
        return self.target(f"dy_weight_data__{self.category_inst.name}.pkl")

    def run(self):
        # prepare categories to sum over
        leaf_category_insts = self.category_inst.get_leaf_categories() or [self.category_inst]

        # load histograms from all input datasets and add them
        h = None
        for dataset_name, inp in self.input().items():
            h_ = inp.collection[0]["hists"][self.variable].load(formatter="pickle")

            # select and sum over leaf categories
            h_ = h_[{"category": [hist.loc(cat.name) for cat in leaf_category_insts]}]
            h_ = h_[{"category": sum}]

            # use the nominal shift only
            h_ = h_[{"shift": hist.loc("nominal")}]

            # add it
            h = h_ if h is None else (h + h_)

        # compute the dy weight data
        dy_weight_data = compute_weight_data(self, h)

        # store them
        self.output().dump(dy_weight_data, formatter="pickle")


class ExportDYWeights(HBTTask, ConfigTask):
    """
    Example command:

        > law run hbt.ExportDYWeights \
            --configs 22pre_v14,22post_v14,... \
            --version prod8_dy
    """

    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar.sh")

    single_config = False

    def requires(self):
        return {
            config: ComuteDYWeights.req(
                self,
                config=config,
                datasets=("tt_sl_powheg", "tt_dl_powheg"),  # supports groups
                hist_producer="no_dy_weight",
                categories=("mumu__dy__os",),
                variables=("njets-dilep_pt",),
            )
            for config in self.configs
        }

    def output(self):
        return self.target("hbt_corrections.json.gz")

    def run(self):
        import correctionlib.schemav2 as cs
        from hbt.studies.dy_weights.create_clib_file import create_dy_weight_correction

        # load all weight data per config and merge them into a single dictionary
        dy_weight_data = law.util.merge_dicts(*(
            inp.load(formatter="pickle") for inp in self.input().values()
        ))

        # create and save the correction set
        cset = cs.CorrectionSet(
            schema_version=2,
            description="Corrections derived for the hh2bbtautau analysis.",
            corrections=[create_dy_weight_correction(dy_weight_data)],
        )
        with self.output().localize("w") as outp:
            outp.path += ".json.gz"
            with gzip.open(outp.abspath, "wt") as f:
                f.write(cset.json(exclude_unset=True))

            # validate the content
            law.util.interruptable_popen(f"correction summary {outp.abspath}", shell=True)


def compute_weight_data(task: ComuteDYWeights, h: hist.Hist) -> dict:
    """
    Compute the DY weight data from the given histogram *h* that supposed to contain the following axis:

        - process (str)
        - njets (int)
        - dilep_pt (float)

    The returned dictionary follows a nested structure:

        year -> syst -> (min_njet, max_njet) -> [(lower_bound, upper_bound, formula), ...]

        - *year* is one of 2022, 2022EE, 2023, 2023BPix
        - *syst* is one of nominal, up, down (maybe up1, down1, up2, down2, ... in case of multiple sources)
        - *(min_njet, max_njet)* is a tuple of integers defining the right-exclusive range of njets
        - the inner-most list contains 3-tuples with lower and upper bounds of a formula
    """
    # prepare constants
    inf = float("inf")
    era = f"{task.config_inst.campaign.x.year}{task.config_inst.campaign.x.postfix}"

    # dummy values, but this should be where the magic happens
    return {
        era: {
            "nominal": {
                (0, 1): [
                    (0.0, 50.0, "1.0"),
                    (50, inf, "1.1"),
                ],
                (1, 11): [
                    (0.0, 50.0, "1.0"),
                    (50.0, inf, "1.1"),
                ],
            },
            "up": {
                (0, 1): [
                    (0.0, inf, "1.05"),
                ],
                (1, 11): [
                    (0.0, inf, "1.05"),
                ],
            },
            "down": {
                (0, 1): [
                    (0.0, inf, "0.95"),
                ],
                (1, 11): [
                    (0.0, inf, "0.95"),
                ],
            },
        },
    }
