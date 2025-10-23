# coding: utf-8

"""
Tasks to create correction_lib file for scale factor calculation for DY events.
"""

from __future__ import annotations

import re
import gzip

import law
import order as od

from columnflow.tasks.framework.base import ConfigTask, TaskShifts
from columnflow.tasks.framework.mixins import DatasetsProcessesMixin, ProducerClassesMixin, CalibratorClassesMixin, SelectorClassMixin, ReducerClassMixin
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.reduction import ProvideReducedEvents
from columnflow.util import maybe_import

from hbt.tasks.base import HBTTask

hist = maybe_import("hist")


class SomeDYName(
    HBTTask,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ReducerClassMixin,
    ProducerClassesMixin,
    DatasetsProcessesMixin
):
    """
    some description
    """

    single_config = True
    allow_empty_processes = True

    @classmethod
    def modify_param_values(cls, params):
        params = super().modify_param_values(params)
        params["processes"] = tuple()
        return params

    @classmethod
    def resolve_param_values(cls, params):
        params["known_shifts"] = TaskShifts()
        return super().resolve_param_values(params)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.pt_ll_variable_inst = self.config_inst.variables.n.dilep_pt
        self.event_weight_columns = [
            "normalization_weight",
            "normalized_pdf_weight",
            "normalized_murmuf_weight",
            "normalized_pu_weight",
            "normalized_isr_weight",
            "normalized_fsr_weight",
            "normalized_njet_btag_weight_pnet",
            "electron_weight",
            "muon_weight",
            "tau_weight",
            "trigger_weight",
        ]

        self.dataset_event_weight_columns = {
            "tt_*": ["top_pt_weight"],
        }

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts

    def output(self):

        my_dict = {
            "plots": [],
            "json": self.target("corrections.json.gz"),
            "data": self.target("data.pkl", optional=True),
        }

        return my_dict

    def requires(self):
        if self.output()["data"].exists():
            return []
        reqs = {}
        for dataset in self.datasets:
            reqs[dataset] = {
                "reduction": ProvideReducedEvents.req(self, dataset=dataset),
                "production": {
                    prod: ProduceColumns.req(self, dataset=dataset, producer=prod)
                    for prod in self.producers
                },
            }

        return reqs

    def run(self):
        outputs = self.output()

        # read data, potentially from cache
        if outputs["data"].exists():
            data = outputs["data"].load(formatter="pickle")
        else:
            data = self.load_data()
            outputs["data"].dump(data, formatter="pickle")

        # fit here

        import correctionlib.schemav2 as cs
        from hbt.studies.dy_weights.create_clib_file import create_dy_weight_correction

    def load_data(self):
        # work here :)
        pass
