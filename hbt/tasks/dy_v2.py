# coding: utf-8

"""
Tasks to create correction_lib file for scale factor calculation for DY events.
"""

from __future__ import annotations

import re
import gzip

import luigi
import law
import order as od
import numpy as np
import awkward as ak

from columnflow.tasks.framework.base import ConfigTask, TaskShifts
from columnflow.tasks.framework.mixins import (
    DatasetsProcessesMixin, ProducerClassesMixin, CalibratorClassesMixin,
    SelectorClassMixin, ReducerClassMixin,
    )
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.reduction import ProvideReducedEvents
from columnflow.util import maybe_import
from columnflow.columnar_util import (
    ChunkedIOHandler, RouteFilter, update_ak_array, attach_coffea_behavior, layout_ak_array,
    set_ak_column,
)

from hbt.tasks.base import HBTTask

hist = maybe_import("hist")


class DYWeights(
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

    reload = luigi.BoolParameter(default=False, significant=False)

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

        self.dilep_pt_variable_inst = self.config_inst.variables.n.dilep_pt
        self.read_columns = [
            "Jet.btagPNetB",
            "channel_id",
            "category_ids",
            "process_id",
            "Electron.pt",
            "Electron.eta",
            "Electron.phi",
            "Electron.mass",
            "Muon.mass",
            "Muon.phi",
            "Muon.eta",
            "Muon.pt",
            "Tau.mass",
            "Tau.phi",
            "Tau.eta",
            "Tau.pt",
        ]
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
        self.write_columns = [
            "channel_id",
            "category_ids",
            "process_id",
            "dilep_pt",
            "weight",
            "njets",
            "nbjets",
        ]

        cat_dyc = self.config_inst.get_category("dyc")
        cat_os = self.config_inst.get_category("os")
        self.category_ids = [
            cat.id for cat in cat_dyc.get_leaf_categories()
            if cat_os.has_category(cat, deep=True)
        ]

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts

    def output(self):
        outputs = {
            "plots": [],
            "json": self.target("corrections.json.gz"),
            "data": self.target("data.pkl", optional=True),
        }

        return outputs

    def requires(self):
        if not self.reload and self.output()["data"].exists():
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
        if not self.reload and outputs["data"].exists():
            data_events, bkg_events, dy_events = outputs["data"].load(formatter="pickle")
        else:
            data_events, bkg_events, dy_events = self.load_data()
            outputs["data"].dump((data_events, bkg_events, dy_events), formatter="pickle")

        import correctionlib.schemav2 as cs
        from hbt.studies.dy_weights.create_clib_file import create_dy_weight_correction

    def load_data(self):
        # prepare columns to write
        route_filter = RouteFilter(keep=self.write_columns)

        data_events = []
        bkg_events = []
        dy_events = []

        # loop over datasets and load inputs
        for dataset_name, inps in self.input().items():
            self.publish_message(f"Loading dataset '{dataset_name}'")
            # define columns to read
            read_columns = [
                *self.read_columns,
                *self.event_weight_columns,
            ]
            dataset_weight_columns = []
            for pattern, cols in self.dataset_event_weight_columns.items():
                if law.util.multi_match(dataset_name, pattern):
                    dataset_weight_columns += cols
            read_columns += dataset_weight_columns

            # loop over each file per input
            col = inps["reduction"].collection
            for i in range(len(col)):
                targets = [col.targets[i]["events"]]
                for prod in self.producers:
                    targets.append(inps["production"][prod].collection.targets[i]["columns"])

                # prepare inputs for localization
                with law.localize_file_targets(targets, mode="r") as local_inps:
                    reader = ChunkedIOHandler(
                        [t.abspath for t in local_inps],
                        source_type=len(targets) * ["awkward_parquet"],
                        read_columns=len(targets) * [read_columns],
                        chunk_size=50_000,
                    )
                    for (events, *columns), pos in reader:
                        events = update_ak_array(events, *columns)
                        events = attach_coffea_behavior(events)

                        # filter events for DY weight derivation
                        cat_mask = np.isin(ak.flatten(events.category_ids), self.category_ids)
                        cat_mask = layout_ak_array(cat_mask, events.category_ids)
                        event_mask = (
                            ak.any(cat_mask, axis=1) &
                            (
                                (events.channel_id == self.config_inst.channels.n.ee.id) |
                                (events.channel_id == self.config_inst.channels.n.mumu.id)
                            )
                        )
                        events = events[event_mask]

                        # compute additional columns
                        events = set_ak_column(
                            events,
                            "njets",
                            ak.num(events.Jet, axis=1),
                            value_type=np.int32,
                        )

                        wp_value = self.config_inst.x.btag_working_points["particleNet"]["medium"]
                        bjet_mask = events.Jet["btagPNetB"] >= wp_value
                        events = set_ak_column(
                            events,
                            "nbjets",
                            ak.sum(bjet_mask, axis=1),
                            value_type=np.int32,
                        )

                        events = set_ak_column(
                            events,
                            "dilep_pt",
                            self.dilep_pt_variable_inst.expression(events),
                        )

                        weight = np.ones(len(events), dtype=np.float32)
                        for col in self.event_weight_columns + dataset_weight_columns:
                            weight = weight * events[col]
                        events = set_ak_column(events, "weight", weight)

                        # filter columns to read at the end
                        events = route_filter(events)
                        events.behavior = None

                        # save events by dataset type
                        if dataset_name.startswith("dy_"):
                            dy_events.append(events)
                        elif dataset_name.startswith("data_"):
                            data_events.append(events)
                        else:
                            bkg_events.append(events)

        data_events = ak.concatenate(data_events, axis=0) if data_events else None
        bkg_events = ak.concatenate(bkg_events, axis=0) if bkg_events else None
        dy_events = ak.concatenate(dy_events, axis=0) if dy_events else None

        return data_events, bkg_events, dy_events
