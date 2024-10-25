# coding: utf-8

"""
Tasks that create files for synchronization efforts with other frameworks.
"""

from __future__ import annotations

import law

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    ProducersMixin, MLModelsMixin, ChunkedIOMixin, SelectorMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.reduction import ReducedEventsUser
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.ml import MLEvaluation
from columnflow.util import dev_sandbox

from hbt.tasks.base import HBTTask


class CreateSyncFiles(
    HBTTask,
    MLModelsMixin,
    ProducersMixin,
    ChunkedIOMixin,
    ReducedEventsUser,
    SelectorMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        ReducedEventsUser.reqs,
        RemoteWorkflow.reqs,
        ProduceColumns=ProduceColumns,
        MLEvaluation=MLEvaluation,
    )

    def workflow_requires(self):
        reqs = super().workflow_requires()

        # require the full merge forest
        reqs["events"] = self.reqs.ProvideReducedEvents.req(self)

        if not self.pilot:
            if self.producer_insts:
                reqs["producers"] = [
                    self.reqs.ProduceColumns.req(self, producer=producer_inst.cls_name)
                    for producer_inst in self.producer_insts
                    if producer_inst.produced_columns
                ]
            if self.ml_model_insts:
                reqs["ml"] = [
                    self.reqs.MLEvaluation.req(self, ml_model=m)
                    for m in self.ml_models
                ]

        return reqs

    def requires(self):
        reqs = {"events": self.reqs.ProvideReducedEvents.req(self)}

        if self.producer_insts:
            reqs["producers"] = [
                self.reqs.ProduceColumns.req(self, producer=producer_inst.cls_name)
                for producer_inst in self.producer_insts
                if producer_inst.produced_columns
            ]
        if self.ml_model_insts:
            reqs["ml"] = [
                self.reqs.MLEvaluation.req(self, ml_model=m)
                for m in self.ml_models
            ]

        return reqs

    workflow_condition = ReducedEventsUser.workflow_condition.copy()

    @workflow_condition.output
    def output(self):
        return self.target(f"sync_{self.dataset_inst.name}_{self.branch}.csv")

    @law.decorator.log
    @law.decorator.localize
    def run(self):
        import awkward as ak
        from columnflow.columnar_util import update_ak_array, EMPTY_FLOAT, set_ak_column

        # prepare inputs and outputs
        inputs = self.input()
        output = self.output()

        # iterate over chunks of events and diffs
        files = [inputs["events"]["events"].abspath]
        if self.producer_insts:
            files.extend([inp["columns"].abspath for inp in inputs["producers"]])
        if self.ml_model_insts:
            files.extend([inp["mlcolumns"].abspath for inp in inputs["ml"]])

        # helper to replace our internal empty float placeholder with a custom one
        empty_float = EMPTY_FLOAT  # points to same value for now, but can be changed
        def replace_empty_float(arr):
            if empty_float != EMPTY_FLOAT:
                arr = ak.where(arr == EMPTY_FLOAT, empty_float, arr)
            return arr

        # helper to pad nested fields with an empty float if missing
        def pad_nested(arr, n, *, axis=1):
            return ak.fill_none(ak.pad_none(arr, n, axis=axis), empty_float)

        # helper to pad and select the last element on the first inner axis
        def select(arr, idx):
            return replace_empty_float(pad_nested(arr, idx + 1, axis=1)[:, idx])

        # helper to select leptons
        def select_leptons(events: ak.Array, common_fields: dict[str, int | float]) -> ak.Array:
            # ensure all lepton arrays have the same common fields
            leptons = [events.Electron, events.Muon, events.Tau]
            for i in range(len(leptons)):
                lepton = leptons[i]
                for field, default in common_fields.items():
                    if field not in lepton.fields:
                        lepton = set_ak_column(lepton, field, default)
                leptons[i] = lepton
            # concatenate (first event any lepton, second alsways tau) and add to events
            return set_ak_column(events, "Lepton", ak.concatenate(leptons, axis=1))

        # event chunk loop
        for (events, *columns), pos in self.iter_chunked_io(
            files,
            source_type=len(files) * ["awkward_parquet"],
            pool_size=1,
        ):
            # optional check for overlapping inputs
            if self.check_overlapping_inputs:
                self.raise_if_overlapping([events] + list(columns))

            # add additional columns
            events = update_ak_array(events, *columns)

            # insert leptons
            events = select_leptons(events, {"rawDeepTau2018v2p5VSjet": empty_float})

            # project into dataframe
            df = ak.to_dataframe({
                # index variables
                "event": events.event,
                "run": events.run,
                "lumi": events.luminosityBlock,
                # high-level events variables
                "channel": events.channel_id,
                "os": events.leptons_os * 1,
                "iso": events.tau2_isolated * 1,
                # jet variables
                "jet1_pt": select(events.Jet.pt, 0),
                "jet1_eta": select(events.Jet.eta, 0),
                "jet1_phi": select(events.Jet.phi, 0),
                "jet2_pt": select(events.Jet.pt, 1),
                "jet2_eta": select(events.Jet.eta, 1),
                "jet2_phi": select(events.Jet.phi, 1),
                "lep1_pt": select(events.Lepton.pt, 0),
                "lep1_phi": select(events.Lepton.phi, 0),
                "lep1_eta": select(events.Lepton.eta, 0),
                "lep1_charge": select(events.Lepton.charge, 0),
                "lep1_deeptauvsjet": select(events.Lepton.rawDeepTau2018v2p5VSjet, 0),
                "lep2_pt": select(events.Lepton.pt, 1),
                "lep2_phi": select(events.Lepton.phi, 1),
                "lep2_eta": select(events.Lepton.eta, 1),
                "lep2_charge": select(events.Lepton.charge, 1),
                "lep2_deeptauvsjet": select(events.Lepton.rawDeepTau2018v2p5VSjet, 1),
                # TODO: add additional variables
            })

            # save as csv in output, append if necessary
            output.dump(
                df,
                formatter="pandas",
                index=False,
                header=pos.index == 0,
                mode="w" if pos.index == 0 else "a",
            )


check_overlap_tasks = law.config.get_expanded("analysis", "check_overlapping_inputs", [], split_csv=True)
CreateSyncFiles.check_overlapping_inputs = ChunkedIOMixin.check_overlapping_inputs.copy(
    default=CreateSyncFiles.task_family in check_overlap_tasks,
    add_default_to_description=True,
)
