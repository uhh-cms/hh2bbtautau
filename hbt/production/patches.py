# coding: utf-8

"""
Producers creating columns that make up for broken values in NanoAOD or other upstream methods.
"""

from __future__ import annotations

from columnflow.production import Producer, producer
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import

ak = maybe_import("awkward")


@producer(
    uses={"run", "PuppiMET.{pt,phi}", "Jet.{pt,eta,phi,neEmEF,chEmEF}"},
    produces={"patchedEcalBadCalibFilter"},
)
def patch_ecalBadCalibFilter(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Patch for the broken "Flag_ecalBadCalibFilter" MET filter in prompt (not re-reco-ed) 2022 data.
    As for all MET filters, the output is a boolean that is true if the event passes the filter.

    Resources:
    - Discussion: https://cms-talk.web.cern.ch/t/noise-met-filters-in-run-3/63346/5
    - TWiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2?rev=172#ECal_BadCalibration_Filter_Flag
    """  # noqa
    reject = (
        (events.run >= 362433) & (events.run <= 367144) &
        (events.PuppiMET.pt > 100) &
        ak.any(
            (events.Jet.pt > 50) &
            (events.Jet.eta >= -0.5) & (events.Jet.eta <= -0.1) &
            (events.Jet.phi >= -2.1) & (events.Jet.phi <= -1.8) &
            ((events.Jet.neEmEF > 0.9) | (events.Jet.chEmEF > 0.9)) &
            (events.Jet.delta_phi(events.PuppiMET) > 2.9),
            axis=1,
        )
    )
    return set_ak_column(events, "patchedEcalBadCalibFilter", ~reject)


@patch_ecalBadCalibFilter.init
def patch_ecalBadCalibFilter_init(self: Producer, **kwargs) -> None:
    # only meant to be configured to run on data
    if not self.dataset_inst.is_data:
        raise Exception(f"{self} is only meant to run on data, but running on dataset {self.dataset_inst}")
