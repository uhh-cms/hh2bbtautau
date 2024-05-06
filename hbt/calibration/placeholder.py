# coding: utf-8

"""
Placeholder calibrator to produce missing single trigger columns. Includes all default calibrations.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")

@calibrator(
    uses={
        mc_weight, deterministic_seeds,
    },
    produces={
        mc_weight, deterministic_seeds, "HLT_Ele25_eta2p1_WPTight_Gsf",
        "HLT_IsoMu22", "HLT_IsoMu22_eta2p1", "HLT_IsoTkMu22", "HLT_IsoTkMu22_eta2p1",
    },
)
def placeholder(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    events = self[deterministic_seeds](events, **kwargs)

    HLT_Ele25_eta2p1_WPTight_Gsf = ak.any(
        (events.Electron.pt >= 25.5) &
        (events.Electron.eta <= 2.1),
        axis=1,
    )

    HLT_IsoMu22 = ak.any(
        (events.Muon.pt >= 22.5),
        axis=1,
    )

    HLT_IsoMu22_eta2p1 = ak.any(
        (events.Muon.pt >= 22.5) &
        (events.Muon.eta <= 2.1),
        axis=1,
    )

    events = set_ak_column(events, "HLT_Ele25_eta2p1_WPTight_Gsf", HLT_Ele25_eta2p1_WPTight_Gsf)
    events = set_ak_column(events, "HLT_IsoMu22", HLT_IsoMu22)
    events = set_ak_column(events, "HLT_IsoMu22_eta2p1", HLT_IsoMu22_eta2p1)
    events = set_ak_column(events, "HLT_IsoTkMu22", HLT_IsoMu22)
    events = set_ak_column(events, "HLT_IsoTkMu22_eta2p1", HLT_IsoMu22_eta2p1)

    return events

