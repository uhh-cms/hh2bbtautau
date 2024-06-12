# coding: utf-8

"""
Placeholder calibrator to produce missing single trigger columns.
Does not include default calibrations.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")


@calibrator(
    uses={
        "Electron.pt", "Electron.eta", "Muon.pt", "Muon.eta", "Tau.pt",
    },
    produces={
        "HLT_Ele25_eta2p1_WPTight_Gsf", "HLT_IsoMu22", "HLT_IsoMu22_eta2p1", "HLT_IsoTkMu22", "HLT_IsoTkMu22_eta2p1",
        "HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau30", "HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20",
        "HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_SingleL1",
    },
)
def fake_triggers(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:

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

    HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau30 = (
        ak.any(
            (events.Electron.pt >= 24.5) &
            (events.Electron.eta <= 2.1),
            axis=1,
        ) & ak.any(
            (events.Tau.pt >= 30.5),
            axis=1,
        )
    )

    HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20 = (
        ak.any(
            (events.Electron.pt >= 24.5) &
            (events.Electron.eta <= 2.1),
            axis=1,
        ) & ak.any(
            (events.Tau.pt >= 20.5),
            axis=1,
        )
    )

    HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_SingleL1 = (
        ak.any(
            (events.Electron.pt >= 24.5) &
            (events.Electron.eta <= 2.1),
            axis=1,
        ) & ak.any(
            (events.Tau.pt >= 20.5),
            axis=1,
        )
    )

    events = set_ak_column(events, "HLT_Ele25_eta2p1_WPTight_Gsf", HLT_Ele25_eta2p1_WPTight_Gsf)
    events = set_ak_column(events, "HLT_IsoMu22", HLT_IsoMu22)
    events = set_ak_column(events, "HLT_IsoMu22_eta2p1", HLT_IsoMu22_eta2p1)
    events = set_ak_column(events, "HLT_IsoTkMu22", HLT_IsoMu22)
    events = set_ak_column(events, "HLT_IsoTkMu22_eta2p1", HLT_IsoMu22_eta2p1)
    events = set_ak_column(
        events,
        "HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau30",
        HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau30,
    )
    events = set_ak_column(
        events,
        "HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20",
        HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20,
    )
    events = set_ak_column(
        events,
        "HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_SingleL1",
        HLT_Ele24_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_SingleL1,
    )

    return events
