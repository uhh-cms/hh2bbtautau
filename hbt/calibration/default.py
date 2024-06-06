# coding: utf-8

"""
Calibration methods.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.met import met_phi
from columnflow.calibration.cms.jets import jec, jer
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import
from hbt.calibration.tau import tec


ak = maybe_import("awkward")
np = maybe_import("numpy")

# derive calibrators to add settings
jec_nominal = jec.derive("jec_nominal", cls_dict={"uncertainty_sources": [], "data_only": True})
jec_full = jec.derive("jec_nominal", cls_dict={"mc_only": True})


@calibrator(
    uses={
        mc_weight, jec_nominal, jec_full, jer, tec, deterministic_seeds, met_phi,
    },
    produces={
        mc_weight, jec_nominal, jec_full, jer, tec, deterministic_seeds, met_phi,
    },
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    events = self[deterministic_seeds](events, **kwargs)

    if self.dataset_inst.is_data:
        events = self[jec_nominal](events, **kwargs)
    else:
        events = self[jec_full](events, **kwargs)
        events = self[jer](events, **kwargs)

    events = self[met_phi](events, **kwargs)

    if self.dataset_inst.is_mc:
        events = self[tec](events, **kwargs)

    return events


@calibrator(
    uses=set(),
    produces=set(),
)
def empty(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    return events
