# coding: utf-8

"""
Calibration methods.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.met import met_phi
from columnflow.calibration.cms.jets import jec, jec_nominal, jer
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import

from hbt.calibration.tau import tec


ak = maybe_import("awkward")


@calibrator(
    uses={deterministic_seeds, met_phi},
    produces={deterministic_seeds, met_phi},
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    events = self[deterministic_seeds](events, **kwargs)

    if self.dataset_inst.is_data:
        events = self[jec_nominal](events, **kwargs)
    else:
        events = self[jec](events, **kwargs)
        events = self[jer](events, **kwargs)

    events = self[met_phi](events, **kwargs)

    if self.dataset_inst.is_mc:
        events = self[tec](events, **kwargs)

    return events


@default.init
def default_init(self: Calibrator) -> None:
    if not getattr(self, "dataset_inst", None):
        return

    if self.dataset_inst.is_data:
        calibrators = {jec_nominal}
    else:
        calibrators = {mc_weight, jec, jer, tec}

    self.uses |= calibrators
    self.produces |= calibrators
