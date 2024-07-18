# coding: utf-8

"""
Calibration methods.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.met import met_phi
from columnflow.calibration.cms.jets import jec, jec_nominal, jer
from columnflow.calibration.cms.tau import tec, tec_nominal
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import

from hbt.util import IF_RUN_2

ak = maybe_import("awkward")


# derive calibrators to add settings
jec_full = jec.derive("jec_full", cls_dict={"mc_only": True, "nominal_only": True})


@calibrator(
    uses={
        mc_weight, jec_nominal, jec_full, jer, tec_nominal, tec, deterministic_seeds,
        IF_RUN_2(met_phi),
    },
    produces={
        mc_weight, jec_nominal, jec_full, jer, tec_nominal, tec, deterministic_seeds,
        IF_RUN_2(met_phi),
    },
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    events = self[deterministic_seeds](events, **kwargs)
    if self.dataset_inst.is_data or not self.global_shift_inst.is_nominal:
        events = self[jec_nominal](events, **kwargs)
    else:
        events = self[jec_full](events, **kwargs)
        events = self[jer](events, **kwargs)

    if self.config_inst.campaign.x.run == 2:
        events = self[met_phi](events, **kwargs)

    if self.dataset_inst.is_mc:
        if self.global_shift_inst.is_nominal:
            events = self[tec](events, **kwargs)
        else:
            events = self[tec_nominal](events, **kwargs)

    return events
