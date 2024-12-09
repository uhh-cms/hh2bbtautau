# coding: utf-8

"""
Calibration methods.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.met import met_phi
from columnflow.calibration.cms.jets import jec, jec_nominal, jer
from columnflow.calibration.cms.tau import tec, tec_nominal
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.seeds import deterministic_event_seeds, deterministic_jet_seeds
from columnflow.util import maybe_import

from hbt.util import IF_RUN_2

ak = maybe_import("awkward")


# derive calibrators to add settings
jec_full = jec.derive("jec_full", cls_dict={"mc_only": True, "nominal_only": True})

# custom seed producer skipping GenPart fields
custom_deterministic_event_seeds = deterministic_event_seeds.derive(
    "custom_deterministic_event_seeds",
    cls_dict={"object_count_columns": [
        route
        for route in deterministic_event_seeds.object_count_columns
        if not str(route).startswith("GenPart.")
    ]},

)
# version of jer that uses the first random number from deterministic_seeds
deterministic_jer = jer.derive("deterministic_jer", cls_dict={"deterministic_seed_index": 0})


@calibrator(
    uses={
        mc_weight, custom_deterministic_event_seeds, deterministic_jet_seeds, jec_nominal, jec_full,
        deterministic_jer, tec_nominal, tec, IF_RUN_2(met_phi),
    },
    produces={
        mc_weight, custom_deterministic_event_seeds, deterministic_jet_seeds, jec_nominal, jec_full,
        deterministic_jer, tec_nominal, tec, IF_RUN_2(met_phi),
    },
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # seed producers
    # !! as this is the first step, the object collections should still be pt-sorted,
    # !! so no manual sorting needed here (but necessary if, e.g., jec is applied before)
    events = self[custom_deterministic_event_seeds](events, **kwargs)
    events = self[deterministic_jet_seeds](events, **kwargs)

    if self.dataset_inst.is_data or not self.global_shift_inst.is_nominal:
        events = self[jec_nominal](events, **kwargs)
    else:
        events = self[jec_full](events, **kwargs)
        events = self[deterministic_jer](events, **kwargs)

    if self.config_inst.campaign.x.run == 2:
        events = self[met_phi](events, **kwargs)

    if self.dataset_inst.is_mc:
        if self.global_shift_inst.is_nominal:
            events = self[tec](events, **kwargs)
        else:
            events = self[tec_nominal](events, **kwargs)

    return events
