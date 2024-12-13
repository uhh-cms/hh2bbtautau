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


# custom seed producer skipping GenPart fields
custom_deterministic_event_seeds = deterministic_event_seeds.derive(
    "custom_deterministic_event_seeds",
    cls_dict={"object_count_columns": [
        route
        for route in deterministic_event_seeds.object_count_columns
        if not str(route).startswith("GenPart.")
    ]},

)


@calibrator(
    uses={
        mc_weight, custom_deterministic_event_seeds, deterministic_jet_seeds,
    },
    produces={
        mc_weight, custom_deterministic_event_seeds, deterministic_jet_seeds,
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
        events = self[self.jec_nominal_cls](events, **kwargs)
    else:
        events = self[self.jec_full_cls](events, **kwargs)
        events = self[self.deterministic_jer_cls](events, **kwargs)

    if self.config_inst.campaign.x.run == 2:
        events = self[self.met_phi_cls](events, **kwargs)

    if self.dataset_inst.is_mc:
        if self.global_shift_inst.is_nominal:
            events = self[self.tec_cls](events, **kwargs)
        else:
            events = self[self.tec_nominal_cls](events, **kwargs)

    return events


@default.init
def default_init(self: Calibrator) -> None:
    # set the name of the met collection to use
    met_name = self.config_inst.x.met_name
    raw_met_name = self.config_inst.x.raw_met_name

    # derive calibrators to add settings
    self.jec_full_cls = jec.derive("jec_full", cls_dict={
        "mc_only": True,
        "nominal_only": True,
        "met_name": met_name,
        "raw_met_name": raw_met_name,
    })
    self.jec_nominal_cls = jec_nominal.derive("jec_nominal", cls_dict={
        "met_name": met_name,
        "raw_met_name": raw_met_name,
    })

    # version of jer that uses the first random number from deterministic_seeds
    self.deterministic_jer_cls = jer.derive("deterministic_jer", cls_dict={
        "deterministic_seed_index": 0,
        "met_name": met_name,
    })

    # derive tec calibrators
    self.tec_cls = tec.derive("tec", cls_dict={"met_name": met_name})
    self.tec_nominal_cls = tec_nominal.derive("tec_nominal", cls_dict={"met_name": met_name})

    # derive met_phi calibrator (currently only for run 2)
    self.met_phi_cls = met_phi.derive("met_phi", cls_dict={"met_name": met_name})

    # collect derived calibrators and add them to the calibrator uses and produces
    derived_calibrators = {
        self.jec_full_cls,
        self.jec_nominal_cls,
        self.deterministic_jer_cls,
        self.tec_cls,
        self.tec_nominal_cls,
        IF_RUN_2(self.met_phi_cls),
    }
    self.uses |= derived_calibrators
    self.produces |= derived_calibrators
