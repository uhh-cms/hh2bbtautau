# coding: utf-8

"""
Calibration methods.
"""

from __future__ import annotations

import law

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.met import met_phi
from columnflow.calibration.cms.jets import jec, jec_nominal, jer
from columnflow.calibration.cms.tau import tec, tec_nominal
from columnflow.calibration.cms.egamma import eer, eec
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.supercluster_eta import electron_sceta
from columnflow.production.cms.seeds import (
    deterministic_event_seeds, deterministic_jet_seeds, deterministic_electron_seeds,
)
from columnflow.util import maybe_import

from hbt.util import IF_RUN_2, IF_RUN_3

ak = maybe_import("awkward")


# custom seed producer skipping GenPart fields
custom_deterministic_event_seeds = deterministic_event_seeds.derive(
    "custom_deterministic_event_seeds",
    cls_dict={"object_count_columns": [
        route
        for route in deterministic_event_seeds.object_count_columns
        if not str(route).startswith(("GenPart.", "Photon."))
    ]},
)


@calibrator(
    uses={
        mc_weight, custom_deterministic_event_seeds, deterministic_jet_seeds, deterministic_electron_seeds,
        electron_sceta,
    },
    produces={
        mc_weight, custom_deterministic_event_seeds, deterministic_jet_seeds, deterministic_electron_seeds,
    },
)
def default(self: Calibrator, events: ak.Array, task: law.Task, **kwargs) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # seed producers
    # !! as this is the first step, the object collections should still be pt-sorted,
    # !! so no manual sorting needed here (but necessary if, e.g., jec is applied before)
    events = self[custom_deterministic_event_seeds](events, **kwargs)
    events = self[deterministic_jet_seeds](events, **kwargs)
    events = self[deterministic_electron_seeds](events, **kwargs)

    events = self[electron_sceta](events, **kwargs)
    if self.dataset_inst.is_data or not task.global_shift_inst.is_nominal:
        events = self[self.jec_nominal_cls](events, **kwargs)
        # egamma scale calibrations should only be applied to data so if the global shift is not
        # nominal, we are in the shifted case and will only execute something if it's data
        if self.dataset_inst.is_data:
            if self.has_dep(self.electron_scale_nominal_cls):
                events = self[self.electron_scale_nominal_cls](events, **kwargs)
        else:
            if self.has_dep(self.electron_res_nominal_cls):
                events = self[self.electron_res_nominal_cls](events, **kwargs)
    else:
        events = self[self.jec_full_cls](events, **kwargs)
        events = self[self.deterministic_jer_cls](events, **kwargs)
        # in this block, we are in the nominal case in MC
        if self.has_dep(self.electron_res_cls):
            events = self[self.electron_res_cls](events, **kwargs)
        if self.has_dep(self.electron_scale_cls):
            events = self[self.electron_scale_cls](events, **kwargs)

    if self.has_dep(self.met_phi_cls):
        events = self[self.met_phi_cls](events, **kwargs)

    if self.dataset_inst.is_mc:
        if task.global_shift_inst.is_nominal:
            events = self[self.tec_cls](events, **kwargs)
        else:
            events = self[self.tec_nominal_cls](events, **kwargs)

    return events


@default.init
def default_init(self: Calibrator, **kwargs) -> None:
    # from traceback import print_stack; print_stack()
    # set the name of the met collection to use
    met_name = self.config_inst.x.met_name
    raw_met_name = self.config_inst.x.raw_met_name

    # derive calibrators to add settings once
    flag = f"custom_calibs_registered_{self.cls_name}"
    if not self.config_inst.x(flag, False):
        # jec calibrators
        self.config_inst.x.calib_jec_full_cls = jec.derive("jec_full", cls_dict={
            "mc_only": True,
            "met_name": met_name,
            "raw_met_name": raw_met_name,
        })
        self.config_inst.x.calib_jec_nominal_cls = jec_nominal.derive("jec_nominal", cls_dict={
            "met_name": met_name,
            "raw_met_name": raw_met_name,
        })
        # version of jer that uses the first random number from deterministic_seeds
        self.config_inst.x.calib_deterministic_jer_cls = jer.derive("deterministic_jer", cls_dict={
            "deterministic_seed_index": 0,
            "met_name": met_name,
        })
        # derive tec calibrators
        self.config_inst.x.calib_jec_cls = tec.derive("tec", cls_dict={
            "met_name": met_name,
        })
        self.config_inst.x.calib_jec_cls = tec_nominal.derive("tec_nominal", cls_dict={
            "met_name": met_name,
        })
        # derive met_phi calibrator (currently only used in run 2)
        self.config_inst.x.calib_met_phi_cls = met_phi.derive("met_phi", cls_dict={
            "met_name": met_name,
        })

        # derive electron scale calibrators
        self.config_inst.x.calib_electron_scale_cls = eec.derive("eec_full", cls_dict={
        })

        self.config_inst.x.calib_electron_scale_nominal_cls = eec.derive("eec_nominal", cls_dict={
            "with_uncertainties": False,
        })

        # derive electron resolution calibrator
        self.config_inst.x.calib_electron_res_cls = eer.derive("eer_full", cls_dict={
            "deterministic_seed_index": 0,
        })

        self.config_inst.x.calib_electron_res_nominal_cls = eer.derive("eer_nominal", cls_dict={
            "deterministic_seed_index": 0,
            "with_uncertainties": False,
        })

        # change the flag
        self.config_inst.set_aux(flag, True)

    self.jec_full_cls = self.config_inst.x.calib_jec_full_cls
    self.jec_nominal_cls = self.config_inst.x.calib_jec_nominal_cls
    self.deterministic_jer_cls = self.config_inst.x.calib_deterministic_jer_cls
    self.tec_cls = self.config_inst.x.calib_jec_cls
    self.tec_nominal_cls = self.config_inst.x.calib_jec_cls
    self.met_phi_cls = self.config_inst.x.calib_met_phi_cls
    self.electron_scale_cls = self.config_inst.x.calib_electron_scale_cls
    self.electron_scale_nominal_cls = self.config_inst.x.calib_electron_scale_nominal_cls
    self.electron_res_cls = self.config_inst.x.calib_electron_res_cls
    self.electron_res_nominal_cls = self.config_inst.x.calib_electron_res_nominal_cls

    # collect derived calibrators and add them to the calibrator uses and produces
    derived_calibrators = {
        self.jec_full_cls,
        self.jec_nominal_cls,
        self.deterministic_jer_cls,
        self.tec_cls,
        self.tec_nominal_cls,
        IF_RUN_2(self.met_phi_cls),
        IF_RUN_3(self.electron_scale_cls),
        IF_RUN_3(self.electron_scale_nominal_cls),
        IF_RUN_3(self.electron_res_cls),
        IF_RUN_3(self.electron_res_nominal_cls),
    }
    self.uses |= derived_calibrators
    self.produces |= derived_calibrators
