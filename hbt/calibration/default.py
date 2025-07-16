# coding: utf-8

"""
Calibration methods.
"""

from __future__ import annotations

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.met import met_phi
from columnflow.calibration.cms.jets import jec, jer
from columnflow.calibration.cms.tau import tec
from columnflow.calibration.cms.egamma import eer, eec
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.electron import electron_sceta
from columnflow.production.cms.seeds import (
    deterministic_event_seeds, deterministic_jet_seeds, deterministic_electron_seeds,
)
from columnflow.util import maybe_import

from hbt.util import IF_RUN_2, IF_RUN_3, IF_DATA, IF_MC

ak = maybe_import("awkward")


# custom seed producer skipping GenPart fields
custom_deterministic_event_seeds_mc = deterministic_event_seeds.derive(
    "custom_deterministic_event_seeds_mc",
    cls_dict={
        "object_count_columns": [
            route for route in deterministic_event_seeds.object_count_columns
            if not str(route).startswith(("GenPart.", "Photon."))
        ],
    },
)
custom_deterministic_event_seeds_data = custom_deterministic_event_seeds_mc.derive(
    "custom_deterministic_event_seeds_data",
    cls_dict={
        "event_columns": [
            route for route in custom_deterministic_event_seeds_mc.event_columns
            if not str(route).startswith("Pileup.nPU")
        ],
        "object_count_columns": [
            route for route in custom_deterministic_event_seeds_mc.object_count_columns
            if not str(route).startswith("GenJet.")
        ],
    },
)


@calibrator(
    uses={
        IF_MC(mc_weight), IF_MC(custom_deterministic_event_seeds_mc), IF_DATA(custom_deterministic_event_seeds_data),
        deterministic_jet_seeds, deterministic_electron_seeds, electron_sceta,
    },
    produces={
        IF_MC(mc_weight), IF_MC(custom_deterministic_event_seeds_mc), IF_DATA(custom_deterministic_event_seeds_data),
        deterministic_jet_seeds, deterministic_electron_seeds,
    },
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    task = kwargs["task"]

    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # seed producers
    # !! as this is the first step, the object collections should still be pt-sorted,
    # !! so no manual sorting needed here (but necessary if, e.g., jec is applied before)
    if self.dataset_inst.is_mc:
        events = self[custom_deterministic_event_seeds_mc](events, **kwargs)
    else:
        events = self[custom_deterministic_event_seeds_data](events, **kwargs)
    events = self[deterministic_jet_seeds](events, **kwargs)
    events = self[deterministic_electron_seeds](events, **kwargs)

    # optional electron sceta production
    if "superclusterEta" not in events.Electron.fields:
        events = self[electron_sceta](events, **kwargs)

    # data/mc specific calibrations
    if self.dataset_inst.is_data:
        # nominal jec
        events = self[self.jec_nominal_cls](events, **kwargs)
        # nominal eec
        events = self[self.eec_nominal_cls](events, **kwargs)
    else:
        # for mc, when the nominal shift is requested, apply calibrations with uncertainties (i.e. full), otherwise
        # invoke calibrators configured not to evaluate and save uncertainties
        if task.global_shift_inst.is_nominal:
            # full jec and jer
            events = self[self.jec_full_cls](events, **kwargs)
            events = self[self.deterministic_jer_jec_full_cls](events, **kwargs)
            # full tec
            events = self[self.tec_full_cls](events, **kwargs)
            # full eer
            events = self[self.deterministic_eer_full_cls](events, **kwargs)
            # full eec (after eer!)
            events = self[self.eec_full_cls](events, **kwargs)
        else:
            # nominal jec and jer
            events = self[self.jec_nominal_cls](events, **kwargs)
            events = self[self.deterministic_jec_jec_nominal_cls](events, **kwargs)
            # nominal tec
            events = self[self.tec_nominal_cls](events, **kwargs)
            # nominal eer
            events = self[self.deterministic_eer_nominal_cls](events, **kwargs)

    # apply met phi correction
    if self.has_dep(self.met_phi_cls):
        events = self[self.met_phi_cls](events, **kwargs)

    return events


@default.init
def default_init(self: Calibrator, **kwargs) -> None:
    # set the name of the met collection to use
    met_name = self.config_inst.x.met_name
    raw_met_name = self.config_inst.x.raw_met_name

    # derive calibrators to add settings once
    flag = f"custom_calibs_registered_{self.cls_name}"
    if not self.config_inst.x(flag, False):
        def add_calib_cls(name, base, cls_dict=None):
            self.config_inst.set_aux(f"calib_{name}_cls", base.derive(name, cls_dict=cls_dict or {}))

        # jec calibrators
        add_calib_cls("jec_full", jec, cls_dict={
            "mc_only": True,
            "met_name": met_name,
            "raw_met_name": raw_met_name,
        })
        add_calib_cls("jec_nominal", jec, cls_dict={
            "uncertainty_sources": [],
            "met_name": met_name,
            "raw_met_name": raw_met_name,
        })
        # versions of jer that use the first random number from deterministic_seeds
        add_calib_cls("deterministic_jer_jec_full", jer, cls_dict={
            "deterministic_seed_index": 0,
            "met_name": met_name,
        })
        add_calib_cls("deterministic_jec_jec_nominal", jer, cls_dict={
            "deterministic_seed_index": 0,
            "met_name": met_name,
            "jec_uncertainty_sources": [],
        })
        # derive tec calibrators
        add_calib_cls("tec_full", tec, cls_dict={
            "propagate_met": False,  # not needed after JET-to-MET propagation
        })
        add_calib_cls("tec_nominal", tec, cls_dict={
            "propagate_met": False,  # not needed after JET-to-MET propagation
            "with_uncertainties": False,
        })
        # derive electron scale and resolution calibrators
        add_calib_cls("eec_full", eec)
        add_calib_cls("eec_nominal", eec, cls_dict={
            "with_uncertainties": False,
        })
        add_calib_cls("deterministic_eer_full", eer, cls_dict={
            "deterministic_seed_index": 0,
        })
        add_calib_cls("deterministic_eer_nominal", eer, cls_dict={
            "deterministic_seed_index": 0,
            "with_uncertainties": False,
        })
        # derive met_phi calibrator (currently only used in run 2)
        add_calib_cls("met_phi", met_phi, cls_dict={
            "met_name": met_name,
        })

        # change the flag
        self.config_inst.set_aux(flag, True)

    # store references to classes
    self.jec_full_cls = self.config_inst.x.calib_jec_full_cls
    self.jec_nominal_cls = self.config_inst.x.calib_jec_nominal_cls
    self.deterministic_jer_jec_full_cls = self.config_inst.x.calib_deterministic_jer_jec_full_cls
    self.deterministic_jec_jec_nominal_cls = self.config_inst.x.calib_deterministic_jec_jec_nominal_cls
    self.tec_full_cls = self.config_inst.x.calib_tec_full_cls
    self.tec_nominal_cls = self.config_inst.x.calib_tec_nominal_cls
    self.eec_full_cls = self.config_inst.x.calib_eec_full_cls
    self.eec_nominal_cls = self.config_inst.x.calib_eec_nominal_cls
    self.deterministic_eer_full_cls = self.config_inst.x.calib_deterministic_eer_full_cls
    self.deterministic_eer_nominal_cls = self.config_inst.x.calib_deterministic_eer_nominal_cls
    self.met_phi_cls = self.config_inst.x.calib_met_phi_cls

    # collect derived calibrators and add them to the calibrator uses and produces
    derived_calibrators = {
        self.jec_full_cls,
        self.jec_nominal_cls,
        self.deterministic_jer_jec_full_cls,
        self.deterministic_jec_jec_nominal_cls,
        self.tec_full_cls,
        self.tec_nominal_cls,
        IF_RUN_3(self.eec_full_cls),
        IF_RUN_3(self.eec_nominal_cls),
        IF_RUN_3(self.deterministic_eer_full_cls),
        IF_RUN_3(self.deterministic_eer_nominal_cls),
        IF_RUN_2(self.met_phi_cls),
    }
    self.uses |= derived_calibrators
    self.produces |= derived_calibrators
