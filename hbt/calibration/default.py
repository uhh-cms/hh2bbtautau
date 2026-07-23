# coding: utf-8

"""
Calibration methods.
"""

from __future__ import annotations

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.met import met_phi_run2, met_phi
from columnflow.calibration.cms.jets import jec, jer_horn_handling
from columnflow.calibration.cms.tau import tec
from columnflow.calibration.cms.egamma import electron_scale_smear
from columnflow.calibration.cms.muon import muon_sr
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.electron import electron_sceta
from columnflow.util import maybe_import

from hbt.util import IF_RUN_3, IF_RUN_3_2024, IF_MC

np = maybe_import("numpy")
ak = maybe_import("awkward")


# pt clamping for the evaluation of the L2L3Residual jec corrections in 2024
def jec_clamp_2024_data_l2l3residual(calibrator, corrector, variable_map):
    # apply only to L2L3Residual for data in 2024
    if (
        calibrator.config_inst.campaign.x.year == 2024 and
        calibrator.dataset_inst.is_data and
        corrector.level == "L2L3Residual"
    ):
        min_eta, max_eta = 2.0, 2.5
        clamp_pt = np.float32(35.0)
        clamp_mask = (
            ((abs_eta := abs(variable_map["JetEta"])) > min_eta) &
            (abs_eta < max_eta) &
            (variable_map["JetPt"] > clamp_pt)
        )
        variable_map["JetPt"] = ak.where(clamp_mask, clamp_pt, variable_map["JetPt"])
    return variable_map


@calibrator(
    uses={IF_MC(mc_weight), electron_sceta},
    produces={IF_MC(mc_weight)},
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    task = kwargs["task"]
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # optional electron sceta production
    if "superclusterEta" not in events.Electron.fields:
        events = self[electron_sceta](events, **kwargs)

    # data/mc specific calibrations
    if self.dataset_inst.is_data:
        # nominal jec
        events = self[self.jec_nominal_cls](events, **kwargs)
        # nominal ess
        events = self[self.ess_nominal_cls](events, **kwargs)
        # nominal muon scale and resolution
        events = self[self.muon_sr_nominal_cls](events, **kwargs)
    else:
        # for mc, when the nominal shift is requested, apply calibrations with uncertainties (i.e. full), otherwise
        # invoke calibrators configured not to evaluate and save uncertainties
        if task.global_shift_inst.is_nominal:
            # full jec and jer
            events = self[self.jec_full_cls](events, **kwargs)
            events = self[self.jer_jec_full_cls](events, **kwargs)
            # full tec
            events = self[self.tec_full_cls](events, **kwargs)
            # full ess
            events = self[self.ess_full_cls](events, **kwargs)
            # full muon scale and resolution
            events = self[self.muon_sr_full_cls](events, **kwargs)
        else:
            # nominal jec and jer
            events = self[self.jec_nominal_cls](events, **kwargs)
            events = self[self.jer_jec_nominal_cls](events, **kwargs)
            # nominal tec
            events = self[self.tec_nominal_cls](events, **kwargs)
            # nominal ess
            events = self[self.ess_nominal_cls](events, **kwargs)
            # nominal muon scale and resolution
            events = self[self.muon_sr_nominal_cls](events, **kwargs)

    # apply met phi correction
    if self.has_dep(self.met_phi_cls):
        events = self[self.met_phi_cls](events, **kwargs)

    return events


@default.init
def default_init(self: Calibrator, **kwargs) -> None:
    super(default, self).init_func(**kwargs)

    # set the name of the met collection to use
    met_name = self.config_inst.x.met_name
    raw_met_name = self.config_inst.x.raw_met_name

    # derive calibrators to add settings once
    flag = f"custom_calibs_registered_{self.cls_name}"
    if not self.config_inst.x(flag, False):
        def add_calib_cls(name, base, cls_dict=None):
            self.config_inst.set_aux(f"calib_{name}_cls", base.derive(name, cls_dict=cls_dict or {}))
        # jec calibrators
        update_jec_corrector_variables = (
            jec_clamp_2024_data_l2l3residual
            if self.config_inst.campaign.x.year == 2024
            else None
        )
        add_calib_cls("jec_full", jec, cls_dict={
            "mc_only": True,
            "met_name": met_name,
            "raw_met_name": raw_met_name,
            "update_corrector_variables": update_jec_corrector_variables,
        })
        add_calib_cls("jec_nominal", jec, cls_dict={
            "uncertainty_sources": [],
            "met_name": met_name,
            "raw_met_name": raw_met_name,
            "update_corrector_variables": update_jec_corrector_variables,
        })
        add_calib_cls("jer_jec_full", jer_horn_handling, cls_dict={
            "met_name": met_name,
        })
        add_calib_cls("jer_jec_nominal", jer_horn_handling, cls_dict={
            "met_name": met_name,
            "jec_uncertainty_sources": [],
        })
        # derive tec calibrators
        add_calib_cls("tec_full", tec, cls_dict={
            "met_name": met_name,
            "propagate_met": False,  # not needed after JET-to-MET propagation
        })
        add_calib_cls("tec_nominal", tec, cls_dict={
            "met_name": met_name,
            "propagate_met": False,  # not needed after JET-to-MET propagation
            "with_uncertainties": False,
        })
        # derive electron scale and resolution calibrators
        add_calib_cls("ess_full", electron_scale_smear)
        add_calib_cls("ess_nominal", electron_scale_smear, cls_dict={
            "with_uncertainties": False,
        })
        # derive muon scale and resolution calibrators
        add_calib_cls("muon_sr_full", muon_sr, cls_dict={
            "store_original": True,
        })
        add_calib_cls("muon_sr_nominal", muon_sr, cls_dict={
            "store_original": True,
            "with_uncertainties": False,
        })
        # derive met_phi calibrator
        add_calib_cls("met_phi", met_phi_run2 if self.config_inst.campaign.x.run == 2 else met_phi)

        # change the flag
        self.config_inst.set_aux(flag, True)

    # store references to classes
    self.jec_full_cls = self.config_inst.x.calib_jec_full_cls
    self.jec_nominal_cls = self.config_inst.x.calib_jec_nominal_cls
    self.jer_jec_full_cls = self.config_inst.x.calib_jer_jec_full_cls
    self.jer_jec_nominal_cls = self.config_inst.x.calib_jer_jec_nominal_cls
    self.tec_full_cls = self.config_inst.x.calib_tec_full_cls
    self.tec_nominal_cls = self.config_inst.x.calib_tec_nominal_cls
    self.ess_full_cls = self.config_inst.x.calib_ess_full_cls
    self.ess_nominal_cls = self.config_inst.x.calib_ess_nominal_cls
    self.muon_sr_full_cls = self.config_inst.x.calib_muon_sr_full_cls
    self.muon_sr_nominal_cls = self.config_inst.x.calib_muon_sr_nominal_cls
    self.met_phi_cls = self.config_inst.x.calib_met_phi_cls

    # collect derived calibrators and add them to the calibrator uses and produces
    derived_calibrators = {
        self.jec_full_cls,
        self.jec_nominal_cls,
        self.jer_jec_full_cls,
        self.jer_jec_nominal_cls,
        self.tec_full_cls,
        self.tec_nominal_cls,
        IF_RUN_3(self.ess_full_cls),
        IF_RUN_3(self.ess_nominal_cls),
        IF_RUN_3(self.muon_sr_full_cls),
        IF_RUN_3(self.muon_sr_nominal_cls),
        # TODO: 2024: remove condition when met phi corrections are made available
        ~IF_RUN_3_2024(self.met_phi_cls),
    }

    self.uses |= derived_calibrators
    self.produces |= derived_calibrators
