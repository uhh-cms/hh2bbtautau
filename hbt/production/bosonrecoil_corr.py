# coding: utf-8

"""
Bosonic recoil corrections.

This module applies bosonic recoil corrections to PuppiMET using corrections derived from 
boson+jets control regions (e.g. in DY). It computes the recoil “U” vector from the measured MET 
and the full and visible boson momenta (visible=no neutrinos considered), applies a correction
(using the QuantileMapHist method, other available methods: QuantileMapFit, Rescaling), 
and then recomputes the corrected MET. In addition, it calculates systematic variations for 
recoil response and resolution uncertainties using the Recoil_correction_Uncertainty correction.

Inputs (example):
  # TODO: figure out how to make the MET name configurable via the config setting Run 2 / Run 3
  - PuppiMET.{pt,phi}: Measured MET (PuppiMET)
  - Jet.{pt}: Jet collection (pt), required to provide nmber of jets (as a per-event scalar)
  # TODO: figure out how to supply GenBoson and VisBoson (either produce them earlier or compute
          them directly in this function)
  - GenBoson.{pt,phi}: Gen-level (full) boson momentum (including neutrinos)
  - VisBoson.{pt,phi}: Visible boson momentum (without neutrinos)

The module produces the following output columns:
  - met_pt_recoil, met_phi_recoil : Nominal corrected MET (using QuantileMapHist corrections)
  - met_pt_recoil_RespUp, met_phi_recoil_RespUp
  - met_pt_recoil_RespDown, met_phi_recoil_RespDown
  - met_pt_recoil_ResolUp, met_phi_recoil_ResolUp
  - met_pt_recoil_ResolDown, met_phi_recoil_ResolDown

Configuration:
  - external_files.recoil: should point to corrections file. For example, in your config:
  
      cfg.x.external_files = DotDict.wrap({
          "recoil": "/afs/cern.ch/work/p/pgadow/public/mirrors/htt-common/2025-02-03/Recoil_corrections_v1.json.gz",
      })
      
  - config_inst.x.era: the era string (e.g. "2022preEE_NLO") to be passed to the corrections
    Possible choices: 2022preEE_NLO, 2022preEE_LO, 2022preEE_NNLO, 2022postEE_NLO, 2022postEE_LO, 2022postEE_NNLO
"""

import functools
import math

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, InsertableDict
from columnflow.columnar_util import set_ak_column, flat_np_view, layout_ak_array

# maybe import awkward in case this Producer is actually run, this needs to be set as columnflow
# would else give an error during setup, as these packages are not in the default sandbox
ak = maybe_import("awkward")
np = maybe_import("numpy")
vector = maybe_import("vector") 

# Helper to set float32 columns
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


#
# Helper functions for kinematics.
#
def GetXYfromPTPHI(pt, phi):
    """Convert (pt, phi) to (x, y)."""
    return pt * np.cos(phi), pt * np.sin(phi)


def GetPTPHIfromXY(x, y):
    """Convert (x, y) to (pt, phi)."""
    pt = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    return pt, phi


def GetU(met_pt, met_phi, full_pt, full_phi, vis_pt, vis_phi):
    """
    Compute the recoil vector U:
      U = MET + (visible boson) – (full boson)
    and then project U into components parallel (Upara) and perpendicular (Uperp)
    to the full boson direction.
    """
    met_x, met_y = GetXYfromPTPHI(met_pt, met_phi)
    full_x, full_y = GetXYfromPTPHI(full_pt, full_phi)
    vis_x, vis_y = GetXYfromPTPHI(vis_pt, vis_phi)
    Ux = met_x + vis_x - full_x
    Uy = met_y + vis_y - full_y
    U_pt, U_phi = GetPTPHIfromXY(Ux, Uy)
    # Projection along and perpendicular to full boson phi:
    Upara = U_pt * np.cos(U_phi - full_phi)
    Uperp = U_pt * np.sin(U_phi - full_phi)
    return Upara, Uperp


def GetMETfromU(upara, uperp, full_pt, full_phi, vis_pt, vis_phi):
    """
    Reconstruct MET from the corrected U components.
    """
    U_pt = np.sqrt(upara * upara + uperp * uperp)
    # Recover U_phi (note the rotation by full_phi)
    U_phi = np.arctan2(uperp, upara) + full_phi
    Ux, Uy = GetXYfromPTPHI(U_pt, U_phi)
    full_x, full_y = GetXYfromPTPHI(full_pt, full_phi)
    vis_x, vis_y = GetXYfromPTPHI(vis_pt, vis_phi)
    met_x = Ux - vis_x + full_x
    met_y = Uy - vis_y + full_y
    met_pt, met_phi = GetPTPHIfromXY(met_x, met_y)
    return met_pt, met_phi


def GetH(met_pt, met_phi, full_pt, full_phi, vis_pt, vis_phi):
    """
    Compute the H vector defined as:
      H = -MET - (visible boson)
    Then project H into components relative to the full boson direction.
    """
    met_x, met_y = GetXYfromPTPHI(met_pt, met_phi)
    vis_x, vis_y = GetXYfromPTPHI(vis_pt, vis_phi)
    Hx = -met_x - vis_x
    Hy = -met_y - vis_y
    H_pt, H_phi = GetPTPHIfromXY(Hx, Hy)
    Hpara = H_pt * np.cos(H_phi - full_phi)
    Hperp = H_pt * np.sin(H_phi - full_phi)
    return Hpara, Hperp


def GetMETfromH(hpara, hperp, full_pt, full_phi, vis_pt, vis_phi):
    """
    Reconstruct MET from the H components.
    """
    H_pt = np.sqrt(hpara * hpara + hperp * hperp)
    H_phi = np.arctan2(hperp, hpara) + full_phi
    Hx, Hy = GetXYfromPTPHI(H_pt, H_phi)
    vis_x, vis_y = GetXYfromPTPHI(vis_pt, vis_phi)
    met_x = -Hx - vis_x
    met_y = -Hy - vis_y
    met_pt, met_phi = GetPTPHIfromXY(met_x, met_y)
    return met_pt, met_phi



def reconstruct_boson(genparts):
    """
    Reconstruct the full and visible boson information from GenPart.
    Currently, this is a dummy implementation providing just zero,
    assuming that there is just one full boson and one vis boson per event.
    """

    # Get the boson indices.
    boson_indices = ak.where((genparts.pdgId == 23) | (genparts.pdgId == 25))[0]

    # Get the boson indices.
    vis_boson_indices = ak.where((genparts.pdgId == 23) | (genparts.pdgId == 25))[0]

    # Get the full boson information.
    full_boson = genparts[boson_indices[0]]
    vis_boson = genparts[vis_boson_indices[0]]

    # Dummy implementation: just return zero.
    return full_boson.zeros_like(), vis_boson.zeros_like()

#
# The recoil corrections producer.
#
@producer(
    uses={
        # PuppiMET information
        "PuppiMET.{pt,phi}",
        # Number of jets (as a per-event scalar)
        "Jet.{pt}",
        # TODO: option 1 is to provide use GenPart to reconstruct visible and full bosons
        "GenPart.{pt,eta,phi,mass,genPartIdxMother,pdgId}",
        # TODO: option 2 is to compute upstream full boson information (dummy name: GenBoson)
        #                      and visible boson information (dummy name: VisBoson)
        # Gen-level boson information (full boson momentum)
        # "GenBoson.{pt,phi}",
        # Visible boson information (without neutrinos)
        # "VisBoson.{pt,phi}",
    },
    produces={
        "met_pt_recoil", "met_phi_recoil",
        # TODO: figure out how to better provide outputs in style of columnflow
        "met_pt_recoil_RespUp", "met_phi_recoil_RespUp",
        "met_pt_recoil_RespDown", "met_phi_recoil_RespDown",
        "met_pt_recoil_ResolUp", "met_phi_recoil_ResolUp",
        "met_pt_recoil_ResolDown", "met_phi_recoil_ResolDown",
    },
    mc_only=True,
    # function to determine the recoil correction file from external files
    get_recoil_file=(lambda self, external_files: external_files.recoil),
    # function to get the era from the configuration (e.g. "2022preEE_NLO")
    get_recoil_era=(lambda self: self.config_inst.x.era),
)
def recoil_corrections(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer for bosonic recoil corrections.

    Steps:
      1) Compute the recoil vector U (Upara, Uperp) from the PuppiMET and the boson information.
      2) Apply the nominal recoil correction using the QuantileMapHist method.
      3) Recompute the corrected MET from the corrected U components.
      4) Compute systematic variations (Recoil uncertainties) by recalculating MET variations using the 
         Recoil_correction_Uncertainty correction.
    """
    # Retrieve inputs as numpy arrays.

    # TODO: dummy implementation to get visible and full boson information from GenPart
    full_boson, vis_boson = reconstruct_boson(events.GenPart)

    met_pt    = np.asarray(events.PuppiMET.pt)
    met_phi   = np.asarray(events.PuppiMET.phi)

    full_pt   = np.asarray(full_boson.pt)
    full_phi  = np.asarray(full_boson.phi)
    vis_pt    = np.asarray(vis_boson.pt)
    vis_phi   = np.asarray(vis_boson.phi)

    # full_pt   = np.asarray(events.GenBoson.pt)
    # full_phi  = np.asarray(events.GenBoson.phi)
    # vis_pt    = np.asarray(events.VisBoson.pt)
    # vis_phi   = np.asarray(events.VisBoson.phi)

    # nJet is expected to be a per-event scalar; convert to float for the correction functions.
    njet      = ak.num(events.Jet.pt, axis=1).astype(np.float32)
    # For the corrections, we use the full boson pt as ptll.
    ptll      = full_pt

    # Retrieve the era from configuration.
    era = self.get_recoil_era()

    #-------------------------------------------------------------------------
    # Nominal recoil correction:
    # (see here: https://cms-higgs-leprare.docs.cern.ch/htt-common/V_recoil/#example-snippet)
    # 1) Compute Upara and Uperp from the original MET and boson information.
    upara, uperp = GetU(met_pt, met_phi, full_pt, full_phi, vis_pt, vis_phi)

    # 2) Apply the nominal recoil correction using the QuantileMapHist method.
    #    Correction function signature:
    #      (era: str, njet: float, ptll: float, var: "Upara"/"Uperp", value: real)
    upara_corr = self.recoil_corr.evaluate(era, njet, ptll, "Upara", upara)
    uperp_corr = self.recoil_corr.evaluate(era, njet, ptll, "Uperp", uperp)

    # 3) Recompute the corrected MET from the corrected U components.
    met_pt_corr, met_phi_corr = GetMETfromU(upara_corr, uperp_corr, full_pt, full_phi, vis_pt, vis_phi)
    events = set_ak_column_f32(events, "met_pt_recoil", met_pt_corr)
    events = set_ak_column_f32(events, "met_phi_recoil", met_phi_corr)

    #-------------------------------------------------------------------------
    # Recoil uncertainty variations:
    # First, derive H components from the nominal corrected MET.
    hpara, hperp = GetH(met_pt_corr, met_phi_corr, full_pt, full_phi, vis_pt, vis_phi)


    # TODO: figure out how to store variations only for nominal event or more in line
    #       with columnflow philosophy
    # Loop over systematic variations.
    for syst in ["RespUp", "RespDown", "ResolUp", "ResolDown"]:
        # The recoil uncertainty correction for H components expects:
        #   (era: str, njet: float, ptll: float, var: "Hpara"/"Hperp", value: real, syst: string)
        hpara_var = self.recoil_unc.evaluate(era, njet, ptll, "Hpara", hpara, syst)
        hperp_var = self.recoil_unc.evaluate(era, njet, ptll, "Hperp", hperp, syst)
        met_pt_var, met_phi_var = GetMETfromH(hpara_var, hperp_var, full_pt, full_phi, vis_pt, vis_phi)
        events = set_ak_column_f32(events, f"met_pt_recoil_{syst}", met_pt_var)
        events = set_ak_column_f32(events, f"met_phi_recoil_{syst}", met_phi_var)

    return events


@recoil_corrections.requires
def recoil_corrections_requires(self: Producer, reqs: dict) -> None:
    # Ensure that external files are bundled.
    if "external_files" in reqs:
        return
    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@recoil_corrections.setup
def recoil_corrections_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    """
    Setup the recoil corrections by loading the CorrectionSet via correctionlib.
    The external recoil correction file should be provided as external_files.recoil.
    """
    bundle = reqs["external_files"]

    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate

    # Load the correction set from the external file.
    cset = correctionlib.CorrectionSet.from_string(
        self.get_recoil_file(bundle.files).load(formatter="gzip").decode("utf-8")
    )
    # Retrieve the corrections used for the nominal correction and for uncertainties.
    self.recoil_corr = cset["Recoil_correction_QuantileMapHist"]
    self.recoil_unc  = cset["Recoil_correction_Uncertainty"]
