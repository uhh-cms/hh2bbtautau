# coding: utf-8

"""
Tau energy correction methods.
"""

import functools

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.util import propagate_met
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, flat_np_view

ak = maybe_import("awkward")
np = maybe_import("numpy")


# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@calibrator(
    uses={
        "nTau", "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass", "Tau.charge", "Tau.genPartFlav",
        "Tau.decayMode",
        "MET.pt", "MET.phi",
    },
    produces={
        "Tau.pt", "Tau.mass",
        "Tau.pt_tec_dm0_up", "Tau.pt_tec_dm0_down", "Tau.mass_tec_dm0_up", "Tau.mass_tec_dm0_down",
        "Tau.pt_tec_dm1_up", "Tau.pt_tec_dm1_down", "Tau.mass_tec_dm1_up", "Tau.mass_tec_dm1_down",
        "Tau.pt_tec_dm10_up", "Tau.pt_tec_dm10_down", "Tau.mass_tec_dm10_up", "Tau.mass_tec_dm10_down",
        "Tau.pt_tec_dm11_up", "Tau.pt_tec_dm11_down", "Tau.mass_tec_dm11_up", "Tau.mass_tec_dm11_down",
        "MET.pt", "MET.phi",
        "MET.pt_dm0_up", "MET.pt_dm0_down", "MET.phi_dm0_up", "MET.phi_dm0_down",
        "MET.pt_dm1_up", "MET.pt_dm1_down", "MET.phi_dm1_up", "MET.phi_dm1_down",
        "MET.pt_dm10_up", "MET.pt_dm10_down", "MET.phi_dm10_up", "MET.phi_dm10_down",
        "MET.pt_dm11_up", "MET.pt_dm11_down", "MET.phi_dm11_up", "MET.phi_dm11_down",
    },
)
def tec(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Calibrator for tau energy.

    Resources:
    https://twiki.cern.ch/twiki/bin/view/CMS/TauIDRecommendationForRun2?rev=111
    https://gitlab.cern.ch/cms-tau-pog/jsonpog-integration/-/blob/TauPOG_v2/POG/TAU/README.md
    """
    # the correction tool only supports flat arrays, so convert inputs to flat np view first
    pt = flat_np_view(events.Tau.pt, axis=1)
    mass = flat_np_view(events.Tau.mass, axis=1)
    eta = flat_np_view(events.Tau.eta, axis=1)
    dm = flat_np_view(events.Tau.decayMode, axis=1)
    match = flat_np_view(events.Tau.genPartFlav, axis=1)

    # get the scale factors for the four supported decay modes
    dm_mask = (dm == 0) | (dm == 1) | (dm == 10) | (dm == 11)
    scales_nom = np.ones_like(dm_mask, dtype=np.float32)
    scales_up = np.ones_like(dm_mask, dtype=np.float32)
    scales_down = np.ones_like(dm_mask, dtype=np.float32)

    args = (pt[dm_mask], eta[dm_mask], dm[dm_mask], match[dm_mask], self.config_inst.x.tau_tagger)
    scales_nom[dm_mask] = self.tec_corrector.evaluate(*args, "nom")
    scales_up[dm_mask] = self.tec_corrector.evaluate(*args, "up")
    scales_down[dm_mask] = self.tec_corrector.evaluate(*args, "down")

    # custom adjustment 1: reset where the matching value is unhandled
    # custom adjustment 2: reset electrons faking taus where the pt is too small
    mask1 = (match < 1) | (match > 5)
    mask2 = ((match == 1) | (match == 3) & (pt <= 20.0))

    # apply reset masks
    mask = mask1 | mask2
    scales_nom[mask] = 1.0
    scales_up[mask] = 1.0
    scales_down[mask] = 1.0

    # create varied collections per decay mode
    for _dm in [0, 1, 10, 11]:
        for direction, scales in [("up", scales_up), ("down", scales_down)]:
            # copy pt and mass
            pt_varied = ak.copy(events.Tau.pt)
            mass_varied = ak.copy(events.Tau.mass)
            pt_view = flat_np_view(pt_varied, axis=1)
            mass_view = flat_np_view(mass_varied, axis=1)

            # correct pt and mass for taus with that decay mode
            mask = dm == _dm
            pt_view[mask] *= scales[mask]
            mass_view[mask] *= scales[mask]

            # propagate changes to MET
            met_pt_varied, met_phi_varied = propagate_met(
                events.Tau.pt,
                events.Tau.phi,
                pt_varied,
                events.Tau.phi,
                events.MET.pt,
                events.MET.phi,
            )

            # save columns
            postfix = f"tec_dm{_dm}_{direction}"
            events = set_ak_column_f32(events, f"Tau.pt_{postfix}", pt_varied)
            events = set_ak_column_f32(events, f"Tau.mass_{postfix}", mass_varied)
            events = set_ak_column_f32(events, f"MET.pt_{postfix}", met_pt_varied)
            events = set_ak_column_f32(events, f"MET.phi_{postfix}", met_phi_varied)

    # apply the nominal correction
    # note: changes are applied to the views and directly propagate to the original ak arrays
    tau_sum_before = events.Tau.sum(axis=1)
    pt *= scales
    mass *= scales

    # propagate changes to MET
    met_pt, met_phi = propagate_met(
        tau_sum_before.pt,
        tau_sum_before.phi,
        events.Tau.pt,
        events.Tau.phi,
        events.MET.pt,
        events.MET.phi,
    )

    # save columns
    events = set_ak_column_f32(events, "MET.pt", met_pt)
    events = set_ak_column_f32(events, "MET.phi", met_phi)

    return events


@tec.requires
def tec_requires(self: Calibrator, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@tec.setup
def tec_setup(self: Calibrator, reqs: dict, inputs: dict) -> None:
    bundle = reqs["external_files"]

    # create the tec corrector
    import correctionlib
    correction_set = correctionlib.CorrectionSet.from_string(
        bundle.files.tau.load(formatter="gzip").decode("utf-8"),
    )
    self.tec_corrector = correction_set["tau_energy_scale"]
