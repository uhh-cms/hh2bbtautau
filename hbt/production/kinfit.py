# coding: utf-8

"""
Kinematic fit related producers.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.columnar_util import set_ak_column, attach_coffea_behavior
from columnflow.util import maybe_import

from hbt.util import create_lvector_xyz, stack_lvectors, rotate_px_py

np = maybe_import("numpy")
ak = maybe_import("awkward")


# helpers
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        "{Electron,Muon,Tau,HHBJet}.{pt,eta,phi,mass}",
        "reg_dnn_moe_nu{1,2}_p{x,y,z}",
    },
    produces={
        "h_kinfit.{b,l}_scale",
        "h_kinfit.{b1,b2,l1,l2,hbb,hll,hh}.{pt,eta,phi,mass}",
    },
    require_producers=["reg_dnn_moe"],
)
def h_kinfits(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Simplified kinematic fit of the bb and regressed tautau branches to fit both Higgs mass. This is not a real
    kinematic fit within uncertainties etc, but a simple scaling of the input four-vectors so that their invariant mass
    matches the target mass of 125 GeV. The scaling is constraint by the minimum _relative_ shift of the input vectors
    in quadrature.
    """
    # get bjets
    events = attach_coffea_behavior(events, {"HHBJet": "Jet"})
    b1 = events.HHBJet[:, 0]
    b2 = events.HHBJet[:, 1]

    # get regressed leptons
    vis_leps = stack_lvectors([events.Electron, events.Muon, events.Tau])[..., [0, 1]]
    ref_phi = vis_leps.sum(axis=-1).phi
    nu1 = create_lvector_xyz(
        *rotate_px_py(events.reg_dnn_moe_nu1_px, events.reg_dnn_moe_nu1_py, ref_phi),
        events.reg_dnn_moe_nu1_pz,
    )
    nu2 = create_lvector_xyz(
        *rotate_px_py(events.reg_dnn_moe_nu2_px, events.reg_dnn_moe_nu2_py, ref_phi),
        events.reg_dnn_moe_nu2_pz,
    )
    l1 = stack_lvectors([vis_leps[..., 0], nu1]).sum(axis=-1)
    l2 = stack_lvectors([vis_leps[..., 1], nu2]).sum(axis=-1)

    # helper to compute the scale that brings p1 and p2 to a target mass
    def get_scale(p1: ak.Array, p2: ak.Array, target_mass=125.0) -> np.ndarray:
        pp = p1.e * p2.e - p1.x * p2.x - p1.y * p2.y - p1.z * p2.z
        return target_mass / np.sqrt(2 * (p2.m**2 + pp))

    # fit bb branch
    b_scale = get_scale(b1, b2)
    events = set_ak_column_f32(events, "h_kinfit.b_scale", b_scale)

    # fit tautau branch
    l_scale = get_scale(l1, l2)
    events = set_ak_column_f32(events, "h_kinfit.l_scale", l_scale)

    # store scaled four vectors
    vecs = {
        "b1": b1 * b_scale,
        "b2": b2 * b_scale,
        "l1": l1 * l_scale,
        "l2": l2 * l_scale,
    }
    vecs["hbb"] = vecs["b1"] + vecs["b2"]
    vecs["hll"] = vecs["l1"] + vecs["l2"]
    vecs["hh"] = vecs["hbb"] + vecs["hll"]
    for name, p in vecs.items():
        for attr in ["pt", "eta", "phi", "mass"]:
            events = set_ak_column_f32(events, f"h_kinfit.{name}.{attr}", getattr(p, attr))

    return events
