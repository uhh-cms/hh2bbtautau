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
    hbb=True,
    hll=True,
    add_regression=True,
    output_name=None,
)
def h_kinfit(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Simplified kinematic fit of the bb and regressed tautau branches to fit both Higgs mass. This is not a real
    kinematic fit within uncertainties etc, but a simple scaling of the input four-vectors so that their invariant mass
    matches the target mass of 125 GeV. The scaling is constraint by the minimum _relative_ shift of the input vectors
    in quadrature.
    """
    # helper to compute the scale that brings p1 and p2 to a target mass
    def get_scale(p1: ak.Array, p2: ak.Array, target_mass=125.0) -> np.ndarray:
        pp = p1.e * p2.e - p1.x * p2.x - p1.y * p2.y - p1.z * p2.z
        return target_mass / np.sqrt(2 * (p2.m**2 + pp))

    # arrays of scaled four-vectors to be stored as columns below
    vecs: dict[str, np.ndarray] = {}

    # bb branch
    events = attach_coffea_behavior(events, {"HHBJet": "Jet"})
    b1 = events.HHBJet[:, 0]
    b2 = events.HHBJet[:, 1]
    if self.hbb:
        b_scale = get_scale(b1, b2)
        events = set_ak_column_f32(events, f"{self.output_name}.b_scale", b_scale)
    else:
        b_scale = 1.0
    vecs["b1"] = b1 * b_scale
    vecs["b2"] = b2 * b_scale
    vecs["hbb"] = vecs["b1"] + vecs["b2"]

    # ll branch
    vis_leps = stack_lvectors([events.Electron, events.Muon, events.Tau])[..., [0, 1]]
    l1 = vis_leps[..., 0]
    l2 = vis_leps[..., 1]
    if self.add_regression:
        ref_phi = vis_leps.sum(axis=-1).phi
        nu1 = create_lvector_xyz(
            *rotate_px_py(events.reg_dnn_moe_nu1_px, events.reg_dnn_moe_nu1_py, ref_phi),
            events.reg_dnn_moe_nu1_pz,
        )
        nu2 = create_lvector_xyz(
            *rotate_px_py(events.reg_dnn_moe_nu2_px, events.reg_dnn_moe_nu2_py, ref_phi),
            events.reg_dnn_moe_nu2_pz,
        )
        l1 = stack_lvectors([l1, nu1]).sum(axis=-1)
        l2 = stack_lvectors([l2, nu2]).sum(axis=-1)
    if self.hll:
        # fit tautau branch
        l_scale = get_scale(l1, l2)
        events = set_ak_column_f32(events, f"{self.output_name}.l_scale", l_scale)
    else:
        l_scale = 1.0
    vecs["l1"] = l1 * l_scale
    vecs["l2"] = l2 * l_scale
    vecs["hll"] = vecs["l1"] + vecs["l2"]

    # add hh
    vecs["hh"] = vecs["hbb"] + vecs["hll"]

    # store scaled four vectors
    for name, p in vecs.items():
        for attr in ["pt", "eta", "phi", "mass"]:
            events = set_ak_column_f32(events, f"{self.output_name}.{name}.{attr}", getattr(p, attr))

    return events


@h_kinfit.init
def h_kinfit_init(self: Producer, **kwargs) -> None:
    if not self.hbb and not self.hll:
        raise ValueError(f"at least one of hbb or hll must be True for '{self.cls_name}' producer")

    # default output_name
    if not self.output_name:
        self.output_name = self.cls_name

    # always read bjets and leptons, and produce h variables
    self.uses |= {"{HHBJet,Electron,Muon,Tau}.{pt,eta,phi,mass}"}
    self.produces |= {f"{self.output_name}.{{hbb,hll,hh}}.{{pt,eta,phi,mass}}"}

    if self.hbb:
        self.produces |= {f"{self.output_name}.b_scale", f"{self.output_name}.{{b1,b2}}.{{pt,eta,phi,mass}}"}

    if self.hll:
        self.produces |= {f"{self.output_name}.l_scale", f"{self.output_name}.{{l1,l2}}.{{pt,eta,phi,mass}}"}

    if self.add_regression:
        self.require_producers += ["reg_dnn_moe"]
        self.uses |= {"reg_dnn_moe_nu{1,2}_p{x,y,z}"}


h_kinfit.derive("h_kinfit_bb", cls_dict={"hll": False})
h_kinfit.derive("h_kinfit_ll", cls_dict={"hbb": False})
h_kinfit.derive("h_kinfit_llvis", cls_dict={"hbb": False, "add_regression": False})
h_kinfit.derive("h_kinfit_vis", cls_dict={"add_regression": False})
