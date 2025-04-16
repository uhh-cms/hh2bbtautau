# coding: utf-8
import numpy as np


"""
Producers that determine the generator-level particles related to the HH2BBTauTau process.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import attach_coffea_behavior as attach_coffea_behavior_fn

ak = maybe_import("awkward")


# define helper functions
def shape_array(input_array):
    # remove two outer axis
    output_array = ak.flatten(ak.flatten(input_array, axis=-1), axis=-1)
    return ak.pad_none(output_array, 2, axis=1)


def add_field(muons, electrons, nu_mu, nu_e, field: str):
    return (
        ak.fill_none(muons[field][:, 0], 0) +
        ak.fill_none(electrons[field][:, 0], 0) +
        ak.fill_none(nu_mu[field][:, 0], 0) +
        ak.fill_none(nu_e[field][:, 0], 0)
    )


def get_padded_indices(array, ascending=True):
    return ak.argsort(ak.fill_none(array.pdgId, 0), axis=1, ascending=ascending)


def get_children(array, particle_id, ascending=True):
    # small helper to extract children from mother particle array
    selected_particle = shape_array(array[abs(array.pdgId) == particle_id])
    # indices = ak.argsort(ak.fill_none(selected_particle.pdgId, 0), axis=1, ascending=ascending)
    return selected_particle[get_padded_indices(selected_particle, ascending=ascending)]


@producer(
    uses={"GenPart.*", attach_coffea_behavior},
    produces={"higgs_family.*", "bottoms_inv_mass", "taus_inv_mass"},
)
def higgs_decay_products(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "higgs_family" that stores the H and their decay products. The structure will be as follows:

    .. code-block:: python

        [
            # event 1
            [
                [H1,H2], [b+, b-], [tau+, tau-], [tau+ nu, tau- nu], [W+, W-] [mu+, mu-], [e+, e-], [mu- nu, mu+ nu], [e- mu, e+ mu]
            ],
            # event 2
            ...
        ],

    where H1 decays to bb and H2 decays to tautau. If a certain particle is not present in an event, the corresponding array entry will be None.
    """
    mother_gen_flags = ["isLastCopy", "fromHardProcess"]
    children_gen_flags = ["isFirstCopy", "fromHardProcess"]

    # find and sort higgses
    abs_id = abs(events.GenPart.pdgId)
    higgs = events.GenPart[abs_id == 25]
    higgs = higgs[higgs.hasFlags(*mother_gen_flags)]
    higgs_bb = higgs[ak.any(abs(higgs.distinctChildren.pdgId) == 5, axis=-1)]
    higgs_tautau = higgs[ak.any(abs(higgs.distinctChildren.pdgId) == 15, axis=-1)]
    higgs_combined = ak.concatenate([higgs_bb, higgs_tautau], axis=1)

    higgs_children = higgs.distinctChildren
    higgs_children = higgs_children[higgs_children.hasFlags(*children_gen_flags)]

    #sort different decay products
    bottoms = higgs_children[abs(higgs_children.pdgId) == 5]
    bottoms = ak.flatten(bottoms, axis=2)
    bottoms = bottoms[get_padded_indices(bottoms, ascending=True)]

    taus = higgs_children[abs(higgs_children.pdgId) == 15]
    tau_children = taus.distinctChildrenDeep[taus.distinctChildrenDeep.hasFlags("isFirstCopy")]
    taus = ak.flatten(taus, axis=2)
    taus = taus[get_padded_indices(taus, ascending=True)]

    muons = get_children(array=tau_children, particle_id=13, ascending=True)
    electrons = get_children(array=tau_children, particle_id=11, ascending=True)

    # nu_tau = ak.flatten(ak.flatten(tau_children[abs(tau_children.pdgId) == 16], axis = 3), axis = 2)

    nu_tau = get_children(array=tau_children, particle_id=16, ascending=True)
    nu_mu = get_children(array=tau_children, particle_id=14, ascending=False)
    nu_e = get_children(array=tau_children, particle_id=12, ascending=False)

    # events = self[attach_coffea_behavior](events, collections = "GenPart", **kwargs)
    # add new category axis with none
    tau_children_sorted = ak.concatenate([
        nu_tau[:, None, :],
        muons[:, None, :],
        electrons[:, None, :],
        nu_mu[:, None, :],
        nu_e[:, None, :]
    ], axis=1)

    behaving_children = attach_coffea_behavior_fn(
        tau_children_sorted,
        collections={"higgs_family": {
            "type_name": "GenParticle",
            "check_attr": "metric_table",
            "skip_fields": "*Idx*G"}
        }
    )

    W_1 = behaving_children[:, 3:, 0].sum()
    W_2 = behaving_children[:, 3:, 1].sum()
    W_1["pdgId"] = 24
    W_2["pdgId"] = -24
    W_1["pt"] = W_1.pt
    W_2["pt"] = W_2.pt
    field_list = np.array(tau_children.fields, dtype=str)
    for W_iter in [W_1, W_2]:
        for field in field_list:
            if field != "pt" and field != "pdgId":
                W_iter[field] = None
    W_bosons = ak.concatenate([W_1[:, None], W_2[:, None]], axis=1)
    higgs_family = ak.concatenate([
        higgs_combined[:, None, :],
        bottoms[:, None, :],
        taus[:, None, :],
        W_bosons[:, None, :],
        tau_children_sorted
    ], axis=1)

    # save the column
    events = set_ak_column(events, "higgs_family", higgs_family)

    #save validation columns
    bottoms_inv_mass = attach_coffea_behavior_fn(
        bottoms,
        collections={"higgs_family": {
            "type_name": "GenParticle",
            "check_attr": "metric_table",
            "skip_fields": "*Idx*G"}
        }).sum().absolute()

    taus_inv_mass = attach_coffea_behavior_fn(
        taus,
        collections={"higgs_family": {
            "type_name": "GenParticle",
            "check_attr": "metric_table",
            "skip_fields": "*Idx*G"}
        }).sum().absolute()

    events = set_ak_column(events, "bottoms_inv_mass", bottoms_inv_mass)
    events = set_ak_column(events, "taus_inv_mass", taus_inv_mass)
    from IPython import embed; embed(header="END - 163 in higgs_decay_products.py ")
    return events


# @top_decay_products.skip
# def gen_top_decay_products_skip(self: Producer) -> bool:
#     """
#     Custom skip function that checks whether the dataset is a MC simulation containing top
#     quarks in the first place.
#     """
#     # never skip when there is not dataset
#     if not getattr(self, "dataset_inst", None):
#         return False

#     return self.dataset_inst.is_data or not self.dataset_inst.has_tag("has_top")
