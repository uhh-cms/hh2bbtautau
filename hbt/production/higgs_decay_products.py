# coding: utf-8
import numpy as np


"""
Producers that determine the generator-level particles related to the HH2BBTauTau process.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import attach_coffea_behavior as attach_coffea_behavior_fn, EMPTY_FLOAT, EMPTY_INT
# from functools import partial

ak = maybe_import("awkward")


# Define helper functions
def shape_array(input_array: ak.Array) -> ak.Array:
    """Shapes the input array so it can be correctly concatenated later
    Args:
        input_array (ak.Array): The array to be shaped
    Returns:
        output_array (ak.Array): The shaped array
    """
    output_array = ak.flatten(input_array, axis=3)
    output_array = ak.flatten(output_array, axis=2)
    output_array = ak.pad_none(output_array, 2, axis=1)

    return output_array


@producer(
    uses={"GenPart.*", attach_coffea_behavior},
    produces={"higgs_family.*"},  # "bottoms_inv_mass", "taus_inv_mass"},
)
def higgs_decay_products(self: Producer, events: ak.Array, **kwargs):
    # TODO: change
    """
    Creates a new ragged column "higgs_family" that stores the H and their decay products. The structure will be as
    follows:

    .. code-block:: python
        [
            # event 1
            [
                [H_bb, H_tautau], [b+, b-], [tau+, tau-], [W+, W-], [tau+ nu, tau- nu], [mu+, mu-], [e+, e-],
                [mu- nu, mu+ nu], [e- mu, e+ mu]
            ],
            # event 2
            ...
        ],

    where H1 decays to bb and H2 decays to tautau. If a certain particle is not present in an event, the
    corresponding array entry will be EMPTY_FLOAT / EMPTY_INT.
    """
    mother_gen_flags = ["isLastCopy", "fromHardProcess"]
    children_gen_flags = ["isFirstCopy", "fromHardProcess"]
    # To prevent tau children from loop processes from being stored
    tau_children_flags = ["isFirstCopy", "isTauDecayProduct"]

    # find and sort higgses
    abs_id = abs(events.GenPart.pdgId)
    higgs = events.GenPart[abs_id == 25]
    higgs = higgs[higgs.hasFlags(*mother_gen_flags)]
    higgs_bb = higgs[ak.any(abs(higgs.distinctChildren.pdgId) == 5, axis=-1)]
    higgs_tautau = higgs[ak.any(abs(higgs.distinctChildren.pdgId) == 15, axis=-1)]
    higgs_combined = ak.concatenate([higgs_bb, higgs_tautau], axis=1)

    higgs_children = higgs.distinctChildren
    higgs_children = higgs_children[higgs_children.hasFlags(*children_gen_flags)]

    # sort different decay products
    bottoms = higgs_children[abs(higgs_children.pdgId) == 5]
    bottoms = ak.flatten(bottoms, axis=2)
    bottoms = bottoms[ak.argsort(ak.fill_none(bottoms.pdgId, 0), axis=1, ascending=True)]

    taus = higgs_children[abs(higgs_children.pdgId) == 15]
    tau_children = taus.distinctChildrenDeep[taus.distinctChildrenDeep.hasFlags(*tau_children_flags)]
    taus = ak.flatten(taus, axis=2)
    taus = taus[ak.argsort(ak.fill_none(taus.pdgId, 0), axis=1, ascending=True)]

    # sort the tau decay products
    nu_tau = shape_array(tau_children[abs(tau_children.pdgId) == 16])
    nu_tau = nu_tau[ak.argsort(ak.fill_none(nu_tau.pdgId, 0), axis=1, ascending=True)]
    muons = shape_array(tau_children[abs(tau_children.pdgId) == 13])
    muons = muons[ak.argsort(ak.fill_none(muons.pdgId, 0), axis=1, ascending=True)]
    electrons = shape_array(tau_children[abs(tau_children.pdgId) == 11])
    electrons = electrons[ak.argsort(ak.fill_none(electrons.pdgId, 0), axis=1, ascending=True)]
    nu_mu = shape_array(tau_children[abs(tau_children.pdgId) == 14])
    nu_mu = nu_mu[ak.argsort(ak.fill_none(nu_mu.pdgId, 0), axis=1, ascending=False)]
    nu_e = shape_array(tau_children[abs(tau_children.pdgId) == 12])
    nu_e = nu_e[ak.argsort(ak.fill_none(nu_e.pdgId, 0), axis=1, ascending=False)]
    """
    qq = w_children[w_children.pdgId > 0]
    qq = qq[qq.pdgId < 5]
    qq = ak.flatten(qq, axis=3)
    qq = ak.firsts(qq, axis=2)
    """
    # tau hadronic decay mainly into pions, but also Kaon
    # pions pdg IDs: pi^0: 111, pi^+: 211
    # Kaon pdg IDs: K^0: 311, K^+: 321
    # Attention: only hadronic children, tau neutrino is stored elsewhere
    tau_minus = taus[:, 1]
    tau_minus_children = tau_minus.distinctChildrenDeep[tau_minus.distinctChildrenDeep.hasFlags(*tau_children_flags)]
    pi_zero = tau_minus_children[tau_minus_children.pdgId == 111]
    pi_plus = tau_minus_children[tau_minus_children.pdgId == 211]
    pi_minus = tau_minus_children[tau_minus_children.pdgId == -211]
    K_zero = tau_minus_children[tau_minus_children.pdgId == 311]
    K_plus = tau_minus_children[tau_minus_children.pdgId == 321]
    K_minus = tau_minus_children[tau_minus_children.pdgId == -321]
    tau_minus_children = ak.concatenate([
        pi_zero,
        pi_plus,
        pi_minus,
        K_zero,
        K_plus,
        K_minus,
    ], axis=1)
    tau_minus_children = ak.pad_none(tau_minus_children, 1)
    # tau_minus_children = ak.concatenate([
    #     pi_zero[:, None, :],
    #     pi_plus[:, None, :],
    #     pi_minus[:, None, :],
    #     K_zero[:, None, :],
    #     K_plus[:, None, :],
    #     K_minus[:, None, :],
    # ], axis=1)

    tau_plus = taus[:, 0]
    tau_plus_children = tau_plus.distinctChildrenDeep[tau_plus.distinctChildrenDeep.hasFlags(*tau_children_flags)]
    pi_zero = tau_plus_children[tau_plus_children.pdgId == 111]
    pi_plus = tau_plus_children[tau_plus_children.pdgId == 211]
    pi_minus = tau_plus_children[tau_plus_children.pdgId == -211]
    K_zero = tau_plus_children[tau_plus_children.pdgId == 311]
    K_plus = tau_plus_children[tau_plus_children.pdgId == 321]
    K_minus = tau_plus_children[tau_plus_children.pdgId == -321]
    tau_plus_children = ak.concatenate([
        pi_zero,
        pi_plus,
        pi_minus,
        K_zero,
        K_plus,
        K_minus,
    ], axis=1)
    tau_plus_children = ak.pad_none(tau_plus_children, 1)

    tau_hadronic_children = ak.concatenate([
        tau_minus_children[:, None],
        tau_plus_children[:, None],
    ], axis=1)

    # qq = taus.children[taus.children.pdgId > 0]
    # qq = qq[qq.pdgId < 5]
    # qq = ak.flatten(qq, axis=2)
    # qbarqbar = taus.children[taus.children.pdgId < 0]
    # qbarqbar = qbarqbar[qbarqbar.pdgId > -5]
    # qbarqbar = ak.flatten(qbarqbar, axis=2)

    # Different structure for the hadronic children: tau_children_sorted[:, 5] = tau_minus_children, [:, 6]
    #   are the tau_plus_children. Creating Ws didn't work otherwise
    tau_children_sorted = ak.concatenate([
        nu_tau[:, None, :],
        muons[:, None, :],
        electrons[:, None, :],
        nu_mu[:, None, :],
        nu_e[:, None, :],
        tau_hadronic_children,
    ], axis=1)

    behaving_children = attach_coffea_behavior_fn(tau_children_sorted, collections={"higgs_family": {"type_name":
        "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G"}})

    # Add relevant fields to selfbuild Ws
    W_1 = behaving_children[:, 1:5, 0].sum()
    W_1.add(behaving_children[:, 5].sum())
    W_2 = behaving_children[:, 1:5, 1].sum()
    W_1.add(behaving_children[:, 6].sum())
    W_1["pdgId"] = 24
    W_2["pdgId"] = -24
    W_1["pt"] = W_1.pt
    W_2["pt"] = W_2.pt
    W_1["eta"] = W_1.eta
    W_2["eta"] = W_2.eta
    W_1["phi"] = W_1.phi
    W_2["phi"] = W_2.phi
    field_list = np.array(tau_children.fields, dtype=str)

    for W_iter in [W_1, W_2]:
        for field in field_list:
            if field != "pt" and field != "pdgId":
                W_iter[field] = None

    W_bosons = ak.concatenate([W_1[:, None], W_2[:, None]], axis=1)
    # higgs_family = ak.concatenate([
    #     higgs_combined[:, None, :],
    #     bottoms[:, None, :],
    #     taus[:, None, :],
    #     W_bosons[:, None, :],
    #     tau_children_sorted[:, :5],
    #     tau_hadronic_children[:, None, :]
    # ], axis=1)

    def fill_none_fields(array):
        float_fields = ("eta", "mass", "phi", "pt", "vx", "vy", "vz", "iso")
        skip_fields = ("genPartIdxMother", "genPartIdxMotherG", "distinctParentIdxG", "childrenIdxG",
            "distinctChildrenIdxG", "distinctChildrenDeepIdxG")
        if not hasattr(array, "fields"):
            return array
        new_dict = {}
        for field in array.fields:
            if field in skip_fields:
                continue
            elif field in float_fields:
                new_dict[field] = ak.values_astype(ak.fill_none(array[field], EMPTY_FLOAT), type(EMPTY_FLOAT))
            else:
                new_dict[field] = ak.values_astype(ak.fill_none(array[field], EMPTY_INT), type(EMPTY_INT))
        array = ak.zip({field: new_dict[field] for field in new_dict.keys()})
        return array

    higgs_family = ak.zip({
        "higgs": fill_none_fields(higgs_combined),
        "bottoms": fill_none_fields(bottoms),
        "taus": fill_none_fields(taus),
        "W_bosons": fill_none_fields(W_bosons),
        "tau_leptonic_decay_products": fill_none_fields(tau_children_sorted[:, :5]),
        "tau_hadronic_decay_products": fill_none_fields(tau_hadronic_children),
        # "tau_1_hadronic_decay_products": fill_none_fields(tau_hadronic_children[:, 0]),
        # "tau_2_hadronic_decay_products": fill_none_fields(tau_hadronic_children[:, 1]),
    }, with_name="higgs_family", depth_limit=1)
    events = set_ak_column(events, "higgs_family", higgs_family)
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
