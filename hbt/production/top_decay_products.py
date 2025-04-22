# coding: utf-8

"""
Producers that determine the generator-level particles related to a top quark decay.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior

ak = maybe_import("awkward")


@producer(
    uses={"GenPart.*", attach_coffea_behavior},
    produces={"top_family.*", "reco_top_mass", "top_mass"},
)
def top_decay_products(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "higgs_family" that stores the tops and their decay products. The structure
    will be as follows:

    .. code-block:: python

        [
            # event 1
            [
                [t1,t2], [b_t1, b_t2], [W_t1, W_t2], [q_1, q_2], [qbar_1, qbar_2], [l_1, l_2], [lbar_1, lbar_2],
                [nu_1, nu_2], [nubar_1, nubar_2]
            ],
            # event 2
            ...
        ],

    where the first entry in each array belongs to the first top, and the second entry in each array belongs to the
    second top.
    """

    # find hard top quarks
    mother_gen_flags = ["isLastCopy", "fromHardProcess"]
    children_gen_flags = ["isFirstCopy", "fromHardProcess"]
    # events = events[0:1000]
    abs_id = abs(events.GenPart.pdgId)
    tops = events.GenPart[abs_id == 6]
    tops = tops[tops.hasFlags(*children_gen_flags)]

    # distinct top quark children (b's and W's)
    tops_children = tops.distinctChildrenDeep
    tops_children = tops_children[tops_children.hasFlags(*children_gen_flags)]
    tops = events.GenPart[abs_id == 6]
    tops = tops[tops.hasFlags(*mother_gen_flags)]

    # sort different decay products
    # making this very ugly:
    bottoms = tops_children[abs(tops_children.pdgId) == 5]
    bottoms = ak.flatten(bottoms, axis=2)
    w_bosons = tops_children[abs(tops_children.pdgId) == 24]
    w_children = w_bosons.distinctChildrenDeep
    w_children = w_children[w_children.hasFlags(*children_gen_flags)]
    w_bosons = ak.flatten(w_bosons, axis=2)
    qq = w_children[w_children.pdgId > 0]
    qq = qq[qq.pdgId < 5]
    qq = ak.flatten(qq, axis=3)
    qq = ak.firsts(qq, axis=2)
    # qq = ak.flatten(qq, axis=2)
    qbarqbar = w_children[w_children.pdgId < 0]
    qbarqbar = qbarqbar[qbarqbar.pdgId > -5]
    qbarqbar = ak.flatten(qbarqbar, axis=3)
    qbarqbar = ak.firsts(qbarqbar, axis=2)
    # qbarqbar = ak.flatten(qbarqbar, axis=2)

    leps = ak.concatenate([
        w_children[w_children.pdgId == 11], w_children[w_children.pdgId == 13], w_children[w_children.pdgId == 15],
    ], axis=3)
    leps = ak.flatten(leps, axis=3)
    leps = ak.firsts(leps, axis=2)

    # leps_mask = ak.num(leps, axis=2) != 0
    # leps= ak.mask(leps, ak.num(leps, axis=2) != 0)  #this one
    # leps = ak.flatten(leps, axis=2)
    antileps = ak.concatenate([
        w_children[w_children.pdgId == -11], w_children[w_children.pdgId == -13], w_children[w_children.pdgId == -15],
    ], axis=3)
    antileps = ak.flatten(antileps, axis=3)
    antileps = ak.firsts(antileps, axis=2)
    # antileps= ak.mask(antileps, ak.num(leps, axis=2) != 0)   #this one
    # antileps = ak.flatten(antileps, axis=2)
    neutrinos = ak.concatenate([
        w_children[w_children.pdgId == 12], w_children[w_children.pdgId == 14], w_children[w_children.pdgId == 16],
    ], axis=3)
    neutrinos = ak.flatten(neutrinos, axis=3)
    neutrinos = ak.firsts(neutrinos, axis=2)
    # neutrinos = ak.flatten(neutrinos, axis=2)
    antineutrinos = ak.concatenate([
        w_children[w_children.pdgId == -12], w_children[w_children.pdgId == -14], w_children[w_children.pdgId == -16],
    ], axis=3)
    antineutrinos = ak.flatten(antineutrinos, axis=3)
    antineutrinos = ak.firsts(antineutrinos, axis=2)

    other_w_children = w_bosons.distinctChildrenDeep[abs(w_bosons.distinctChildrenDeep.pdgId) > 18]

    other_w_children = other_w_children[other_w_children.hasFlags("isFirstCopy")]
    other_w_children = ak.firsts(other_w_children, axis=2)

    #  No real difference
    # other_top_children = tops.distinctChildren[abs(tops.distinctChildren.pdgId) != 5]
    # other_top_children = other_top_children[abs(other_top_children.pdgId) != 24]
    # other_top_children = other_top_children[other_top_children.hasFlags("isFirstCopy")]
    # other_top_children = ak.firsts(other_top_children, axis=2)

    # from IPython import embed; embed(header="debugger")
    # tops = tops[:, None, :],
    # bottoms = bottoms[:, None, :],
    # w_bosons  = w_bosons[:, None, :],
    # qq = qq[:, None, :],
    # qbarqbar  = qbarqbar[:, None, :],
    # leps      = leps[:, None, :],
    # antileps  = antileps[:, None, :],
    # neutrinos = neutrinos[:, None, :],
    # antineutrinos = antineutrinos[:, None, :]

    # build the column
    top_family = ak.concatenate([
        tops[:, None, :],
        bottoms[:, None, :],
        w_bosons[:, None, :],
        qq[:, None, :],
        qbarqbar[:, None, :],
        leps[:, None, :],
        antileps[:, None, :],
        neutrinos[:, None, :],
        antineutrinos[:, None, :],
        other_w_children[:, None, :],
    ], axis=1)

    # save the column
    events = set_ak_column(events, "top_family", top_family)

    events = self[attach_coffea_behavior](events, collections={"top_family": {"type_name": "GenParticle",
        "check_attr": "metric_table", "skip_fields": "*Idx*G"}}, **kwargs)

    # Validation: Reconstruct top mass from invariant mass of decay products
    relevant_children = top_family[:, [1, 3, 4, 5, 6, 7, 8, 9], 0]
    relevant_children_summed = relevant_children.sum(axis=1)
    relevant_children_inv_mass = relevant_children_summed.absolute()

    tops_inv_mass = events.top_family[:, 0, 0].absolute()

    events = set_ak_column(events, "reco_top_mass", relevant_children_inv_mass)
    events = set_ak_column(events, "top_mass", tops_inv_mass)
    # from IPython import embed; embed(header="debugger")
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
