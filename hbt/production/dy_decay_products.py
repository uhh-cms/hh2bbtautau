# coding: utf-8

"""
Producers that determine the generator-level particles related to a top quark decay.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import attach_coffea_behavior as attach_coffea_behavior_fn

ak = maybe_import("awkward")


@producer(
    uses={"GenPart.*", attach_coffea_behavior},
    produces={"dy_family.*"},
)
def dy_decay_products(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    #TODO: Fix description
    """
    Creates a new ragged column "higgs_family" that stores the tops and their decay products. The structure will be as follows:

    .. code-block:: python

        [
            # event 1
            [
                [t1,t2], [b_t1, b_t2], [W_t1, W_t2], [q_1, q_2], [qbar_1, qbar_2], [l_1, l_2], [lbar_1, lbar_2], [nu_1, nu_2], [nubar_1, nubar_2] 
            ],
            # event 2
            ...
        ],

    where the first entry in each array belongs to the first top, and the second entry in each array belongs to the second top. 
    """

    mother_gen_flags = ["isLastCopy", "fromHardProcess"]
    children_gen_flags = ["isFirstCopy", "fromHardProcess"]

    # find first particles
    first_particles = events.GenPart[events.GenPart.genPartIdxMother == -1]

    #Old stuff
    # direct_quarks = first_particles[abs(first_particles.pdgId) > 0]
    # direct_quarks = direct_quarks[abs(direct_quarks.pdgId) < 7]
    # direct_quarks = direct_quarks[direct_quarks.hasFlags("fromHardProcess")]
    # protons = first_particles[first_particles.pdgId == 2212]
    # proton_quarks = protons.distinctChildren[protons.distinctChildren.hasFlags("fromHardProcess")]
    # proton_quarks = proton_quarks[abs(proton_quarks.pdgId) > 0]
    # proton_quarks = proton_quarks[abs(proton_quarks.pdgId) < 7]
    # proton_quarks = ak.firsts(proton_quarks, axis = 2)
    # gluons = first_particles[first_particles.pdgId == 21]
    # #gluon quarks have no Z as children
    # gluon_quarks = gluons.distinctChildren[gluons.distinctChildren.hasFlags("fromHardProcess")]
    # gluon_quarks = gluons.distinctChildren
    # gluon_quarks = gluon_quarks[abs(gluon_quarks.pdgId) > 0]
    # gluon_quarks = gluon_quarks[abs(gluon_quarks.pdgId) < 7]
    # gluon_quarks = ak.firsts(gluon_quarks, axis = 2)
    # gluon_z0 = gluons.distinctChildren[gluons.distinctChildren.pdgId == 23]
    # gluon_z0 = gluons.distinctChildren[gluons.distinctChildren.hasFlags(*children_gen_flags)]
    # gluon_z0 = ak.firsts(ak.flatten(gluon_z0, axis = 2), axis = 1)

    # mother_quarks = ak.concatenate([
    #     direct_quarks,
    #     proton_quarks,
    #     gluon_quarks
    # ], axis = 1)

    # z0 = mother_quarks.distinctChildren
    # z0 = ak.flatten(z0, axis = 2)
    # z0 = z0[z0.pdgId == 23]
    # z0 = z0[z0.hasFlags(*children_gen_flags)]


    # propagators = ak.concatenate([
    #     z0,
    #     gluon_z0[:,None],
    #     # photons
    # ], axis = 1)

    #Distinguish 3 cases: Z0 is child of particle -1, Z0 is grandchild of particle -1, Z0 does not exist in GenPart list:
    # case 1:
    case1_z0 = first_particles.distinctChildren[first_particles.distinctChildren.pdgId == 23]
    case1_z0 = case1_z0[case1_z0.hasFlags(*children_gen_flags)]
    case1_z0 = ak.firsts(case1_z0, axis = 2)

    # case 2:
    case2_z0 = first_particles.distinctChildren.distinctChildren[first_particles.distinctChildren.distinctChildren.pdgId == 23]
    case2_z0 = case2_z0[case2_z0.hasFlags(*children_gen_flags)]
    case2_z0 = ak.flatten(case2_z0, axis = 3)
    case2_z0 = ak.firsts(case2_z0, axis = 2)

    # add them together
    z0s = ak.concatenate([case1_z0, case2_z0], axis = 1)
    z0s = ak.drop_none(z0s)
    # z0s = ak.firsts(ak.drop_none(z0s))

    # case 3:
    #TODO: first copy or children_gen_flags?
    no_z0_mask = ak.num(z0s, axis = 1) == 0
        #find leps that are children or grandchildren of particle -1
    direct_leps = ak.mask(first_particles, no_z0_mask).distinctChildren
    direct_leps = direct_leps[abs(direct_leps.pdgId) >= 11]
    direct_leps = direct_leps[abs(direct_leps.pdgId) <= 16]
    # direct_leps = direct_leps[direct_leps.hasFlags("isFirstCopy")]
    direct_leps = direct_leps[direct_leps.hasFlags(*children_gen_flags)]
    direct_leps = ak.flatten(direct_leps, axis = 2)

    indirect_leps = ak.mask(first_particles, no_z0_mask).distinctChildren.distinctChildren
    indirect_leps = indirect_leps[abs(indirect_leps.pdgId) >= 11]
    indirect_leps = indirect_leps[abs(indirect_leps.pdgId) <= 16]
    # indirect_leps = indirect_leps[indirect_leps.hasFlags("isFirstCopy")]
    indirect_leps = indirect_leps[indirect_leps.hasFlags(*children_gen_flags)]
    indirect_leps = ak.flatten(ak.flatten(indirect_leps, axis = 3), axis = 2)

    #for debugging
    # no_z0_events = events.GenPart[no_z0_mask]
    # direct_leps = no_z0_events.distinctChildren
    # direct_leps = direct_leps[abs(direct_leps.pdgId) >= 11]
    # direct_leps = direct_leps[abs(direct_leps.pdgId) <= 16]
    # direct_leps = direct_leps[direct_leps.hasFlags(*children_gen_flags)]
    # direct_leps = ak.flatten(direct_leps, axis = 2)

    # indirect_leps = no_z0_events.distinctChildren.distinctChildren
    # indirect_leps = indirect_leps[abs(indirect_leps.pdgId) >= 11]
    # indirect_leps = indirect_leps[abs(indirect_leps.pdgId) <= 16]
    # indirect_leps = indirect_leps[indirect_leps.hasFlags(*children_gen_flags)]
    # indirect_leps = ak.flatten(ak.flatten(indirect_leps, axis = 3), axis = 2)

    case_3_leps = ak.concatenate([
        direct_leps,
        indirect_leps], axis = 1
    )
    # no_case_3_leps_mask = ak.num(case_3_leps) == 0


    # build missing z0s for case 3
    #TODO: add missing fields / make genparticle (?)
    case_3_leps = attach_coffea_behavior_fn(case_3_leps, collections = {"GenPart":{"type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G"}})
    case_3_lep_mask = ak.num(ak.drop_none(case_3_leps), axis = 1) > 0
    built_z0s = ak.mask(case_3_leps, case_3_lep_mask).sum(axis = 1)
    z0_mask = ak.fill_none(built_z0s.mass, -10000000) != -10000000
    built_z0s = ak.mask(built_z0s, z0_mask)

    #construct dy family column
    #TODO sort by charges
    leptons = z0s.distinctChildrenDeep[z0s.distinctChildrenDeep.hasFlags("isFirstCopy")]
    leptons = ak.flatten(leptons, axis = 2)
    leptons = leptons[abs(leptons.pdgId) >= 11]
    leptons = leptons[abs(leptons.pdgId) <= 16]
    leptons = ak.concatenate([

        leptons,
        case_3_leps], axis = 1)

    #find tau children
    tau_mask = abs(leptons.pdgId) == 15
    tau_children = ak.mask(leptons, tau_mask).distinctChildrenDeep[ak.mask(leptons, tau_mask).distinctChildrenDeep.hasFlags("isFirstCopy")]
    tau_children = ak.firsts(tau_children, axis = 1)

    leptons = ak.concatenate([
        leptons[:,None,:],
        tau_children[:,None,:]], axis = 1)

    leptons = ak.flatten(leptons, axis = 2)

    #build z0s for case 3
    case_3_lep_mask = ak.num(ak.drop_none(case_3_leps), axis = 1) > 0
    leptons = attach_coffea_behavior_fn(leptons, collections = {"GenPart":{"type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G"}})
    built_z0s = ak.mask(leptons, case_3_lep_mask).sum(axis = 1)


    no_built_z0_mask = ak.fill_none(built_z0s.mass, -10000000) == -10000000
    z0s = ak.where(no_built_z0_mask, z0s, built_z0s[:,None])

    # z0s["pt"] = ak.where
    #attach coffea behaviour to z0s
    from IPython import embed; embed(header="debugger")

    dy_family = ak.concatenate([
        z0s[:,None,:],
        leptons[:,None,:]
    ],axis = 1)


    # save the column
    events = set_ak_column(events, "dy_family", dy_family)
    # z0_leps_attached = attach_coffea_behavior_fn(z0_leps, collections = {"GenPart":{"type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G"}})
    # inv_mass_z0_leps = z0_leps_attached.sum(axis = 1).absolute()
    # events = set_ak_column(events, "inv_mass_z0_leps", inv_mass_z0_leps)


    # quarks_attached = attach_coffea_behavior_fn(mother_quarks, collections = {"GenPart":{"type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G"}})
    # inv_mass_quarks = quarks_attached.sum(axis=1).absolute()
    # events = set_ak_column(events, "inv_mass_quarks", inv_mass_quarks)
    return events



#TODO: Add this
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
