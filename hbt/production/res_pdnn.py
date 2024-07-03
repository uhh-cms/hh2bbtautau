# coding: utf-8

"""
Producer for evaluating the pDNN developed for the resonant run 2 analysis.
See https://github.com/uhh-cms/tautauNN
"""

from __future__ import annotations

import functools
import enum

import law

from columnflow.production import Producer, producer
from columnflow.production.util import attach_coffea_behavior, default_collections
from columnflow.columnar_util import set_ak_column, attach_behavior, EMPTY_FLOAT
from columnflow.util import maybe_import, dev_sandbox, InsertableDict, DotDict

np = maybe_import("numpy")
ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)


# helper function
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


class Era(enum.Enum):
    """
    Enum for the different eras.
    Everything above 2018 is considered 2018, since the network is only trained for Run2.
    """
    e2016APV = 0
    e2016 = 1
    e2017 = 2
    e2018 = 3
    e2022 = e2018
    e2023 = e2018
    e2024 = e2018


@producer(
    uses={
        attach_coffea_behavior,
        # custom columns created upstream, probably by a selector
        "channel_id",
        # nano columns
        "event",
        "Tau.{eta,phi,pt,mass,charge,decayMode}",
        "Electron.{eta,phi,pt,mass,charge}",
        "Muon.{eta,phi,pt,mass,charge}",
        "HHBJet.{pt,eta,phi,mass,hhbtag}", "HHBJet.btagDeepFlav{B,CvB,CvL}", "HHBJet.btagPNet{B,CvB,CvL,QvG}",
        "MET.{pt,phi}", "MET.cov{XX,XY,YY}",
        "FatJet.{eta,phi,pt,mass}",
    },
    # produced columns are added in the deferred init below
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
    # kwargs passed to the parametrized network
    mass=500,
    spin=2,
)
def res_pdnn(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    The producer for the combined multiclass classfifer network used by HBT resonnant analysis.
    The output scores are classifying if incoming events are HH, Drell-Yan or ttbar.
    The network uses continous, categorical and parametrized inputs.
    The network is parametrized in *year*, *spin*, *mass*.
    Since the network was trained on Run2 data, Run3 uses 2018 as parameter.

    A list of all inputs in the correct order can be found in the tautauNN repo:
    https://github.com/uhh-cms/tautauNN/blob/f1ca194/evaluation/interface.py#L67
    """
    tf = maybe_import("tensorflow")

    # ensure coffea behavior
    events = self[attach_coffea_behavior](
        events,
        collections={"HHBJet": default_collections["Jet"]},
        **kwargs,
    )

    # prepare events for network
    f = DotDict()

    # get visible tau decay products, consider them all as tau types
    vis_taus = attach_behavior(
        ak.concatenate((events.Electron, events.Muon, events.Tau), axis=1),
        "Tau",
    )
    vis_tau1, vis_tau2 = vis_taus[:, 0], vis_taus[:, 1]

    # compute angle from visible mother particle of vis_tau1 and vis_tau2
    # used to rotate the kinematics of dau{1,2}, met, bjet{1,2} and fatjets relative to it
    phi_lep = np.arctan2(vis_tau1.py + vis_tau2.py, vis_tau1.px + vis_tau2.px)

    # lepton 1
    f.vis_tau1_px, f.vis_tau1_py = get_rotated_kinematics(phi_lep, vis_tau1.px, vis_tau1.py)
    f.vis_tau1_pz = vis_tau1.pz
    f.vis_tau1_e = vis_tau1.energy

    # lepton 2
    f.vis_tau2_px, f.vis_tau2_py = get_rotated_kinematics(phi_lep, vis_tau2.px, vis_tau2.py)
    f.vis_tau2_pz = vis_tau2.pz
    f.vis_tau2_e = vis_tau2.energy

    from IPython import embed; embed(header="ipython debugger")

    # bJet variables
    btag_jet1 = events.Jet[score_indices][:, 0]
    btag_jet2 = events.Jet[score_indices][:, 1]

    f.bjet1_px, f.bjet1_py = get_rotated_kinematics(phi_lep, events.HHBJet)
    f.bjet1_pz = get_pz(btag_jet1)
    f.bjet1_e = get_e(f.bjet1_px, f.bjet1_py, f.bjet1_pz, btag_jet1.mass)

    f.bjet2_px, f.bjet2_py = get_rotated_kinematics(phi_lep, btag_jet2)
    f.bjet2_pz = get_pz(btag_jet2)
    f.bjet2_e = get_e(f.bjet2_px, f.bjet2_py, f.bjet2_pz, btag_jet2.mass)

    # bJet particle net scores
    f.bjet1_btag_df = btag_jet1.btagDeepFlavB
    f.bjet1_cvsb = btag_jet1.btagDeepFlavCvB
    f.bjet1_cvsl = btag_jet1.btagDeepFlavCvL
    f.bjet1_hhbtag = btag_jet1.hhbtag

    f.bjet2_btag_df = btag_jet2.btagDeepFlavB
    f.bjet2_cvsb = btag_jet2.btagDeepFlavCvB
    f.bjet2_cvsl = btag_jet2.btagDeepFlavCvL
    f.bjet2_hhbtag = btag_jet2.hhbtag

    # fatjet variables
    f.fatjet_px, f.fatjet_py = get_rotated_kinematics(phi_lep, events.FatJet)
    f.fatjet_pz = get_pz(events.FatJet)
    f.fatjet_e = get_e(f.fatjet_px, f.fatjet_py, f.fatjet_pz, events.FatJet.mass)

    # pad features with padding values for missing values
    pad_values = {
        "bjet1_e": 0.0,
        "bjet1_px": 0.0,
        "bjet1_py": 0.0,
        "bjet1_pz": 0.0,
        "bjet2_e": 0.0,
        "bjet2_px": 0.0,
        "bjet2_py": 0.0,
        "bjet2_pz": 0.0,
        "bjet1_btag_df": -1.0,
        "bjet1_cvsb": -1.0,
        "bjet1_cvsl": -1.0,
        "bjet2_btag_df": -1.0,
        "bjet2_cvsb": -1.0,
        "bjet2_cvsl": -1.0,
        "fatjet_e": 0.0,
        "fatjet_px": 0.0,
        "fatjet_py": 0.0,
        "fatjet_pz": 0.0,
    }

    for key, pad_value in pad_values.items():
        # skip flat arrays
        if f[key].ndim == 1:
            continue

        # fill nones with pad_value and flat the resulting array to make it 1 dimensional
        f[key] = ak.flatten(ak.fill_none(ak.pad_none(f[key], axis=1), pad_value))

    # composite particles
    # combine daus
    f.htt_e = f.vis_tau1_e + f.vis_tau2_e
    f.htt_px = f.vis_tau1_px + f.vis_tau2_px
    f.htt_py = f.vis_tau1_py + f.vis_tau2_py
    f.htt_pz = f.vis_tau1_pz + f.vis_tau2_pz

    # combine bjets
    f.hbb_e = f.bjet1_e + f.bjet2_e
    f.hbb_px = f.bjet1_px + f.bjet2_px
    f.hbb_py = f.bjet1_py + f.bjet2_py
    f.hbb_pz = f.bjet1_pz + f.bjet2_pz

    # htt + hbb
    f.htthbb_e = f.htt_e + f.hbb_e
    f.htthbb_px = f.htt_px + f.hbb_px
    f.htthbb_py = f.htt_py + f.hbb_py
    f.htthbb_pz = f.htt_pz + f.hbb_pz

    # htt + fatjet
    f.httfatjet_e = f.htt_e + f.fatjet_e
    f.httfatjet_px = f.htt_px + f.fatjet_px
    f.httfatjet_py = f.htt_py + f.fatjet_py
    f.httfatjet_pz = f.htt_pz + f.fatjet_pz

    # MET variable
    f.met_px, f.met_py = get_rotated_kinematics(phi_lep, events.MET)
    f.met_cov00 = events.MET.covXX
    f.met_cov01 = events.MET.covXY
    f.met_cov11 = events.MET.covYY

    # categorical inputs
    # pair_type
    # reduce channel_id by one to match clubanalysis values: 0,1,2
    f.pair_type = ak.values_astype(events.channel_id - 1, np.int32)

    # is_boosted if 1 ak8 (fatjet) jet is present
    # see: https://github.com/LLRCMS/KLUBAnalysis/blob/master/test/skimNtuple_HHbtag.cpp#L4797-L4799
    f.is_boosted = ak.num(events.FatJet) > 0

    # dau decay mode & charge
    f.vis_tau1_decay_mode, f.vis_tau2_decay_mode = select_DAU_decay_mode(events)
    f.vis_tau1_charge, f.vis_tau2_charge = vis_tau1.charge, vis_tau2.charge

    # has_bjet_pair
    # see: https://github.com/uhh-cms/tautauNN/blob/5c5371bc852f2a5c0a40ce0ecd9ffaa0c360e565/tautaunn/config.py#L632
    f.has_bjet_pair = ak.num(events.Jet) > 1

    # parametrized features
    # fill constant value for mass, year and spin
    f.mass = ak.full_like(f.met_px, self.mass, dtype=np.int32)
    f.year = ak.full_like(f.met_px, Era[self.year].value, dtype=np.int32)
    f.spin = ak.full_like(f.met_px, self.spin, dtype=np.int32)

    # filter out events with bad embedding, if raise_only=True only log the events, but do not mask
    selection_mask = mask_network_unknown_embedding(f, raise_only=False)

    # convert the networks input into a tensorflow tensor and evaluate the model
    # taken from https://github.com/uhh-cms/tautauNN/blob/f1ca194/evaluation/interface.py#L159
    categorical_targets = [
        "HH", "Drell-Yan", "ttbar",
    ]

    continous_inputs = tf.concat(
        [t[..., None] for t in [
            f.met_px, f.met_py, f.met_cov00, f.met_cov01, f.met_cov11,
            f.vis_tau1_px, f.vis_tau1_py, f.vis_tau1_pz, f.vis_tau1_e,
            f.vis_tau2_px, f.vis_tau2_py, f.vis_tau2_pz, f.vis_tau2_e,
            f.bjet1_px, f.bjet1_py, f.bjet1_pz, f.bjet1_e, f.bjet1_btag_df, f.bjet1_cvsb, f.bjet1_cvsl, f.bjet1_hhbtag,  # noqa
            f.bjet2_px, f.bjet2_py, f.bjet2_pz, f.bjet2_e, f.bjet2_btag_df, f.bjet2_cvsb, f.bjet2_cvsl, f.bjet2_hhbtag,  # noqa
            f.fatjet_px, f.fatjet_py, f.fatjet_pz, f.fatjet_e,
            f.htt_e, f.htt_px, f.htt_py, f.htt_pz,
            f.hbb_e, f.hbb_px, f.hbb_py, f.hbb_pz,
            f.htthbb_e, f.htthbb_px, f.htthbb_py, f.htthbb_pz,
            f.httfatjet_e, f.httfatjet_px, f.httfatjet_py, f.httfatjet_pz,
            f.mass,
        ]],
        axis=1,
    )

    categorical_inputs = tf.concat(
        [t[..., None] for t in [
            f.pair_type, f.vis_tau1_decay_mode, f.vis_tau2_decay_mode,
            f.vis_tau1_charge, f.vis_tau2_charge,
            f.is_boosted, f.has_bjet_pair,
            f.year, f.spin,
        ]],
        axis=1,
    )

    # mask events that have bad embedding
    masked_network_inputs = {
        "cat_input": tf.cast(tf.boolean_mask(categorical_inputs, selection_mask), np.int32),
        "cont_input": tf.cast(tf.boolean_mask(continous_inputs, selection_mask), np.float32),
    }
    network_scores = self.res_pdnn_model(**masked_network_inputs)

    # fill the scores with EMPTY_FLOAT for events that were masked
    # place the scores in the correct column
    classification_scores = np.ones((len(events), len(categorical_targets)), dtype=np.float32) * EMPTY_FLOAT
    classification_scores[selection_mask] = network_scores["hbt_ensemble"].numpy()
    events = set_ak_column_f32(events, "ressonant_combined_network_output", classification_scores)
    return events


@res_pdnn.init
def res_pdnn_init(self: Producer) -> None:
    self.produces |= {
        f"res_pdnn_s{self.spin}_m{self.mass}_{proc}"
        for proc in ["hh", "tt", "dy"]
    }


@res_pdnn.requires
def res_pdnn_requires(self: Producer, reqs: dict) -> None:
    """
    Add the external files bundle to requirements.
    """
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@res_pdnn.setup
def res_pdnn_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    """
    Sets up the two tautauNN TF models.
    """
    tf = maybe_import("tensorflow")

    # unpack the model archive
    bundle = reqs["external_files"]
    bundle.files
    model_dir = bundle.files_dir.child("res_pdnn_model", type="d")
    bundle.files.res_pdnn.load(model_dir, formatter="tar")

    # load the model
    with self.task.publish_step("loading resonant pDNN model ..."):
        saved_model = tf.saved_model.load(model_dir.child("model_fold0").abspath)
        self.res_pdnn_model = saved_model.signatures["serving_default"]


def select_DAU_decay_mode(events):
    """
    Helper function to select decay modes of tau as done in club analysis.
    Expected values are -1, 0, 1, 10, 11.
    - 0 for 1-prong
    - 1 for 3-prong
    - 10 for 1-prong+pi0
    - 11 for 3-prong+pi0
    - -1 is used for e/mu
    - 2 is mapped to 1 (reason unknown, just KLUB things)
    """
    tau_decay = events.Tau.decayMode

    # set 2 to 1
    tau_decay = ak.where(tau_decay == 2,
        np.full((len(tau_decay)), 1, dtype=np.int32),
        tau_decay)

    # convert mask to array filled with -1
    non_tau_mask = ak.num(tau_decay) < 2
    leptons_without_tau = ak.unflatten(
        (ak.mask((non_tau_mask * np.int32(-1)), non_tau_mask)),
        counts=1)

    decay_mode = ak.drop_none(ak.concatenate((leptons_without_tau, tau_decay), axis=1))
    vis_tau1_decay_mode, vis_tau2_decay_mode = decay_mode[:, 0], decay_mode[:, 1]
    return vis_tau1_decay_mode, vis_tau2_decay_mode


def get_rotated_kinematics(ref_phi: ak.Array, px: ak.Array, py: ak.Array) -> ak.Array:
    """
    Rotates a momentum vector extracted from *events* in the transverse plane to a reference phi
    angle *ref_phi*. Returns the rotated px and py components in a 2-tuple.
    """
    new_phi = np.arctan2(py, px) - ref_phi
    pt = (px**2 + py**2)**0.5
    return pt * np.cos(new_phi), pt * np.sin(new_phi)


def mask_network_unknown_embedding(events, raise_only=False):
    """
    Helper to create mask to filter out values not seen by embedding layers.
    Values taken from: https://github.com/uhh-cms/tautauNN/blob/old_setup/train.py#L157-L162
    """
    embedding_expected_inputs = {
        "pair_type": [0, 1, 2],
        "vis_tau1_decay_mode": [-1, 0, 1, 10, 11],  # -1 for e/mu
        "vis_tau2_decay_mode": [0, 1, 10, 11],
        "vis_tau1_charge": [-1, 1],
        "vis_tau2_charge": [-1, 1],
        "spin": [0, 2],
        "year": [0, 1, 2, 3],
        "is_boosted": [0, 1],
        "has_bjet_pair": [0, 1],
    }

    # get mask for each embedding input, combine them to event mask and return the indices
    masks = []
    for feature, expected_values in embedding_expected_inputs.items():
        # skip rest if no differences exist in unique values
        unique_different_values = np.setdiff1d(events[feature], expected_values)
        if len(unique_different_values) == 0:
            continue

        logger.info(
            f"{feature} has  unknown values to embeding: {unique_different_values}"
            f"expected only {expected_values}.",
        )
        if raise_only:
            continue
        # get event, gather masks to create event level mask
        event_of_feature = events[feature]

        mask_for_expected_feature = ak.any([event_of_feature == value
            for value in expected_values],
            axis=0)

        masks.append(mask_for_expected_feature)

    if not raise_only and (len(masks) > 0):
        logger.info(f"Wrong embedings are masked and network scores is set to {EMPTY_FLOAT}")
    event_mask = ak.all(masks, axis=0)
    return event_mask
