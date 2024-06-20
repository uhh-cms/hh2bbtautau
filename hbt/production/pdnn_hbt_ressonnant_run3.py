# coding: utf-8

"""
Producers for Resonant Network
See https://github.com/uhh-cms/tautauNN/tree/main/tautaunn.
"""
import functools

import law
import enum

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, dev_sandbox, InsertableDict
from columnflow.columnar_util import set_ak_column
from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.util import DotDict

np = maybe_import("numpy")
ak = maybe_import("awkward")
tf = maybe_import("tensorflow")

logger = law.logger.get_logger(__name__)


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


# helper function
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


def select_DAU(events):
    """
    Select DAUs from *events*.
    DAUs are daughter leptons coming from Higgs.

    The lepton selector results in single or dual entries.
    The dual entries are ordered by isolation value:
    One has either 1 or 0 electron or muon (but not together) and 0 to 2 taus.
    """
    # second entry is always tau, first either electron or muon
    daus = ak.concatenate((events.Electron, events.Muon, events.Tau), axis=1)
    # ElectronTau, MuonTau or TauTau
    # sometimes there are more than 2 particles, but network only expects 2
    dau1, dau2 = daus[:, 0], daus[:, 1]
    return dau1, dau2


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
    dau1_decay_mode, dau2_decay_mode = decay_mode[:, 0], decay_mode[:, 1]
    return dau1_decay_mode, dau2_decay_mode


def get_rotated_kinematics(
    ref_phi: ak.Array,
    events: ak.Array,
):
    """
    Rotates a momentum vector extracted from *events* in the transverse plane to a reference phi angle *ref_phi*.
    Returns the rotated px and py components in a 2-tuple.
    """

    def phi_mpi_to_pi(phi: ak.Array):
        """
        Helper function to guarantee that phi stays within [-pi, pi]
        """
        PI = np.pi
        larger_pi_mask = phi > PI
        smaller_pi_mask = phi < -PI
        while ak.any(larger_pi_mask) or ak.any(smaller_pi_mask):
            # TODO check why inplace -=, += is not working
            phi = phi - 2 * PI * larger_pi_mask
            phi = phi + 2 * PI * smaller_pi_mask

            larger_pi_mask = phi > PI
            smaller_pi_mask = phi < -PI
        return phi
    new_phi = phi_mpi_to_pi(events.phi) - ref_phi
    return events.pt * np.cos(new_phi, dtype=np.float32), events.pt * np.sin(new_phi, dtype=np.float32)


def get_px(events: ak.Array):
    # p_x = p_t cos(phi)
    return events.pt * np.cos(events.phi, dtype=np.float32)


def get_py(events: ak.Array):
    # p_y = p_t sin(phi)
    return events.pt * np.sin(events.phi, dtype=np.float32)


def get_pz(events: ak.Array):
    # p_z = p_t sinh(eta)
    return events.pt * np.sinh(events.eta, dtype=np.float32)


def get_e(px: ak.Array, py: ak.Array, pz: ak.Array, mass: ak.Array):
    # E = sqrt(px^2 + py^2 + pz^2 + m^2)
    return np.sqrt(px**2 + py**2 + pz**2 + mass**2)


def mask_network_unknown_embedding(events, raise_only=False):
    """
    Helper to create mask to filter out values not seen by embedding layers.
    Values taken from: https://github.com/uhh-cms/tautauNN/blob/old_setup/train.py#L157-L162
    """
    embedding_expected_inputs = {
        "pair_type": [0, 1, 2],
        "dau1_decay_mode": [-1, 0, 1, 10, 11],  # -1 for e/mu
        "dau2_decay_mode": [0, 1, 10, 11],
        "dau1_charge": [-1, 1],
        "dau2_charge": [-1, 1],
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


@producer(
    uses={
        # custom columns created upstream, probably by a selector
        "channel_id",
        # nano columns
        "event",
        "Jet.{pt,eta,phi,mass}",
        "Jet.btagDeepFlav{B,CvB,CvL}",
        "Jet.btagPNet{B,CvB,CvL,QvG}",
        "Jet.hhbtag",
        "MET.{pt,phi}", "MET.cov{XX,XY,YY}",
        "Electron.{eta,phi,pt,mass,charge}",
        "Tau.{eta,phi,pt,mass,decayMode,charge}",
        "Muon.{eta,phi,pt,mass,charge}",
        "FatJet.{eta,phi,pt,mass}",
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
    produces={
        "ressonant_combined_network_output",
    },
    # kwargs passed to the parametrized network
    mass=250,
    year=3,
    spin=2,
)
def pdnn_hbt_ressonnant_run3(
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

    # prepare events for network
    f = DotDict()

    # continuous inputs
    # DAU variables
    dau1, dau2 = select_DAU(events)

    # compute angle from visible mother particle of dau1 and dau2
    # used to rotate the kinematics of dau{1,2}, met, bjet{1,2} and fatjets relative to it
    f.dau1_px, f.dau1_py = get_px(dau1), get_py(dau1)
    f.dau2_px, f.dau2_py = get_px(dau2), get_py(dau2)

    phi_lep = np.arctan2(
        f.dau1_py + f.dau2_py,
        f.dau1_px + f.dau2_px,
    )

    f.dau1_px, f.dau1_py = get_rotated_kinematics(phi_lep, dau1)
    f.dau1_pz = get_pz(dau1)
    f.dau1_e = get_e(f.dau1_px, f.dau1_py, f.dau1_pz, dau1.mass)

    f.dau2_px, f.dau2_py = get_rotated_kinematics(phi_lep, dau2)
    f.dau2_pz = get_pz(dau2)
    f.dau2_e = get_e(f.dau2_px, f.dau2_py, f.dau2_pz, dau2.mass)

    # bJet variables
    # extract jet1 and jet2 by first and second highest hhbtag score
    score_indices = ak.argsort(events.Jet.hhbtag, axis=1, ascending=False)
    btag_jet1 = events.Jet[score_indices][:, 0]
    btag_jet2 = events.Jet[score_indices][:, 1]

    f.bjet1_px, f.bjet1_py = get_rotated_kinematics(phi_lep, btag_jet1)
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

    # pad_features with padding values for missing values
    pad_values = {
        "bjet1_e": 0.0,
        "bjet1_px": 0.0,
        "bjet1_py": 0.0,
        "bjet1_pz": 0.0,
        "bjet2_e": 0.0,
        "bjet2_px": 0.0,
        "bjet2_py": 0.0,
        "bjet2_pz": 0.0,
        "bjet1_btag_df": - 1.0,
        "bjet1_cvsb": - 1.0,
        "bjet1_cvsl": - 1.0,
        "bjet2_btag_df": - 1.0,
        "bjet2_cvsb": - 1.0,
        "bjet2_cvsl": - 1.0,
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
        f[key] = ak.flatten(ak.fill_none(ak.pad_none(f[key], 1), pad_value))

    # composite particles
    # combine daus
    f.htt_e = f.dau1_e + f.dau2_e
    f.htt_px = f.dau1_px + f.dau2_px
    f.htt_py = f.dau1_py + f.dau2_py
    f.htt_pz = f.dau1_pz + f.dau2_pz

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
    f.dau1_decay_mode, f.dau2_decay_mode = select_DAU_decay_mode(events)
    f.dau1_charge, f.dau2_charge = dau1.charge, dau2.charge

    # has_bjet_pair
    # see: https://github.com/uhh-cms/tautauNN/blob/5c5371bc852f2a5c0a40ce0ecd9ffaa0c360e565/tautaunn/config.py#L632
    f.has_bjet_pair = ak.num(events.Jet) > 1

    # parametrized features
    # fill constant value for mass, year and spin
    f.mass = ak.full_like(f.met_px, self.mass, dtype=np.int32)
    f.year = ak.full_like(f.met_px, Era[self.year], dtype=np.int32)
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
            f.dau1_px, f.dau1_py, f.dau1_pz, f.dau1_e,
            f.dau2_px, f.dau2_py, f.dau2_pz, f.dau2_e,
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
            f.pair_type, f.dau1_decay_mode, f.dau2_decay_mode,
            f.dau1_charge, f.dau2_charge,
            f.is_boosted, f.has_bjet_pair,
            f.year, f.spin,
        ]],
        axis=1,
    )

    # evaluate the model
    model = self.tautauNN_model.signatures["serving_default"]

    # mask events that have bad embedding
    masked_network_inputs = {
        "cat_input": tf.cast(tf.boolean_mask(categorical_inputs, selection_mask), np.int32),
        "cont_input": tf.cast(tf.boolean_mask(continous_inputs, selection_mask), np.float32),
    }
    network_scores = model(**masked_network_inputs)

    # fill the scores with EMPTY_FLOAT for events that were masked
    # place the scores in the correct column
    classification_scores = np.ones((len(events), len(categorical_targets)), dtype=np.float32) * EMPTY_FLOAT
    classification_scores[selection_mask] = network_scores["hbt_ensemble"].numpy()
    events = set_ak_column_f32(events, "ressonant_combined_network_output", classification_scores)
    return events


@pdnn_hbt_ressonnant_run3.requires
def pdnn_hbt_ressonnant_run3_requires(self: Producer, reqs: dict) -> None:
    """
    Add the external files bundle to requirements.
    """
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@pdnn_hbt_ressonnant_run3.setup
def pdnn_hbt_ressonnant_run3_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    """
    Sets up the two tautauNN TF models.
    """
    tf = maybe_import("tensorflow")

    # unpack the external files bundle, create a subdiretory and unpack the tautauNN afs repo in it
    bundle = reqs["external_files"]
    arc = bundle.files.resonnant_combine_network
    repo_dir = bundle.files_dir.child("resonnant_combine_network", type="d")
    arc.load(repo_dir, formatter="tar")

    # also store the version of the external file
    # (could be used to distinguish between model paths in repo)
    self.tautauNN_version = self.config_inst.x.external_files["resonnant_combine_network"][1]
    # save both models (even and odd event numbers)

    with self.task.publish_step("loading combined tautauNN models ..."):
        self.tautauNN_model = tf.saved_model.load(repo_dir.child(
            "hbtres_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_"
            "extended_pair_ED10_LU8x128_CTdense_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03"
            "_YEARy_SPINy_MASSy_RSv6_fi80_lbn_ft_lt20_lr1_LBdefault"
            "_daurot_fatjet_composite_FI0_SDx5").path)
