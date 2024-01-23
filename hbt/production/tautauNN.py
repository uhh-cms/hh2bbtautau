# coding: utf-8

"""
Producers for Tobias Kramer Neural Network.
See https://github.com/uhh-cms/tautauNN/tree/main/tautaunn.
"""

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, dev_sandbox, InsertableDict


np = maybe_import("numpy")
ak = maybe_import("awkward")
tf = maybe_import("tensorflow")


logger = law.logger.get_logger(__name__)


@producer(
    uses={
        # custom columns created upstream, probably by a selector
        "channel_id",
        # nano columns
        "event",
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        "Jet.btagDeepFlavB", "Jet.btagDeepFlavCvB",
        "Jet.hhbtag",
        "MET.pt", "MET.phi",
        "MET.covXX", "MET.covXY", "MET.covYY",
        "Electron.*", "Tau.*", "Muon.*",
        "DeepMETResolutionTune.phi", "DeepMETResolutionTune.pt",
        "DeepMETResponseTune.phi", "DeepMETResponseTune.pt",
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
    produces={
        "tautauNN_score",
    },
)
def tautauNN(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    inputs = {}
    # continous input features
    # MET variable
    inputs["met_e"] = events.MET.pt
    inputs["met_cov00"] = events.MET.covXX
    inputs["met_cov01"] = events.MET.covXY
    inputs["met_cov11"] = events.MET.covYY

    # DAU variables
    # channel_id decode:1=muon_tau, 2= electron_tau, 3=tau_tau
    # DAU helper
    dau = create_DAU(events)
    dau_dphi = phi_mpi_to_pi(dau.phi - events.MET.phi)

    # DAU features
    dau_px = dau.pt * np.cos(dau_dphi, dtype=np.float32)
    dau_py = dau.pt * np.sin(dau_dphi, dtype=np.float32)
    dau_pz = dau.pt * np.sinh(dau.eta, dtype=np.float32)
    dau_e = np.sqrt(dau_px**2 + dau_py**2 + dau_pz**2 + dau.mass**2, dtype=np.float32)

    # decay modes
    dau_decay_mode = create_DAU_decay_mode(events)

    # split into first and second daughter
    for number in (0, 1):
        inputs[f"dau{number + 1}_px"] = dau_px[:, number]
        inputs[f"dau{number + 1}_py"] = dau_py[:, number]
        inputs[f"dau{number + 1}_pz"] = dau_pz[:, number]
        inputs[f"dau{number + 1}_e"] = dau_e[:, number]

        inputs[f"dau{number + 1}_charge"] = dau.charge[:, number]
        inputs[f"dau{number + 1}_decayMode"] = dau_decay_mode[:, number]

    # prepare bjet inputs for network
    # get jet indices to extract out bjet1, bjet2
    score_indices = ak.argsort(events.Jet.hhbtag, axis=1, ascending=False)
    # filter out first 2 bjets and extract features
    btag_jets = events.Jet[score_indices][:, 0:2]

    # bjet helper features
    bjet_dphi = phi_mpi_to_pi(btag_jets.phi - events.DeepMETResponseTune.phi)

    # bjet features
    bjet_px = btag_jets.pt * np.cos(bjet_dphi, dtype=np.float32)
    bjet_py = btag_jets.pt * np.sin(bjet_dphi, dtype=np.float32)
    bjet_pz = btag_jets.pt * np.sinh(btag_jets.eta, dtype=np.float32)
    bjet_e = np.sqrt(bjet_px**2 + bjet_py**2 + bjet_pz**2 + btag_jets.mass**2, dtype=np.float32)

    # bjet particle net scores
    btag_deepflavor_b = btag_jets.btagDeepFlavB
    btag_deepflavor_c = btag_jets.btagDeepFlavCvB * btag_jets.btagDeepFlavB

    # split into first and second bjet
    for number in (0, 1):
        inputs[f"bjet{number + 1}_px"] = bjet_px[:, number]
        inputs[f"bjet{number + 1}_py"] = bjet_py[:, number]
        inputs[f"bjet{number + 1}_pz"] = bjet_pz[:, number]
        inputs[f"bjet{number + 1}_e"] = bjet_px[:, number]
        inputs[f"bjet{number + 1}_btag_deepFlavor"] = btag_deepflavor_b[:, number]
        inputs[f"bjet{number + 1}_cID_deepFlavor"] = btag_deepflavor_c[:, number]

    # reduce channel_id by one to match clubanalysis values: 0,1,2
    inputs["pairType"] = ak.values_astype(events.channel_id - 1, np.float32)

    # mass, year and spin are derived from outside and are constants
    self.mass = 200
    self.year = 2
    self.spin = 2

    inputs["mass"] = ak.full_like(inputs["met_e"], self.mass, dtype=np.float32)
    inputs["year"] = ak.full_like(inputs["met_e"], self.year, dtype=np.float32)
    inputs["spin"] = ak.full_like(inputs["met_e"], self.spin, dtype=np.float32)

    # neural network inputs
    continous_input_features = [
        "met_e", "met_cov00", "met_cov01", "met_cov11",
        "dau1_px", "dau1_py", "dau1_pz", "dau1_e",
        "dau2_px", "dau2_py", "dau2_pz", "dau2_e",
        "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e",
        "bjet1_btag_deepFlavor", "bjet1_cID_deepFlavor",
        "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e",
        "bjet2_btag_deepFlavor", "bjet2_cID_deepFlavor",
        "mass",
    ]

    categorical_input_features = [
        "pairType", "dau1_decayMode", "dau2_decayMode", "dau1_charge", "dau2_charge", "year", "spin",
    ]

    # create tensorflow tensor
    continous_input_tensor = tf.stack(
        [tf.convert_to_tensor(inputs[feature], dtype=np.float32)
            for feature in continous_input_features],
        axis=1,
        name="cont_input")
    categorical_input_tensor = tf.stack(
        [tf.convert_to_tensor(inputs[feature], dtype=np.int32)
            for feature in categorical_input_features],
        axis=1,
        name="cat_input")

    m = self.tautauNN_model.signatures["serving_default"]
    p = {
        "cat_input": categorical_input_tensor[5:10],
        "cont_input": continous_input_tensor[5:10],
    }
    # use the network
    from IPython import embed
    globals().update(locals())
    embed()


@tautauNN.requires
def tautauNN_requires(self: Producer, reqs: dict) -> None:
    """
    Add the external files bundle to requirements.
    """
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@tautauNN.setup
def tautauNN_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    """
    Sets up the two tautauNN TF models.
    """

    # unpack the external files bundle, create a subdiretory and unpack the tautauNN afs repo in it
    bundle = reqs["external_files"]
    arc = bundle.files.tautauNN_regression_model
    repo_dir = bundle.files_dir.child("tautauNN_regression_model", type="d")
    arc.load(repo_dir, formatter="tar")

    # also store the version of the external file
    # (could be used to distinguish between model paths in repo)
    self.tautauNN_version = self.config_inst.x.external_files["tautauNN_regression_model"][1]
    # save both models (even and odd event numbers)

    with self.task.publish_step("loading tautauNN models ..."):
        self.tautauNN_model = tf.saved_model.load(repo_dir.child(
            "ttreg_ED5_LU2x_9x128_CTfcn_ACTelu_BNy_LT50_DO0_BS4096_"
            "OPadam_LR3.0e-03_YEARy_SPINy_MASSy_FI0_SD1_reduced_features").path)
        # "reg_mass_class_l2n400_addCharge_incrMassLoss_lossSum_allMasses_even").path)
        # self.tautauNN_model_even = tf.saved_model.load(repo_dir.child(
        #     "reg_mass_class_l2n400_addCharge_incrMassLoss_lossSum_allMasses_even").path)
        # self.tautauNN_model_odd = tf.saved_model.load(repo_dir.child(
        #     "reg_mass_class_l2n400_addCharge_incrMassLoss_lossSum_allMasses_odd").path)


def phi_mpi_to_pi(phi):
    # helper function to guarantee that phi stays within [-pi, pi]
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


def create_DAU(events):
    """
    Helper function to create DAU variable.
    DAUs are daughter lepton coming from Higgs.
    The selection of electron, mkuon and taus always result in single or dual entries.
    Therefore, concatenate of single leptons with another single lepton always will result in
    a dual-lepton entry.
    There are three different dau-types: ElectronTau, MuonTau and TauTau.
    """

    # combine single-lepton arrays to create dual-lepton entries
    # filter dual at least 2 taus out
    tau_tau_mask = ak.num(events.Tau) > 1
    tau_tau = ak.mask(events.Tau, tau_tau_mask)

    # filter single taus and single electron and muon out
    single_tau_mask = ak.num(events.Tau) == 1
    tau = ak.mask(events.Tau, single_tau_mask)

    electron_tau = ak.concatenate([events.Electron, tau], axis=1)
    electron_tau_mask = ak.num(electron_tau) == 2
    electron_tau = ak.mask(electron_tau, electron_tau_mask)

    muon_tau = ak.concatenate([events.Muon, tau], axis=1)
    muon_tau_mask = ak.num(muon_tau) == 2
    muon_tau = ak.mask(muon_tau, muon_tau_mask)

    # combine different dual-lepton arrays together
    # order is preserved, since previous masking left Nones, where otherwise an entry would be
    # thus no dual-lepton entry is stacked on top of another dual-lepton
    dau = ak.drop_none(ak.concatenate((electron_tau, muon_tau, tau_tau), axis=1))
    return dau


def create_DAU_decay_mode(events):
    """
    Helper function to give leptons an decay mode of -1
    """
    tau_decay = events.Tau.decayMode

    # convert mask to array filled with -1
    non_tau_mask = ak.num(tau_decay) < 2
    leptons_without_tau = ak.unflatten(
        (ak.mask((non_tau_mask * np.int32(-1)), non_tau_mask)),
        counts=1)
    return ak.drop_none(ak.concatenate((leptons_without_tau, tau_decay), axis=1))
