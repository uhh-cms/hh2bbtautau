# coding: utf-8

"""
Producers for Tobias Kramer Neural Network.
See https://github.com/uhh-cms/tautauNN/tree/main/tautaunn.
"""
import functools

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, dev_sandbox, InsertableDict
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")
tf = maybe_import("tensorflow")

logger = law.logger.get_logger(__name__)

# helper function
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


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

    # neural network inputs in the correct order
    # defined at https://github.com/uhh-cms/tautauNN/blob/old_setup/train.py#L77
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

    # prepare events for network

    inputs = {}
    # get DAU variables

    # a DAU is a lepton pair, coming from Higgs
    # What particle is within the pair is encoded within the  channel_id
    # decode:1=muon_tau, 2= electron_tau, 3=tau_tau

    dau = select_DAU(events)

    # rotate DAU relative to MET
    dau_dphi = phi_mpi_to_pi(dau.phi - events.MET.phi)

    # DAU kinematics
    dau_px = dau.pt * np.cos(dau_dphi, dtype=np.float32)
    dau_py = dau.pt * np.sin(dau_dphi, dtype=np.float32)
    dau_pz = dau.pt * np.sinh(dau.eta, dtype=np.float32)
    dau_e = np.sqrt(dau_px**2 + dau_py**2 + dau_pz**2 + dau.mass**2, dtype=np.float32)

    # helper function to set decay mode of leptons to -1
    dau_decay_mode = select_DAU_decay_mode(events)

    # split DAU pair features into first and second DAU feature
    for number in (0, 1):
        inputs[f"dau{number + 1}_px"] = dau_px[:, number]
        inputs[f"dau{number + 1}_py"] = dau_py[:, number]
        inputs[f"dau{number + 1}_pz"] = dau_pz[:, number]
        inputs[f"dau{number + 1}_e"] = dau_e[:, number]

        inputs[f"dau{number + 1}_charge"] = dau.charge[:, number]
        inputs[f"dau{number + 1}_decayMode"] = dau_decay_mode[:, number]

    # reduce channel_id by one to match clubanalysis values: 0,1,2
    inputs["pairType"] = ak.values_astype(events.channel_id - 1, np.float32)

    # check if embedding values of daus are respected
    # TODO will be removed if selector takes care of this!
    check_network_embedding(inputs)

    # get MET variable
    inputs["met_e"] = events.MET.pt
    inputs["met_cov00"] = events.MET.covXX
    inputs["met_cov01"] = events.MET.covXY
    inputs["met_cov11"] = events.MET.covYY

    # get BJET variables

    # jet1 and jet2 are defined by first and second highest hhbtag score
    score_indices = ak.argsort(events.Jet.hhbtag, axis=1, ascending=False)
    # filter out first 2 bjets and extract features
    btag_jets = events.Jet[score_indices][:, 0:2]

    # rotate bJet relative to DeepMETResponseTune
    bjet_dphi = phi_mpi_to_pi(btag_jets.phi - events.DeepMETResponseTune.phi)

    # bJet features
    bjet_px = btag_jets.pt * np.cos(bjet_dphi, dtype=np.float32)
    bjet_py = btag_jets.pt * np.sin(bjet_dphi, dtype=np.float32)
    bjet_pz = btag_jets.pt * np.sinh(btag_jets.eta, dtype=np.float32)
    bjet_e = np.sqrt(bjet_px**2 + bjet_py**2 + bjet_pz**2 + btag_jets.mass**2, dtype=np.float32)

    # get bJet particle net scores
    btag_deepflavor_b = btag_jets.btagDeepFlavB
    btag_deepflavor_c = btag_jets.btagDeepFlavCvB * btag_jets.btagDeepFlavB

    # split into first and second bJet
    for number in (0, 1):
        inputs[f"bjet{number + 1}_px"] = bjet_px[:, number]
        inputs[f"bjet{number + 1}_py"] = bjet_py[:, number]
        inputs[f"bjet{number + 1}_pz"] = bjet_pz[:, number]
        inputs[f"bjet{number + 1}_e"] = bjet_e[:, number]
        inputs[f"bjet{number + 1}_btag_deepFlavor"] = btag_deepflavor_b[:, number]
        inputs[f"bjet{number + 1}_cID_deepFlavor"] = btag_deepflavor_c[:, number]

    # get user input information about year, spin and mass (parametrized network)
    # do also some checks for logic reasons
    allowed_year = (2016, 2017, 2018)
    allowed_spin = (2)
    if self.mass < 0:
        raise ValueError(f"Mass must be positive, but got {self.mass}")
    inputs["mass"] = ak.full_like(inputs["met_e"], self.mass, dtype=np.float32)

    if self.year not in allowed_year:
        raise ValueError(f"Year must be one of {allowed_year}, but got {self.year}")
    inputs["year"] = ak.full_like(inputs["met_e"], self.year, dtype=np.float32)

    if self.spin not in allowed_spin:
        raise ValueError(f"Spin must be one of {allowed_spin}, but got {self.spin}")
    inputs["spin"] = ak.full_like(inputs["met_e"], self.spin, dtype=np.float32)

    # convert everything into tensorflow tensors
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

    # use the network
    model = self.tautauNN_model.signatures["serving_default"]

    final_network_inputs = {
        "cat_input": categorical_input_tensor,
        "cont_input": continous_input_tensor,
    }

    network_scores = model(**final_network_inputs)

    # store the network output
    events = set_ak_column_f32(events, f"tautauNN_regression_output", network_scores["regression_output_hep"].numpy())
    events = set_ak_column_f32(events, f"tautauNN:_classification_output", network_scores["classification_output"].numpy())

    return events


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
    tf = maybe_import("tensorflow")

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


def select_DAU(events):
    """
    Helper function to select DAUs.
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


def select_DAU_decay_mode(events):
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


def check_network_embedding(events):
    # check for specific values in embedding and throw error if they are not within events
    # TODO find a good way to find all UNIQUE values without taking too much CPU time
    embedding_expected_inputs = {
        "pairType": [0, 1, 2],  #
        "dau1_decayMode": [-1, 0, 1, 10, 11], # -1 for e/mu
        "dau2_decayMode": [0, 1, 10, 11],  #
        "dau1_charge": [-1, 1],  #
        "dau2_charge": [-1, 1],  #
    }
    for feature in embedding_expected_inputs.keys():
        expected_values = embedding_expected_inputs[feature]
        events_of_specific_feature = events[feature]

        value_not_in_mask = ~ak.any([events_of_specific_feature == value for value in expected_values], axis=0)

        if ak.sum(value_not_in_mask) > 0:
            indices_wrong_events = ak.local_index(events_of_specific_feature)[value_not_in_mask]
            unique_wrong_values = np.unique(events_of_specific_feature[indices_wrong_events])
            raise ValueError(
                f"Network expects {feature} to be have folllowing values: {expected_values}."
                f"But got {unique_wrong_values}")
