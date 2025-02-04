# coding: utf-8

"""
Producers for the HHBtag score.
See https://github.com/hh-italian-group/HHbtag.
"""

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, dev_sandbox, InsertableDict
from columnflow.columnar_util import EMPTY_FLOAT, layout_ak_array

from hbt.util import IF_RUN_2

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@producer(
    uses={
        "event", "channel_id",
        "Jet.{pt,eta,phi,mass,jetId,btagDeepFlavB}", IF_RUN_2("Jet.puId"),
        # dynamic MET columns added in init
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
)
def hhbtag(
    self: Producer,
    events: ak.Array,
    jet_mask: ak.Array,
    lepton_pair: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Returns the HHBtag score per passed jet.
    """
    # get a mask of events where there are at least two tau candidates and at least two jets
    # and only get the scores for jets in these events
    event_mask = (
        (ak.num(lepton_pair, axis=1) >= 2) &
        (ak.sum(jet_mask, axis=1) >= 2)
    )

    # prepare objects
    n_jets_max = 10
    jets = events.Jet[jet_mask][event_mask][..., :n_jets_max]
    leps = lepton_pair[event_mask][..., [0, 1]]
    htt = leps[..., 0] + leps[..., 1]
    met = events[event_mask][self.config_inst.x.met_name]
    jet_shape = abs(jets.pt) >= 0
    n_jets_capped = ak.num(jets, axis=1)

    # get input features
    input_features = [
        jet_shape * 1,
        jets.pt,
        jets.eta,
        jets.mass / jets.pt,
        jets.energy / jets.pt,
        abs(jets.eta - htt.eta),
        jets.btagDeepFlavB if self.hhbtag_version == "v2" else jets.btagPNetB,
        jets.delta_phi(htt),
        jet_shape * (self.hhb_campaign_year),
        jet_shape * (ak.values_astype(
            self.hhb_channel_map[events[event_mask].channel_id], np.int32,
        )),
        jet_shape * htt.pt,
        jet_shape * htt.eta,
        jet_shape * htt.delta_phi(met),
        jet_shape * (met.pt / htt.pt),
        jet_shape * ak.sum(leps.pt, axis=1),
    ]

    # helper to split events, cast to float32, concatenate across new axis,
    # then pad with zeros for up to n_jets_max jets
    def split(where):
        features = ak.concatenate(
            [
                ak.values_astype(f[where][..., None], np.float32)
                for f in input_features
            ],
            axis=2,
        )
        # fill
        features = ak.fill_none(
            ak.pad_none(features, n_jets_max, axis=1),
            np.zeros(len(input_features), dtype=np.float32),
            axis=1,
        )
        # fix the dimension of the last axis to the known number of input features
        features = features[..., list(range(len(input_features)))]
        return ak.to_numpy(features)

    # reserve an output score array
    scores = np.ones((ak.sum(event_mask), n_jets_max), dtype=np.float32) * EMPTY_FLOAT

    # fill even and odd events if there are any
    even_mask = ak.to_numpy((events[event_mask].event % 2) == 0)
    if ak.sum(even_mask):
        input_features_even = split(even_mask)
        scores_even = self.hhbtag_model_even(input_features_even).numpy()
        scores[even_mask] = scores_even
    if ak.sum(~even_mask):
        input_features_odd = split(~even_mask)
        scores_odd = self.hhbtag_model_odd(input_features_odd).numpy()
        scores[~even_mask] = scores_odd

    # remove the scores of padded jets
    where = ak.from_regular(ak.local_index(scores) < n_jets_capped[..., None], axis=1)
    scores = ak.from_regular(scores, axis=1)[where]

    # add scores to events that had more than n_jets_max selected jets
    # (use zero here as this is also what the hhbtag model does for missing jets)
    layout_ext = events.Jet.pt[jet_mask][event_mask][..., n_jets_max:]
    # when there are no selected events, we can reuse layout_ext and consider it to be scores_ext
    if len(layout_ext) == 0:
        scores_ext = layout_ext
    else:
        scores_ext = layout_ak_array(np.zeros(len(ak.flatten(layout_ext)), dtype=np.int32), layout_ext)
    scores = ak.concatenate([scores, scores_ext], axis=1)

    # prevent Memory Corruption Error
    jet_mask = ak.fill_none(jet_mask, False, axis=-1)

    # insert scores into an array with same shape as input jets (without jet_mask and event_mask)
    all_scores = ak.fill_none(ak.full_like(events.Jet.pt, EMPTY_FLOAT, dtype=np.float32), EMPTY_FLOAT, axis=-1)
    np.asarray(ak.flatten(all_scores))[ak.flatten(jet_mask & event_mask, axis=1)] = np.asarray(ak.flatten(scores))

    return all_scores


@hhbtag.init
def hhbtag_init(self: Producer, **kwargs) -> None:
    self.uses.add(f"{self.config_inst.x.met_name}.{{pt,phi}}")


@hhbtag.requires
def hhbtag_requires(self: Producer, reqs: dict) -> None:
    """
    Add the external files bundle to requirements.
    """
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@hhbtag.setup
def hhbtag_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    """
    Sets up the two HHBtag TF models.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    # unpack the external files bundle, create a subdiretory and unpack the hhbtag repo in it
    bundle = reqs["external_files"]

    arc = bundle.files.hh_btag_repo
    # unpack repo
    repo_dir = bundle.files_dir.child("hh-btag-repo", type="d")
    arc.load(repo_dir, formatter="tar")

    # get the version of the external file
    self.hhbtag_version = self.config_inst.x.external_files["hh_btag_repo"][1]

    # define the model path
    repo_dir = repo_dir.child("hh-btag-master/models/")
    model_path = f"HHbtag_{self.hhbtag_version}_par"
    # save both models (even and odd event numbers)
    with self.task.publish_step("loading hhbtag models ..."):
        self.hhbtag_model_even = tf.saved_model.load(repo_dir.child(f"{model_path}_0").path)
        self.hhbtag_model_odd = tf.saved_model.load(repo_dir.child(f"{model_path}_1").path)

    # prepare mappings for the HHBtag model
    channel_mapping = {
        "mutau": 0 if self.hhbtag_version == "v3" else 1,
        "etau": 1 if self.hhbtag_version == "v3" else 0,
        "tautau": 2 if self.hhbtag_version == "v3" else 2,
        "mumu": 3,
        "ee": 4,
        "emu": 5,
    }

    self.hhb_channel_map = (
        np.array([
            np.nan,
            channel_mapping["etau"],
            channel_mapping["mutau"],
            channel_mapping["tautau"],
            channel_mapping["ee"],
            channel_mapping["mumu"],
            channel_mapping["emu"],
        ])
        if self.hhbtag_version == "v3" else
        np.array([
            np.nan,
            channel_mapping["mutau"],
            channel_mapping["etau"],
            channel_mapping["tautau"],
            np.nan,
            np.nan,
            np.nan,
        ])
    )

    # campaign year mapping
    year_map = {
        "22pre_v14": 0,
        "22post_v14": 1,
        "23pre_v14": 2,
        "23post_v14": 3,
    }

    # we use suffixes for configs, like "_sync"
    year = self.config_inst.campaign.x.year
    config_name = "_".join(self.config_inst.name.split("_")[:2])
    self.hhb_campaign_year = (
        year
        if year < 2021
        else year_map[config_name]
    )
