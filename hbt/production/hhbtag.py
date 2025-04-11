# coding: utf-8

"""
Producers for the HHBtag score.
See https://github.com/hh-italian-group/HHbtag for v1 and v2,
and https://gitlab.cern.ch/hh/bbtautau/hh-btag for v3.
"""

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.columnar_util import EMPTY_FLOAT, layout_ak_array, set_ak_column, full_like, flat_np_view
from columnflow.types import Any

from hbt.util import IF_RUN_2, MET_COLUMN

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@producer(
    uses={
        "event", "channel_id",
        "Jet.{pt,eta,phi,mass,jetId,btagDeepFlavB}", IF_RUN_2("Jet.puId"),
        MET_COLUMN("{pt,phi}"),
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
        (jets.btagDeepFlavB if self.hhbtag_version == "v2" else jets.btagPNetB),
        jets.delta_phi(htt),
        jet_shape * (self.hhbtag_campaign),
        jet_shape * self.hhbtag_channel_map[events[event_mask].channel_id],
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
        scores_even = self.evaluator("hhbtag_even", input_features_even)
        scores[even_mask] = scores_even
    if ak.sum(~even_mask):
        input_features_odd = split(~even_mask)
        scores_odd = self.evaluator("hhbtag_odd", input_features_odd)
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
    all_scores = ak.fill_none(full_like(events.Jet.pt, EMPTY_FLOAT, dtype=np.float32), EMPTY_FLOAT, axis=-1)
    flat_np_view(all_scores, axis=1)[ak.flatten(jet_mask & event_mask, axis=1)] = flat_np_view(scores)

    events = set_ak_column(events, "hhbtag_score", all_scores)

    if self.config_inst.x.sync:
        # for sync save input variables as additional columns in the sync collection
        input_feature_names = [
            "jet_shape", "jets_pt", "jets_eta",
            "jets_ratio_mass_to_pt", "jets_ratio_energy_to_pt",
            "delta_eta_jets_to_htt", "pnet_btag_score",
            "delta_phi_jets_to_htt", "campaign",
            "channel_id", "htt_pt",
            "htt_eta", "delta_phi_htt_to_met",
            "ratio_pt_met_to_htt", "all_lepton_pt",
        ]
        store_sync_columns = dict(zip(input_feature_names, input_features))

        # store inputs
        for column, values in store_sync_columns.items():
            # create empty multi dim placeholder
            value_placeholder = ak.fill_none(
                ak.full_like(events.Jet.pt, EMPTY_FLOAT, dtype=np.float32), EMPTY_FLOAT, axis=-1,
            )
            values = ak.concatenate([values, scores_ext], axis=1)
            # fill placeholder
            np.asarray(ak.flatten(value_placeholder))[ak.flatten(jet_mask & event_mask, axis=1)] = (
                np.asarray(ak.flatten(values))
            )
            events = set_ak_column(events, f"sync_hhbtag_{column}", value_placeholder)

    return events


@hhbtag.init
def hhbtag_init(self: Producer, **kwargs) -> None:
    # produce input columns
    if self.config_inst.x.sync:
        self.produces.add("sync_*")


@hhbtag.requires
def hhbtag_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    """
    Add the external files bundle to requirements.
    """
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@hhbtag.setup
def hhbtag_setup(
    self: Producer,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    **kwargs,
) -> None:
    """
    Sets up the two HHBtag TF models.
    """
    from hbt.ml.tf_evaluator import TFEvaluator

    # unpack the external files bundle and setup the evaluator
    bundle = reqs["external_files"]
    self.evaluator = TFEvaluator()
    self.evaluator.add_model("hhbtag_even", bundle.files.hh_btag_repo.even.abspath)
    self.evaluator.add_model("hhbtag_odd", bundle.files.hh_btag_repo.odd.abspath)

    # get the model version (coincides with the external file version)
    self.hhbtag_version = self.config_inst.x.external_files.hh_btag_repo.version

    # prepare mappings for the HHBtag model (see links above for mapping information)
    channel_map = {
        self.config_inst.channels.n.etau.id: 1 if self.hhbtag_version == "v3" else 0,
        self.config_inst.channels.n.mutau.id: 0 if self.hhbtag_version == "v3" else 1,
        self.config_inst.channels.n.tautau.id: 2,
        # for versions before v3, control channels were not used in the training, so we map them to
        # the most similar analysis channel
        self.config_inst.channels.n.ee.id: 4 if self.hhbtag_version == "v3" else 0,
        self.config_inst.channels.n.mumu.id: 3 if self.hhbtag_version == "v3" else 1,
        self.config_inst.channels.n.emu.id: 5 if self.hhbtag_version == "v3" else 0,
    }
    # convert
    self.hhbtag_channel_map = np.array([
        channel_map.get(cid, np.nan)
        for cid in range(max(channel_map.keys()) + 1)
    ])

    # campaign year mapping
    campaign_key = (self.config_inst.campaign.x.year, self.config_inst.campaign.x.postfix)
    if self.hhbtag_version in ("v1", "v2") and self.config_inst.campaign.x.run == 3:
        # note: we might want to drop this fallback once the new v3 model is validated
        logger.debug("using '2018' as evaluation year for hhbtag model in run 3")
        campaign_key = (2018, "")
    self.hhbtag_campaign = {
        (2016, "APV"): 2016,
        (2016, ""): 2016,
        (2017, ""): 2017,
        (2018, ""): 2018,
        (2022, ""): 0,
        (2022, "EE"): 1,
        (2023, ""): 2,
        (2023, "BPix"): 3,
    }[campaign_key]

    # validate the met name
    hhbtag_met_name = "PuppiMET" if self.hhbtag_version == "v3" else "MET"
    if self.config_inst.x.met_name != hhbtag_met_name:
        raise ValueError(
            f"hhbtag model {self.hhbtag_version} uses {hhbtag_met_name}, but config requests "
            f"{self.config_inst.x.met_name}",
        )

    # start the evaluator
    self.evaluator.start()


@hhbtag.teardown
def hhbtag_teardown(self: Producer, **kwargs) -> None:
    """
    Stops the TF evaluator.
    """
    if (evaluator := getattr(self, "evaluator", None)) is not None:
        evaluator.stop()
