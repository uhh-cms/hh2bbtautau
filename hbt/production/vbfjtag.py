# coding: utf-8

"""
Producers for the VBFjTag score.
See https://github.com/elviramartinv/VBFjtag/
"""

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.columnar_util import EMPTY_FLOAT, layout_ak_array, set_ak_column, full_like, flat_np_view
from columnflow.types import Any

from hbt.util import MET_COLUMN

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@producer(
    uses={
        "event", "channel_id",
        "Jet.{pt,eta,phi,mass,btagPNetB}",
        MET_COLUMN("{pt,phi}"),
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
)
def vbfjtag(
    self: Producer,
    events: ak.Array,
    vbfjet_mask: ak.Array,
    lepton_pair: ak.Array,
    is_hhbjet_mask: ak.Array,  # whether the jet was tagged by the hhbjet tagger
    selected_fatjets: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Returns the VBFjTag score per passed jet.
    Clean the scores from the selected fatjets and bjets before returning them.
    """
    # start the evaluator
    if not self.evaluator.running:
        self.evaluator.start()

    # get a mask of events where there are at least two tau candidates and at least two jets
    # and only get the scores for jets in these events
    event_mask = (
        (ak.num(lepton_pair, axis=1) >= 2) &
        (ak.sum(vbfjet_mask, axis=1) >= 2)
        # maybe just 3 since two of them would be the hhbjets if they are there?
        # maybe add hhbjet_mask >=2, such that only these events are considered?
    )

    # ordering by decreasing eta then pt
    f = 10**(np.ceil(np.log10(ak.max(events.Jet.pt) or 0.0)) + 2)
    jet_sorting_key = abs(events.Jet.eta) * f + events.Jet.pt
    jet_sorting_indices = ak.argsort(jet_sorting_key, axis=-1, ascending=False)

    # back transformations for the saving of the scores
    jet_unsorting_indices = ak.argsort(jet_sorting_indices)

    # sorted vbfmask to avoid repetition
    vbfjet_mask_sorted = vbfjet_mask[jet_sorting_indices]

    # prepare objects
    n_jets_max = 10
    jets = events.Jet[jet_sorting_indices][vbfjet_mask_sorted][event_mask][..., :n_jets_max]
    leps = lepton_pair[event_mask][..., [0, 1]]
    htt = leps[..., 0] + leps[..., 1]
    met = events[event_mask][self.config_inst.x.met_name]
    jet_shape = abs(jets.pt) >= 0
    n_jets_capped = ak.num(jets, axis=1)
    is_hhbjet = ak.values_astype(is_hhbjet_mask[jet_sorting_indices][vbfjet_mask_sorted][event_mask][..., :n_jets_max], np.float32)  # noqa: E501

    # get input features
    input_features = [
        jet_shape * 1,
        jets.pt,
        jets.eta,
        jets.mass / jets.pt,
        jets.energy / jets.pt,
        abs(jets.eta - htt.eta),
        jets.btagPNetB,
        jets.delta_phi(htt),
        is_hhbjet,
        jet_shape * (self.vbfjtag_campaign),
        jet_shape * self.vbfjtag_channel_map[events[event_mask].channel_id],
        jet_shape * htt.pt,
        jet_shape * htt.eta,
        jet_shape * htt.delta_phi(met),
        jet_shape * (met.pt / htt.pt),
        jet_shape * ak.sum(leps.pt, axis=1),
    ]

    # helper to split events, cast to float32, concatenate across new axis,
    # then pad with zeros for up to n_jets_max jets
    def split(where, input_features=input_features):
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
        scores_even = self.evaluator("vbfjtag_even", input_features_even)
        scores[even_mask] = scores_even
    if ak.sum(~even_mask):
        input_features_odd = split(~even_mask)
        scores_odd = self.evaluator("vbfjtag_odd", input_features_odd)
        scores[~even_mask] = scores_odd

    # remove the scores of padded jets
    where = ak.from_regular(ak.local_index(scores) < n_jets_capped[..., None], axis=1)
    scores = ak.from_regular(scores, axis=1)[where]

    # add scores to events that had more than n_jets_max selected jets
    # (use zero here as this is also what the vbfjtag model does for missing jets)
    layout_ext = events.Jet.pt[vbfjet_mask][event_mask][..., n_jets_max:]
    # when there are no selected events, we can reuse layout_ext and consider it to be scores_ext
    if len(layout_ext) == 0:
        scores_ext = layout_ext
    else:
        scores_ext = layout_ak_array(np.zeros(len(ak.flatten(layout_ext)), dtype=np.int32), layout_ext)
    scores = ak.concatenate([scores, scores_ext], axis=1)

    # remove scores for vbfjets matching the fatjet with highest particleNet_XbbVsQCD score
    metric_table_fatjet = (events.Jet[jet_sorting_indices][vbfjet_mask_sorted][event_mask].metric_table(selected_fatjets[event_mask]) > 0.8)  # noqa: E501

    # since only one selected fatjet, take the first element of the metric table on the last axis
    # this works even if no selected fatjets are in the event mask
    cross_cleaned_fatjet_mask = ak.fill_none(ak.firsts(metric_table_fatjet, axis=-1), True, axis=1)

    # bring mask back to full array to merge it with the vbfjet_mask once reordered
    full_cross_cleaned_fatjet_mask = full_like(events.Jet.pt, True, dtype=bool)
    flat_np_view(full_cross_cleaned_fatjet_mask)[ak.flatten(vbfjet_mask_sorted & event_mask)] = flat_np_view(cross_cleaned_fatjet_mask)  # noqa: E501

    # prevent Memory Corruption Error
    vbfjet_mask = ak.fill_none(vbfjet_mask, False, axis=-1)
    vbfjet_mask_sorted = ak.fill_none(vbfjet_mask_sorted, False, axis=-1)

    # insert scores into an array with same shape as input jets (without vbfjet_mask and event_mask)
    all_scores = ak.fill_none(full_like(events.Jet.pt, EMPTY_FLOAT, dtype=np.float32), EMPTY_FLOAT, axis=-1)
    flat_np_view(all_scores, axis=1)[ak.flatten(vbfjet_mask_sorted & event_mask, axis=1)] = flat_np_view(scores)

    # bring the scores and the fatjet cleaned mask back to the original ordering
    all_scores = all_scores[jet_unsorting_indices]
    full_cross_cleaned_fatjet_mask = full_cross_cleaned_fatjet_mask[jet_unsorting_indices]

    # remove scores where the cross cleaning reveals "wrong" vbf jet (either a fatjet or a hhbjet)
    cross_cleaned_fatjet_hhbjet_mask = vbfjet_mask & full_cross_cleaned_fatjet_mask & ~is_hhbjet_mask
    empty_mask = full_like(events.Jet.pt[~cross_cleaned_fatjet_hhbjet_mask], EMPTY_FLOAT, dtype=np.float32)
    flat_np_view(all_scores)[ak.flatten(~cross_cleaned_fatjet_hhbjet_mask)] = flat_np_view(empty_mask)

    events = set_ak_column(events, "vbfjtag_score", all_scores)

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
            values = ak.concatenate([values, scores_ext], axis=1)[jet_unsorting_indices]

            # fill placeholder
            np.asarray(ak.flatten(value_placeholder))[ak.flatten(vbfjet_mask & event_mask, axis=1)] = (
                np.asarray(ak.flatten(values))
            )
            events = set_ak_column(events, f"sync_vbfjtag_{column}", value_placeholder)

    return events


@vbfjtag.init
def vbfjtag_init(self: Producer, **kwargs) -> None:
    # produce input columns
    if self.config_inst.x.sync:
        self.produces.add("sync_*")


@vbfjtag.requires
def vbfjtag_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    """
    Add the external files bundle to requirements.
    """
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@vbfjtag.setup
def vbfjtag_setup(
    self: Producer,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    **kwargs,
) -> None:
    """
    Sets up the two VBFjTag TF models.
    """
    from hbt.ml.tf_evaluator import TFEvaluator

    if not getattr(task, "taf_tf_evaluator", None):
        task.taf_tf_evaluator = TFEvaluator()
    self.evaluator = task.taf_tf_evaluator

    # unpack the external files bundle and setup the evaluator
    bundle = reqs["external_files"]
    self.evaluator.add_model("vbfjtag_even", bundle.files.vbf_jtag_repo.even.abspath)
    self.evaluator.add_model("vbfjtag_odd", bundle.files.vbf_jtag_repo.odd.abspath)

    # get the model version (coincides with the external file version)
    self.vbfjtag_version = self.config_inst.x.external_files.vbf_jtag_repo.version

    # prepare mappings for the VBFjTag model (see links above for mapping information)
    channel_map = {
        self.config_inst.channels.n.etau.id: 1,
        self.config_inst.channels.n.mutau.id: 0,
        self.config_inst.channels.n.tautau.id: 2,
        # for versions before v3, control channels were not used in the training, so we map them to
        # the most similar analysis channel
        self.config_inst.channels.n.ee.id: 4,
        self.config_inst.channels.n.mumu.id: 3,
        self.config_inst.channels.n.emu.id: 5,
    }
    # convert
    self.vbfjtag_channel_map = np.array([
        channel_map.get(cid, np.nan)
        for cid in range(max(channel_map.keys()) + 1)
    ])

    # campaign year mapping
    campaign_key = (self.config_inst.campaign.x.year, self.config_inst.campaign.x.postfix)
    # map campaign year and postfix to a number
    self.vbfjtag_campaign = {
        (2022, ""): 0,
        (2022, "EE"): 1,
        (2023, ""): 2,
        (2023, "BPix"): 3,
    }[campaign_key]

    # validate the met name
    vbfjtag_met_name = "PuppiMET"
    if self.config_inst.x.met_name != vbfjtag_met_name:
        raise ValueError(
            f"vbfjtag model {self.vbfjtag_version} uses {vbfjtag_met_name}, but config requests "
            f"{self.config_inst.x.met_name}",
        )


@vbfjtag.teardown
def vbfjtag_teardown(self: Producer, task: law.Task, **kwargs) -> None:
    """
    Stops the TF evaluator.
    """
    if (evaluator := getattr(task, "taf_tf_evaluator", None)):
        evaluator.stop()
    task.taf_tf_evaluator = None
    self.evaluator = None
