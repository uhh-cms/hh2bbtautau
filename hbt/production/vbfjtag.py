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
        "Jet.{pt,eta,phi,mass,jetId,btagDeepFlavB}",
        MET_COLUMN("{pt,phi}"),
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
)
def vbfjtag(
    self: Producer,
    events: ak.Array,
    vbfjet_mask: ak.Array,
    lepton_pair: ak.Array,
    is_bjet_mask: ak.Array,  # whether the jet was tagged by the hhbjet tagger
    selected_fatjets: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Returns the VBFjTag score per passed jet.
    """
    # get a mask of events where there are at least two tau candidates and at least two jets
    # and only get the scores for jets in these events

    event_mask = (
        (ak.num(lepton_pair, axis=1) >= 2) &
        (ak.sum(vbfjet_mask, axis=1) >= 2)
    )
    # & (ak.sum(is_bjet_mask, axis=1) < ak.sum(vbfjet_mask, axis=1) >= 2) to have at least one jet to give a score to?
    # additional criteria?

    # TODO: ordering by decreasing eta

    # prepare objects
    n_jets_max = 10
    jets = events.Jet[vbfjet_mask][event_mask][..., :n_jets_max]
    leps = lepton_pair[event_mask][..., [0, 1]]
    htt = leps[..., 0] + leps[..., 1]
    met = events[event_mask][self.config_inst.x.met_name]
    jet_shape = abs(jets.pt) >= 0
    n_jets_capped = ak.num(jets, axis=1)
    # is_bjet = ak.fill_none(
    #     ak.from_iter(
    #         [i in is_bjet_indices for i in ak.flatten(jets.jetId, axis
    #             =1)],
    #         highlevel=False,
    #     ),
    #     False,
    #     axis=-1,
    # )
    # TODO: check where the is_bjet_mask applies, all events or less?
    is_bjet = ak.values_astype(is_bjet_mask[event_mask][..., :n_jets_max], np.int32)



    # jet_pt: pT of each of the jets.
    # jet_eta: eta of each of the jets.
    # rel_jet_M_pt: Relative mass of the b-jet candidate: jet M / jet pT.
    # rel_jet_E_pt Relative energy of the b-jet candidate: jet E / jet pT.
    # jet_htt_deta: Eta between the b-jet and the visible 4-momentum of the HTT.
    # jet_btagScore: The score of the b-jet candidate given by the b-tagger : ParticleNet
    # jet_htt_dphi: Phi between the b-jet and the visible 4-momentum of the HTT.
    # jet_isbjet: 1 for jets tagged as b-jet; 0 for non tagged b-jets
    # era_id: 0 - 2022preEE, 1 - 2022postEE, 2 - 2023preBPix, 3 - 2023postBPix
    # channelId: 0 - MuTau, 1 - ETau, 2 - TauTau, 3 - MuMu, 4 - EE, 5 - EMu
    # htt_pt: pT of visible 4-momentum of the HTT candidate.
    # htt_eta: eta of HTT visible 4-momentum of the HTT candidate.
    # htt_met_dphi: Phi between the visible 4-momentum of the HTT candidate and the MET.
    # rel_met_pt_htt_pt: Relative MET: MET / pT of the visible 4-momentum of the HTT candidate.
    # htt_scalar_pt: Sum of the pT of the 2 selected taus.


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
        is_bjet,
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

    # TODO: remove scores for hhbjets and vbfjets matching the fatjet with highest particleNet_XbbVsQCD score
    from IPython import embed; embed(header="vbfjtag debug scores, cross_cleaning")
    cross_cleaning_fatjet_mask = ak.firsts(events.Jet[vbfjet_mask][event_mask].metric_table(selected_fatjets) > 0.8, axis=2)

    # prevent Memory Corruption Error
    vbfjet_mask = ak.fill_none(vbfjet_mask, False, axis=-1)

    # insert scores into an array with same shape as input jets (without vbfjet_mask and event_mask)
    all_scores = ak.fill_none(full_like(events.Jet.pt, EMPTY_FLOAT, dtype=np.float32), EMPTY_FLOAT, axis=-1)
    flat_np_view(all_scores, axis=1)[ak.flatten(vbfjet_mask & event_mask, axis=1)] = flat_np_view(scores)

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
            values = ak.concatenate([values, scores_ext], axis=1)
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

    # unpack the external files bundle and setup the evaluator
    bundle = reqs["external_files"]
    self.evaluator = TFEvaluator()
    self.evaluator.add_model("vbfjtag_even", bundle.files.hh_btag_repo.even.abspath)
    self.evaluator.add_model("vbfjtag_odd", bundle.files.hh_btag_repo.odd.abspath)

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

    # start the evaluator
    self.evaluator.start()


@vbfjtag.teardown
def vbfjtag_teardown(self: Producer, **kwargs) -> None:
    """
    Stops the TF evaluator.
    """
    if (evaluator := getattr(self, "evaluator", None)) is not None:
        evaluator.stop()
