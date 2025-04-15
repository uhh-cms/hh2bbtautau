# coding: utf-8

"""
Producer for evaluating the pDNN developed for the resonant run 2 analysis.
See https://github.com/uhh-cms/tautauNN
"""

from __future__ import annotations

import functools

import law

from columnflow.production import Producer, producer
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import (
    set_ak_column, attach_behavior, flat_np_view, EMPTY_FLOAT, default_coffea_collections,
)
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.types import Any

np = maybe_import("numpy")
ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)

# helper function
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


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
        "HHBJet.{pt,eta,phi,mass,hhbtag,btagDeepFlav*,btagPNet*}",
        "FatJet.{eta,phi,pt,mass}",
        # MET variables added in dynamic init,
    },
    # whether the model is parameterized in mass, spin and year
    # (this is a slight forward declaration but simplifies the code reasonably well in our use case)
    parametrized=None,
    # limited chunk size to avoid memory issues
    max_chunk_size=5_000,
    # produced columns are added in the deferred init below
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
    # not exposed to be called from the command line
    exposed=False,
)
def _res_dnn_evaluation(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Base producer for dnn evaluations of the resonant run 2 analyses, whose models are considered
    external and thus part of producers rather than standalone ml model objects.
    The output scores are classifying if incoming events are HH, Drell-Yan or ttbar.
    The network uses continous, categorical and parametrized inputs. A list of all inputs in the
    correct order can be found in the tautauNN repo:
    https://github.com/uhh-cms/tautauNN/blob/f1ca194/evaluation/interface.py#L67
    """
    # ensure coffea behavior
    events = self[attach_coffea_behavior](
        events,
        collections={"HHBJet": default_coffea_collections["Jet"]},
        **kwargs,
    )

    # define the pair type (KLUBs channel id)
    pair_type = np.zeros(len(events), dtype=np.int32)
    for channel_id, pair_type_id in self.channel_id_to_pair_type.items():
        pair_type[events.channel_id == channel_id] = pair_type_id

    # get visible tau decay products, consider them all as tau types
    vis_taus = attach_behavior(
        ak.concatenate((events.Electron, events.Muon, events.Tau), axis=1),
        type_name="Tau",
    )
    vis_tau1, vis_tau2 = vis_taus[:, 0], vis_taus[:, 1]

    # get decay mode of first lepton (e, mu or tau)
    tautau_mask = events.channel_id == self.config_inst.channels.n.tautau.id
    dm1 = -1 * np.ones(len(events), dtype=np.int32)
    if ak.any(tautau_mask):
        dm1[tautau_mask] = events.Tau.decayMode[tautau_mask][:, 0]

    # get decay mode of second lepton (also a tau, but position depends on channel)
    leptau_mask = (
        (events.channel_id == self.config_inst.channels.n.etau.id) |
        (events.channel_id == self.config_inst.channels.n.mutau.id)
    )
    dm2 = -1 * np.ones(len(events), dtype=np.int32)
    if ak.any(leptau_mask):
        dm2[leptau_mask] = events.Tau.decayMode[leptau_mask][:, 0]
    if ak.any(tautau_mask):
        dm2[tautau_mask] = events.Tau.decayMode[tautau_mask][:, 1]

    # the dnn treats dm 2 as 1, so we need to map it
    dm1 = np.where(dm1 == 2, 1, dm1)
    dm2 = np.where(dm2 == 2, 1, dm2)

    # whether the events is resolvede, boosted or neither
    has_jet_pair = ak.num(events.HHBJet) >= 2
    has_fatjet = ak.num(events.FatJet) >= 1

    # before preparing the network inputs, define a mask of events which have caregorical features
    # that are actually covered by the networks embedding layers; other events cannot be evaluated!
    event_mask = (
        np.isin(pair_type, self.embedding_expected_inputs["pair_type"]) &
        np.isin(dm1, self.embedding_expected_inputs["decay_mode1"]) &
        np.isin(dm2, self.embedding_expected_inputs["decay_mode2"]) &
        np.isin(vis_tau1.charge, self.embedding_expected_inputs["charge1"]) &
        np.isin(vis_tau2.charge, self.embedding_expected_inputs["charge2"]) &
        (has_jet_pair | has_fatjet) &
        (self.year_flag in self.embedding_expected_inputs["year"])
    )

    # apply to all arrays needed until now
    _events = events[event_mask]
    pair_type = pair_type[event_mask]
    vis_tau1, vis_tau2 = vis_tau1[event_mask], vis_tau2[event_mask]
    tautau_mask = tautau_mask[event_mask]
    dm1, dm2 = dm1[event_mask], dm2[event_mask]
    has_jet_pair, has_fatjet = has_jet_pair[event_mask], has_fatjet[event_mask]

    # prepare network inputs
    f = DotDict()

    # compute angle from visible mother particle of vis_tau1 and vis_tau2
    # used to rotate the kinematics of dau{1,2}, met, bjet{1,2} and fatjets relative to it
    phi_lep = np.arctan2(vis_tau1.py + vis_tau2.py, vis_tau1.px + vis_tau2.px)

    # lepton 1
    f.vis_tau1_px, f.vis_tau1_py = rotate_to_phi(phi_lep, vis_tau1.px, vis_tau1.py)
    f.vis_tau1_pz, f.vis_tau1_e = vis_tau1.pz, vis_tau1.energy

    # lepton 2
    f.vis_tau2_px, f.vis_tau2_py = rotate_to_phi(phi_lep, vis_tau2.px, vis_tau2.py)
    f.vis_tau2_pz, f.vis_tau2_e = vis_tau2.pz, vis_tau2.energy

    # there might be less than two jets or no fatjet, so pad them
    bjets = ak.pad_none(_events.HHBJet, 2, axis=1)
    fatjet = ak.pad_none(_events.FatJet, 1, axis=1)[:, 0]

    # bjet 1
    f.bjet1_px, f.bjet1_py = rotate_to_phi(phi_lep, bjets[:, 0].px, bjets[:, 0].py)
    f.bjet1_pz, f.bjet1_e = bjets[:, 0].pz, bjets[:, 0].energy
    f.bjet1_btag_df = bjets[:, 0].btagDeepFlavB
    f.bjet1_cvsb = bjets[:, 0].btagDeepFlavCvB
    f.bjet1_cvsl = bjets[:, 0].btagDeepFlavCvL
    f.bjet1_hhbtag = bjets[:, 0].hhbtag

    # bjet 2
    f.bjet2_px, f.bjet2_py = rotate_to_phi(phi_lep, bjets[:, 1].px, bjets[:, 1].py)
    f.bjet2_pz, f.bjet2_e = bjets[:, 1].pz, bjets[:, 1].energy
    f.bjet2_btag_df = bjets[:, 1].btagDeepFlavB
    f.bjet2_cvsb = bjets[:, 1].btagDeepFlavCvB
    f.bjet2_cvsl = bjets[:, 1].btagDeepFlavCvL
    f.bjet2_hhbtag = bjets[:, 1].hhbtag

    # fatjet variables
    f.fatjet_px, f.fatjet_py = rotate_to_phi(phi_lep, fatjet.px, fatjet.py)
    f.fatjet_pz, f.fatjet_e = fatjet.pz, fatjet.energy

    # mask values as done during training of the network
    def mask_values(mask, value, *fields):
        for field in fields:
            arr = ak.fill_none(f[field], value, axis=0)
            flat_np_view(arr)[mask] = value
            f[field] = arr

    mask_values(~has_jet_pair, 0.0, "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e")
    mask_values(~has_jet_pair, 0.0, "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e")
    mask_values(~has_jet_pair, -1.0, "bjet1_btag_df", "bjet1_cvsb", "bjet1_cvsl", "bjet1_hhbtag")
    mask_values(~has_jet_pair, -1.0, "bjet2_btag_df", "bjet2_cvsb", "bjet2_cvsl", "bjet2_hhbtag")
    mask_values(~has_fatjet, 0.0, "fatjet_px", "fatjet_py", "fatjet_pz", "fatjet_e")

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
    mask_values(~has_jet_pair, 0.0, "hbb_e", "hbb_px", "hbb_py", "hbb_pz")

    # htt + hbb
    f.htthbb_e = f.htt_e + f.hbb_e
    f.htthbb_px = f.htt_px + f.hbb_px
    f.htthbb_py = f.htt_py + f.hbb_py
    f.htthbb_pz = f.htt_pz + f.hbb_pz
    mask_values(~has_jet_pair, 0.0, "htthbb_e", "htthbb_px", "htthbb_py", "htthbb_pz")

    # htt + fatjet
    f.httfatjet_e = f.htt_e + f.fatjet_e
    f.httfatjet_px = f.htt_px + f.fatjet_px
    f.httfatjet_py = f.htt_py + f.fatjet_py
    f.httfatjet_pz = f.htt_pz + f.fatjet_pz
    mask_values(~has_fatjet, 0.0, "httfatjet_e", "httfatjet_px", "httfatjet_py", "httfatjet_pz")

    # MET variables
    _met = _events[self.config_inst.x.met_name]
    f.met_px, f.met_py = rotate_to_phi(
        phi_lep,
        _met.pt * np.cos(_met.phi),
        _met.pt * np.sin(_met.phi),
    )
    f.met_cov00, f.met_cov01, f.met_cov11 = _met.covXX, _met.covXY, _met.covYY

    # build continous inputs
    # (order exactly as documented in link above)
    continous_inputs = [
        np.asarray(t[..., None], dtype=np.float32) for t in [
            f.met_px, f.met_py, f.met_cov00, f.met_cov01, f.met_cov11,
            f.vis_tau1_px, f.vis_tau1_py, f.vis_tau1_pz, f.vis_tau1_e,
            f.vis_tau2_px, f.vis_tau2_py, f.vis_tau2_pz, f.vis_tau2_e,
            f.bjet1_px, f.bjet1_py, f.bjet1_pz, f.bjet1_e, f.bjet1_btag_df, f.bjet1_cvsb, f.bjet1_cvsl, f.bjet1_hhbtag,
            f.bjet2_px, f.bjet2_py, f.bjet2_pz, f.bjet2_e, f.bjet2_btag_df, f.bjet2_cvsb, f.bjet2_cvsl, f.bjet2_hhbtag,
            f.fatjet_px, f.fatjet_py, f.fatjet_pz, f.fatjet_e,
            f.htt_e, f.htt_px, f.htt_py, f.htt_pz,
            f.hbb_e, f.hbb_px, f.hbb_py, f.hbb_pz,
            f.htthbb_e, f.htthbb_px, f.htthbb_py, f.htthbb_pz,
            f.httfatjet_e, f.httfatjet_px, f.httfatjet_py, f.httfatjet_pz,
            (self.mass * np.ones(len(_events), dtype=np.float32)) if self.parametrized else None,
        ]
        if t is not None
    ]

    # build categorical inputs
    # (order exactly as documented in link above)
    categorical_inputs = [
        np.asarray(t[..., None], dtype=np.int32) for t in [
            pair_type,
            dm1, dm2,
            vis_tau1.charge, vis_tau2.charge,
            has_jet_pair, has_fatjet,
            (self.year_flag * np.ones(len(_events), dtype=np.int32)) if self.parametrized else None,
            (self.spin * np.ones(len(_events), dtype=np.int32)) if self.parametrized else None,
        ] if t is not None
    ]

    # evaluate the model
    scores = self.evaluator(
        "res",
        inputs=[
            np.concatenate(continous_inputs, axis=1),
            np.concatenate(categorical_inputs, axis=1),
        ],
    )

    # in very rare cases (1 in 25k), the network output can be none, likely for numerical reasons,
    # so issue a warning and set them to a default value
    nan_mask = ~np.isfinite(scores)
    if np.any(nan_mask):
        logger.warning(
            f"{nan_mask.sum() // scores.shape[1]} out of {scores.shape[0]} events have NaN scores; "
            "setting them to EMPTY_FLOAT",
        )
        scores[nan_mask] = EMPTY_FLOAT

    # prepare output columns with the shape of the original events and assign values into them
    for i, column in enumerate(self.output_columns):
        values = EMPTY_FLOAT * np.ones(len(events), dtype=np.float32)
        values[event_mask] = scores[:, i]
        events = set_ak_column_f32(events, column, values)

    if self.config_inst.x.sync:
        # store input columns for sync
        cont_inputs_names = [
            "met_px", "met_py", "met_cov00", "met_cov01", "met_cov11",
            "vis_tau1_px", "vis_tau1_py", "vis_tau1_pz", "vis_tau1_e",
            "vis_tau2_px", "vis_tau2_py", "vis_tau2_pz", "vis_tau2_e",
            "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e", "bjet1_btag_df", "bjet1_cvsb", "bjet1_cvsl", "bjet1_hhbtag",
            "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e", "bjet2_btag_df", "bjet2_cvsb", "bjet2_cvsl", "bjet2_hhbtag",
            "fatjet_px", "fatjet_py", "fatjet_pz", "fatjet_e",
            "htt_e", "htt_px", "htt_py", "htt_pz",
            "hbb_e", "hbb_px", "hbb_py", "hbb_pz",
            "htthbb_e", "htthbb_px", "htthbb_py", "htthbb_pz",
            "httfatjet_e", "httfatjet_px", "httfatjet_py", "httfatjet_pz",
        ]

        cat_inputs_names = [
            "pair_type", "dm1", "dm2", "vis_tau1_charge", "vis_tau2_charge", "has_jet_pair", "has_fatjet",
        ]
        for column, values in zip(
            cont_inputs_names + cat_inputs_names,
            continous_inputs + categorical_inputs,
        ):
            values_placeholder = EMPTY_FLOAT * np.ones(len(events), dtype=np.float32)
            values_placeholder[event_mask] = ak.flatten(values)
            events = set_ak_column_f32(events, "sync_res_dnn_" + column, values_placeholder)
    return events


@_res_dnn_evaluation.init
def _res_dnn_evaluation_init(self: Producer, **kwargs) -> None:
    self.uses.add(f"{self.config_inst.x.met_name}.{{pt,phi,covXX,covXY,covYY}}")


@_res_dnn_evaluation.requires
def _res_dnn_evaluation_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@_res_dnn_evaluation.setup
def _res_dnn_evaluation_setup(
    self: Producer,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    **kwargs,
) -> None:
    from hbt.ml.tf_evaluator import TFEvaluator

    # some checks
    if not isinstance(self.parametrized, bool):
        raise AttributeError("'parametrized' must be set in the producer configuration")

    # unpack the model archive
    bundle = reqs["external_files"]
    bundle.files
    model_dir = bundle.files_dir.child(self.cls_name, type="d")
    getattr(bundle.files, self.cls_name).load(model_dir, formatter="tar")

    # setup the evaluator
    self.evaluator = TFEvaluator()
    self.evaluator.add_model("res", model_dir.child("model_fold0").abspath, signature_key="serving_default")

    # categorical values handled by the network
    # (names and values from training code that was aligned to KLUB notation)
    self.embedding_expected_inputs = {
        "pair_type": [0, 1, 2],  # see mapping below
        "decay_mode1": [-1, 0, 1, 10, 11],  # -1 for e/mu
        "decay_mode2": [0, 1, 10, 11],
        "charge1": [-1, 1],
        "charge2": [-1, 1],
        "is_boosted": [0, 1],  # whether a selected fatjet is present
        "has_jet_pair": [0, 1],  # whether two or more jets are present
        "spin": [0, 2],
        "year": [0, 1, 2, 3],  # 0: 2016APV, 1: 2016, 2: 2017, 3: 2018
    }

    # our channel ids mapped to KLUB "pair_type"
    self.channel_id_to_pair_type = {
        # known during training
        self.config_inst.channels.n.mutau.id: 0,
        self.config_inst.channels.n.etau.id: 1,
        self.config_inst.channels.n.tautau.id: 2,
        # unknown during training
        self.config_inst.channels.n.ee.id: 1,
        self.config_inst.channels.n.mumu.id: 0,
        self.config_inst.channels.n.emu.id: 1,
    }

    # define the year based on the incoming campaign
    # (the training was done only for run 2, so map run 3 campaigns to 2018)
    self.year_flag = {
        (2016, "APV"): 0,
        (2016, ""): 1,
        (2017, ""): 2,
        (2018, ""): 3,
        (2022, ""): 3,
        (2022, "EE"): 3,
        (2023, ""): 3,
        (2023, "BPix"): 3,
    }[(self.config_inst.campaign.x.year, self.config_inst.campaign.x.postfix)]

    # start the evaluator
    self.evaluator.start()


@_res_dnn_evaluation.teardown
def _res_dnn_evaluation_teardown(self: Producer, **kwargs) -> None:
    """
    Stops the TF evaluator.
    """
    if (evaluator := getattr(self, "evaluator", None)) is not None:
        evaluator.stop()


#
# parameterized network
# trained with Radion (spin 0) and Graviton (spin 2) samples up to mX = 3000 GeV in all run 2 eras
#

res_pdnn = _res_dnn_evaluation.derive("res_pdnn", cls_dict={
    "parametrized": True,
    "exposed": True,
    "mass": 500,
    "spin": 0,
})


@res_pdnn.init
def res_pdnn_init(self: Producer, **kwargs) -> None:
    super(res_pdnn, self).init_func(**kwargs)

    # check spin value and mass values
    if self.spin not in {0, 2}:
        raise ValueError(f"invalid spin value: {self.spin}")
    if self.mass < 250:
        raise ValueError(f"invalid mass value: {self.mass}")

    # output column names (in this order)
    self.output_columns = [
        f"res_pdnn_s{self.spin}_m{self.mass}_{name}"
        for name in ["hh", "tt", "dy"]
    ]

    # update produced columns
    self.produces |= set(self.output_columns)


#
# non-parameterized network
# trained only with Radion (spin 0) samples up to mX = 800 GeV across all run 2 eras
#

res_dnn = _res_dnn_evaluation.derive("res_dnn", cls_dict={
    "parametrized": False,
    "exposed": True,
})


@res_dnn.init
def res_dnn_init(self: Producer, **kwargs) -> None:
    super(res_dnn, self).init_func(**kwargs)

    # output column names (in this order)
    self.output_columns = [
        f"res_dnn_{name}"
        for name in ["hh", "tt", "dy"]
    ]

    # update produced columns
    self.produces |= set(self.output_columns)
    if self.config_inst.x.sync:
        self.produces.add("sync_*")


#
# preprocessing helpers
#

def rotate_to_phi(ref_phi: ak.Array, px: ak.Array, py: ak.Array) -> tuple[ak.Array, ak.Array]:
    """
    Rotates a momentum vector extracted from *events* in the transverse plane to a reference phi
    angle *ref_phi*. Returns the rotated px and py components in a 2-tuple.
    """
    new_phi = np.arctan2(py, px) - ref_phi
    pt = (px**2 + py**2)**0.5
    return pt * np.cos(new_phi), pt * np.sin(new_phi)
