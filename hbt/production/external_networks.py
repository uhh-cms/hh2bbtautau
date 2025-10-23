# coding: utf-8

"""
Producers for evaluating torch-based models.
"""

from __future__ import annotations

import functools

import law

from columnflow.production import Producer
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import (
    set_ak_column, attach_behavior, flat_np_view, EMPTY_FLOAT, default_coffea_collections,
)
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.types import Any

from hbt.util import MET_COLUMN

np = maybe_import("numpy")
ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)

# helper functions
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


def rotate_to_phi(ref_phi: ak.Array, px: ak.Array, py: ak.Array) -> tuple[ak.Array, ak.Array]:
    """
    Rotates a momentum vector extracted from *events* in the transverse plane to a reference phi
    angle *ref_phi*. Returns the rotated px and py components in a 2-tuple.
    """
    new_phi = np.arctan2(py, px) - ref_phi
    pt = (px**2 + py**2)**0.5
    return pt * np.cos(new_phi), pt * np.sin(new_phi)


class _external_dnn(Producer):
    """
    Base class for evaluating DNNs trained externally with PyTorch and our "standard" set of input features.
    """

    uses = {
        attach_coffea_behavior,
        "channel_id",
        "event",
        "Tau.{eta,phi,pt,mass,charge,decayMode}",
        "Electron.{eta,phi,pt,mass,charge}",
        "Muon.{eta,phi,pt,mass,charge}",
        "HHBJet.{pt,eta,phi,mass,hhbtag,btagPNet*}",
        "FatJet.{eta,phi,pt,mass}",
        MET_COLUMN("{pt,phi,covXX,covXY,covYY}"),
    }

    # limited chunk size to avoid memory issues
    max_chunk_size: int = 10_000

    # the empty value to insert to output columns in case of missing or broken values
    empty_value: float = EMPTY_FLOAT

    # optionally save input features
    produce_features: bool | None = None
    features_prefix: str = ""

    # produced columns are added in the deferred init below
    sandbox = dev_sandbox("bash::$HBT_BASE/sandboxes/venv_hbt.sh")

    # not exposed to command line selection
    exposed = False

    @property
    def output_prefix(self) -> str:
        # prefix for output columns
        return self.cls_name

    @property
    def external_name(self) -> str:
        # name of the model bundle in the external files
        return self.cls_name

    def init_func(self, **kwargs) -> None:
        # set feature production options when requested
        if self.produce_features is None:
            self.produce_features = self.config_inst.x.sync
            if not self.features_prefix:
                self.features_prefix = "sync"
        if self.features_prefix and not self.features_prefix.endswith("_"):
            self.features_prefix = f"{self.features_prefix}_"

        # add features to produced columns
        if self.produce_features:
            self.produces.add(f"{self.features_prefix}{self.cls_name}_*")

        # update shifts dynamically
        self.shifts.add("minbias_xs_{up,down}")  # variations of minbias_xs used in met phi correction
        self.shifts.update({  # all calibrations that change jet and lepton momenta
            shift_inst.name
            for shift_inst in self.config_inst.shifts
            if shift_inst.has_tag({"jec", "jer", "tec", "eec", "eer"})
        })

        # output column names
        # (could be generalized to allow inheriting classes to define different targets)
        self.output_columns = [
            f"{self.output_prefix}_{name}"
            for name in ["hh", "tt", "dy"]
        ]

        # update produced columns
        self.produces |= set(self.output_columns)

    def requires_func(self, task: law.Task, reqs: dict, **kwargs) -> None:
        if "external_files" in reqs:
            return

        from columnflow.tasks.external import BundleExternalFiles
        reqs["external_files"] = BundleExternalFiles.req(task)

    def setup_func(self, task: law.Task, reqs: dict[str, DotDict[str, Any]], **kwargs) -> None:
        from hbt.ml.torch_evaluator import TorchEvaluator

        if not getattr(task, "taf_torch_evaluator", None):
            task.taf_torch_evaluator = TorchEvaluator()
        self.evaluator = task.taf_torch_evaluator

        bundle = reqs["external_files"]
        bundle.files
        model_path = getattr(bundle.files, self.external_name)
        self.evaluator.add_model(self.cls_name, model_path.abspath)

        # categorical values handled by the network
        # (names and values from training code that was aligned to KLUB notation)
        self.embedding_expected_inputs = {
            "channel_id": [1, 2, 3],  # see mapping below
            "decay_mode1": [-1, 0, 1, 10, 11],  # -1 for e/mu
            "decay_mode2": [0, 1, 10, 11],
            "charge1": [-1, 1],
            "charge2": [-1, 1],
            "is_boosted": [0, 1],  # whether a selected fatjet is present
            "has_jet_pair": [0, 1],  # whether two or more jets are present
        }

    def teardown_func(self, task: law.Task, **kwargs) -> None:
        """
        Stops the Torch evaluator.
        """
        if (evaluator := getattr(task, "taf_torch_evaluator", None)):
            evaluator.stop()
        task.taf_torch_evaluator = None
        self.evaluator = None

    def call_func(self, events: ak.Array, **kwargs) -> ak.Array:
        # start the evaluator
        if not self.evaluator.running:
            self.evaluator.start()

        # ensure coffea behavior
        events = self[attach_coffea_behavior](
            events,
            collections={"HHBJet": default_coffea_collections["Jet"]},
            **kwargs,
        )

        # get the channel id
        channel_id = events.channel_id

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
            np.isin(channel_id, self.embedding_expected_inputs["channel_id"]) &
            np.isin(dm1, self.embedding_expected_inputs["decay_mode1"]) &
            np.isin(dm2, self.embedding_expected_inputs["decay_mode2"]) &
            np.isin(vis_tau1.charge, self.embedding_expected_inputs["charge1"]) &
            np.isin(vis_tau2.charge, self.embedding_expected_inputs["charge2"]) &
            (has_jet_pair | has_fatjet)
        )

        # hook to update the event mask base on additional event info
        event_mask = self.update_event_mask(events, event_mask)

        # apply to all arrays needed until now
        _events = events[event_mask]
        channel_id = channel_id[event_mask]
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
        f.bjet1_tag_b = bjets[:, 0].btagPNetB
        f.bjet1_tag_cvsb = bjets[:, 0].btagPNetCvB
        f.bjet1_tag_cvsl = bjets[:, 0].btagPNetCvL
        f.bjet1_hhbtag = bjets[:, 0].hhbtag

        # bjet 2
        f.bjet2_px, f.bjet2_py = rotate_to_phi(phi_lep, bjets[:, 1].px, bjets[:, 1].py)
        f.bjet2_pz, f.bjet2_e = bjets[:, 1].pz, bjets[:, 1].energy
        f.bjet2_tag_b = bjets[:, 1].btagPNetB
        f.bjet2_tag_cvsb = bjets[:, 1].btagPNetCvB
        f.bjet2_tag_cvsl = bjets[:, 1].btagPNetCvL
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
        mask_values(~has_jet_pair, -1.0, "bjet1_tag_b", "bjet1_tag_cvsb", "bjet1_tag_cvsl", "bjet1_hhbtag")
        mask_values(~has_jet_pair, -1.0, "bjet2_tag_b", "bjet2_tag_cvsb", "bjet2_tag_cvsl", "bjet2_hhbtag")
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

        # assign categorical inputs via names too
        f.channel_id = channel_id
        f.dm1 = dm1
        f.dm2 = dm2
        f.vis_tau1_charge = vis_tau1.charge
        f.vis_tau2_charge = vis_tau2.charge
        f.has_jet_pair = has_jet_pair
        f.has_fatjet = has_fatjet

        # build continous inputs
        # (order exactly as documented in link above)
        continous_inputs = [
            np.asarray(t[..., None], dtype=np.float32) for t in [
                f.met_px, f.met_py, f.met_cov00, f.met_cov01, f.met_cov11,
                f.vis_tau1_px, f.vis_tau1_py, f.vis_tau1_pz, f.vis_tau1_e,
                f.vis_tau2_px, f.vis_tau2_py, f.vis_tau2_pz, f.vis_tau2_e,
                f.bjet1_px, f.bjet1_py, f.bjet1_pz, f.bjet1_e, f.bjet1_tag_b, f.bjet1_tag_cvsb, f.bjet1_tag_cvsl,
                f.bjet1_hhbtag,
                f.bjet2_px, f.bjet2_py, f.bjet2_pz, f.bjet2_e, f.bjet2_tag_b, f.bjet2_tag_cvsb, f.bjet2_tag_cvsl,
                f.bjet2_hhbtag,
                f.fatjet_px, f.fatjet_py, f.fatjet_pz, f.fatjet_e,
                f.htt_e, f.htt_px, f.htt_py, f.htt_pz,
                f.hbb_e, f.hbb_px, f.hbb_py, f.hbb_pz,
                f.htthbb_e, f.htthbb_px, f.htthbb_py, f.htthbb_pz,
                f.httfatjet_e, f.httfatjet_px, f.httfatjet_py, f.httfatjet_pz,
            ]
            if t is not None
        ]

        # build categorical inputs
        # (order exactly as documented in link above)
        categorical_inputs = [
            np.asarray(t[..., None], dtype=np.int32) for t in [
                f.channel_id,
                f.dm1, f.dm2,
                f.vis_tau1_charge, f.vis_tau2_charge,
                f.has_jet_pair, f.has_fatjet,
            ] if t is not None
        ]

        # evaluate the model
        scores = self.evaluator(
            self.cls_name,
            (
                np.concatenate(categorical_inputs, axis=1),
                np.concatenate(continous_inputs, axis=1),
            ),
        )

        # in very rare cases (1 in 25k), the network output can be none, likely for numerical reasons,
        # so issue a warning and set them to a default value
        nan_mask = ~np.isfinite(scores)
        if np.any(nan_mask):
            logger.warning(
                f"{nan_mask.sum() // scores.shape[1]} out of {scores.shape[0]} events have NaN scores; "
                f"setting them to {self.empty_value}",
            )
            scores[nan_mask] = self.empty_value

        # prepare output columns with the shape of the original events and assign values into them
        for i, column in enumerate(self.output_columns):
            values = self.empty_value * np.ones(len(events), dtype=np.float32)
            values[event_mask] = scores[:, i]
            events = set_ak_column_f32(events, column, values)

        if self.produce_features:
            # store input columns for sync
            cont_inputs_cols = [
                "met_px", "met_py", "met_cov00", "met_cov01", "met_cov11",
                "vis_tau1_px", "vis_tau1_py", "vis_tau1_pz", "vis_tau1_e",
                "vis_tau2_px", "vis_tau2_py", "vis_tau2_pz", "vis_tau2_e",
                "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e", "bjet1_tag_b", "bjet1_tag_cvsb", "bjet1_tag_cvsl",
                "bjet1_hhbtag",
                "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e", "bjet2_tag_b", "bjet2_tag_cvsb", "bjet2_tag_cvsl",
                "bjet2_hhbtag",
                "fatjet_px", "fatjet_py", "fatjet_pz", "fatjet_e",
                "htt_e", "htt_px", "htt_py", "htt_pz",
                "hbb_e", "hbb_px", "hbb_py", "hbb_pz",
                "htthbb_e", "htthbb_px", "htthbb_py", "htthbb_pz",
                "httfatjet_e", "httfatjet_px", "httfatjet_py", "httfatjet_pz",
            ]
            cat_inputs_cols = [
                "channel_id", "dm1", "dm2", "vis_tau1_charge", "vis_tau2_charge", "has_jet_pair", "has_fatjet",
            ]
            for c in cont_inputs_cols + cat_inputs_cols:
                values = self.empty_value * np.ones(len(events), dtype=np.float32)
                values[event_mask] = ak.flatten(np.asarray(f[c][..., None], dtype=np.float32))
                events = set_ak_column_f32(events, f"{self.features_prefix}{self.cls_name}_{c}", values)

        return events

    def update_event_mask(self, events: ak.Array, event_mask: ak.Array) -> ak.Array:
        return event_mask


class torch_test_dnn(_external_dnn):
    exposed = True
