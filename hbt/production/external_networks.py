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
    set_ak_column, attach_behavior, flat_np_view, EMPTY_FLOAT, ak_concatenate_safe, layout_ak_array,
)
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.types import Any, Literal

from hbt.util import MET_COLUMN, rotate_px_py

np = maybe_import("numpy")
scipy = maybe_import("scipy")
ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)

# helper functions
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)

BTagType = Literal["pnet", "upart", "none"]


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
        "HHBJet.{pt,eta,phi,mass,hhbtag,btagPNet*,btagUParTAK4*}",
        "FatJet.{eta,phi,pt,mass}",
        MET_COLUMN("{pt,phi,covXX,covXY,covYY}"),
    }

    # which type of btagging variables to use
    btag_type: BTagType = "pnet"

    # limited chunk size to avoid memory issues
    max_chunk_size: int = 10_000

    # the empty value to insert to output columns in case of missing or broken values
    empty_value: float = EMPTY_FLOAT

    # optionally save input features
    produce_features: bool | None = None
    features_prefix: str = ""

    # aggregation method
    ensemble_aggregation: Literal["mean"] | None = None

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

    @property
    def local_model_path(self) -> str | law.LocalFileTarget | None:
        # when set to an existing location, the model is loaded from there instead of the external files
        return None

    def init_func(self, **kwargs) -> None:
        super().init_func(**kwargs)

        assert self.btag_type in {"pnet", "upart", "none"}

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
        super().requires_func(task=task, reqs=reqs, **kwargs)

        if self.local_model_path is None and "external_files" not in reqs:
            from columnflow.tasks.external import BundleExternalFiles
            reqs["external_files"] = BundleExternalFiles.req(task)

    def setup_func(self, task: law.Task, reqs: dict[str, DotDict[str, Any]], **kwargs) -> None:
        super().setup_func(task=task, reqs=reqs, **kwargs)

        from hbt.ml.evaluators import TorchEvaluator

        if not getattr(task, "taf_torch_evaluator", None):
            task.taf_torch_evaluator = TorchEvaluator()
        self.evaluator = task.taf_torch_evaluator

        if (local_model_path := self.local_model_path) is None:
            bundle = reqs["external_files"]
            bundle.files
            model_path = getattr(bundle.files, self.external_name)
            self.evaluator.add_model(self.cls_name, model_path.abspath)
        else:
            self.evaluator.add_model(self.cls_name, law.target.file.get_path(local_model_path))

        # categorical values handled by the network
        # (names and values from training code that was aligned to KLUB notation)
        self.embedding_expected_inputs = {
            "pair_type": [0, 1, 2],  # old KLUB naming, 0: mutau, 1: etau, 2: tautau
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
        super().teardown_func(task=task, **kwargs)

        if (evaluator := getattr(task, "taf_torch_evaluator", None)):
            evaluator.stop()
        task.taf_torch_evaluator = None
        self.evaluator = None

    def call_func(self, events: ak.Array, task: law.Task, **kwargs) -> ak.Array:
        # start the evaluator
        if not self.evaluator.running:
            self.evaluator.start()

        # precompute variables stored directly in the events for easier access later on
        events = self.update_events(events)

        # prepare continuous and categorical network inputs
        # ! NOTE: the order in which inputs are assigned to the DotDicts must match exactly the networks' feature order
        cont = DotDict()
        cat = DotDict()
        self.define_categorical_inputs(events, cat)
        self.define_continuous_inputs(events, cont, cat)

        # apply event mask to all features
        event_mask = self.define_event_mask(events, cat, cont)
        if (n_mask := ak.sum(event_mask)) == 0:
            task.logger.warning(
                f"{self.cls_name}: 0 / {len(events)} selected for evaluation ({task.dataset_inst.name})",
            )
        for n, v in cont.items():
            cont[n] = v[event_mask]
        for n, v in cat.items():
            cat[n] = v[event_mask]

        # check for non-finite continuous inputs
        invalid_stats = {}
        for n, v in cont.items():
            if (m := np.asarray(~np.isfinite(v))).any():
                invalid_stats[n] = m
        if invalid_stats:
            raise Exception(
                f"found {len(invalid_stats)} continuous feature(s) in {n_mask} events with non-finite values:\n  - " +
                "\n  - ".join(f"{n}: {m.sum()} -> {100 * m.mean():.2f}%" for n, m in invalid_stats.items()),
            )

        # build continuous inputs
        continuous_inputs = np.concatenate(
            [np.asarray(t[..., None], dtype=np.float32) for t in cont.values()],
            axis=1,
        )

        # build categorical inputs
        categorical_inputs = np.concatenate(
            [np.asarray(t[..., None], dtype=np.int32) for t in cat.values()],
            axis=1,
        )

        # evaluate the model
        scores = self.evaluator(self.cls_name, categorical_inputs, continuous_inputs)

        # aggregate ensemble output if needed
        if scores.ndim == 3:
            scores = self.aggregate_ensemble_output(scores)

        # validate scores (probably replacing nans)
        scores = self.validate_scores(scores)

        # store scores in events
        events = self.store_scores(events, scores, event_mask)

        # optionally store input features
        if self.produce_features:
            for name in cont:
                values = self.empty_value * np.ones(len(events), dtype=np.float32)
                values[event_mask] = ak.flatten(np.asarray(cont[name][..., None], dtype=np.float32))
                events = set_ak_column_f32(events, f"{self.features_prefix}{self.cls_name}_{name}", values)
            for name in cat:
                values = int(self.empty_value) * np.ones(len(events), dtype=np.int32)
                values[event_mask] = ak.flatten(np.asarray(cat[name][..., None], dtype=np.int32))
                events = set_ak_column_i32(events, f"{self.features_prefix}{self.cls_name}_{name}", values)

        return events

    def update_events(self, events: ak.Array) -> ak.Array:
        # ensure coffea behavior for HHBJets
        events = self[attach_coffea_behavior](events, collections={"HHBJet": "Jet"})

        # store visible tau decay products, consider them all as tau types
        vis_tau = attach_behavior(
            ak_concatenate_safe((events.Electron, events.Muon, events.Tau), axis=1),
            type_name="Tau",
        )
        events = set_ak_column(events, "feat_vis_tau", vis_tau)

        # compute angle from visible mother particle of vis_tau1 and vis_tau2
        # used to rotate the kinematics of dau{1,2}, met, bjet{1,2} and fatjets relative to it
        dilep_phi = np.arctan2(
            vis_tau[:, 0].py + vis_tau[:, 1].py,
            vis_tau[:, 0].px + vis_tau[:, 1].px,
            dtype=np.float64,
        )
        events = set_ak_column(events, "feat_dilep_phi", dilep_phi)

        return events

    def define_categorical_inputs(self, events: ak.Array, cat: DotDict) -> None:
        # define the pair type (KLUBs channel id)
        pair_type = np.zeros(len(events), dtype=np.int32)
        for channel_id, pair_type_id in self.channel_id_to_pair_type.items():
            pair_type[events.channel_id == channel_id] = pair_type_id
        cat.pair_type = pair_type

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
        cat.dm1 = np.where(dm1 == 2, 1, dm1)
        cat.dm2 = np.where(dm2 == 2, 1, dm2)

        # visible tau charge
        cat.vis_tau1_charge = np.asarray(events.feat_vis_tau[:, 0].charge, dtype=np.int32)
        cat.vis_tau2_charge = np.asarray(events.feat_vis_tau[:, 1].charge, dtype=np.int32)

        # whether the events is resolved, boosted or neither
        cat.has_jet_pair = np.asarray(ak.num(events.HHBJet) >= 2, dtype=np.int32)
        cat.has_fatjet = np.asarray(ak.num(events.FatJet) >= 1, dtype=np.int32)

    def define_continuous_inputs(self, events: ak.Array, cont: DotDict, cat: DotDict) -> None:
        rot = functools.partial(rotate_px_py, ref_phi=-events.feat_dilep_phi)
        has_jet_pair = np.asarray(cat.has_jet_pair, dtype=bool)
        has_fatjet = np.asarray(cat.has_fatjet, dtype=bool)

        # MET variables
        _met = events[self.config_inst.x.met_name]
        cont.met_px, cont.met_py = rot(_met.pt * np.cos(_met.phi), _met.pt * np.sin(_met.phi))
        cont.met_cov00, cont.met_cov01, cont.met_cov11 = _met.covXX, _met.covXY, _met.covYY

        # lepton 1
        cont.vis_tau1_px, cont.vis_tau1_py = rot(events.feat_vis_tau.px[:, 0], events.feat_vis_tau.py[:, 0])
        cont.vis_tau1_pz, cont.vis_tau1_e = events.feat_vis_tau.pz[:, 0], events.feat_vis_tau.energy[:, 0]

        # lepton 2
        cont.vis_tau2_px, cont.vis_tau2_py = rot(events.feat_vis_tau.px[:, 1], events.feat_vis_tau.py[:, 1])
        cont.vis_tau2_pz, cont.vis_tau2_e = events.feat_vis_tau.pz[:, 1], events.feat_vis_tau.energy[:, 1]

        # there might be less than two jets or no fatjet, so pad them
        bjets = ak.pad_none(events.HHBJet, 2, axis=1)
        fatjet = ak.pad_none(events.FatJet, 1, axis=1)[:, 0]

        # bjet 1
        cont.bjet1_px, cont.bjet1_py = rot(bjets[:, 0].px, bjets[:, 0].py)
        cont.bjet1_pz, cont.bjet1_e = bjets[:, 0].pz, bjets[:, 0].energy
        if self.btag_type == "pnet":
            cont.bjet1_tag_b = bjets[:, 0].btagPNetB
            cont.bjet1_tag_cvsb = bjets[:, 0].btagPNetCvB
            cont.bjet1_tag_cvsl = bjets[:, 0].btagPNetCvL
        elif self.btag_type == "upart":
            cont.bjet1_tag_b = bjets[:, 0].btagUParTAK4B
        cont.bjet1_hhbtag = bjets[:, 0].hhbtag

        # bjet 2
        cont.bjet2_px, cont.bjet2_py = rot(bjets[:, 1].px, bjets[:, 1].py)
        cont.bjet2_pz, cont.bjet2_e = bjets[:, 1].pz, bjets[:, 1].energy
        if self.btag_type == "pnet":
            cont.bjet2_tag_b = bjets[:, 1].btagPNetB
            cont.bjet2_tag_cvsb = bjets[:, 1].btagPNetCvB
            cont.bjet2_tag_cvsl = bjets[:, 1].btagPNetCvL
        elif self.btag_type == "upart":
            cont.bjet2_tag_b = bjets[:, 1].btagUParTAK4B
        cont.bjet2_hhbtag = bjets[:, 1].hhbtag

        # fatjet variables
        cont.fatjet_px, cont.fatjet_py = rot(fatjet.px, fatjet.py)
        cont.fatjet_pz, cont.fatjet_e = fatjet.pz, fatjet.energy

        # mask values of various fields as done during training of the network
        def mask_fields(mask, value, *fields):
            if not ak.any(mask):
                return
            for field in fields:
                if field not in cont:
                    continue
                arr = flat_np_view(ak.fill_none(cont[field], value, axis=0), copy=True)
                arr[flat_np_view(mask)] = value
                cont[field] = layout_ak_array(arr, cont[field]) if cont[field].ndim > 1 else arr

        mask_fields(~has_jet_pair, 0.0, "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e")
        mask_fields(~has_jet_pair, 0.0, "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e")
        mask_fields(~has_jet_pair, -1.0, "bjet1_tag_b", "bjet1_tag_cvsb", "bjet1_tag_cvsl", "bjet1_hhbtag")
        mask_fields(~has_jet_pair, -1.0, "bjet2_tag_b", "bjet2_tag_cvsb", "bjet2_tag_cvsl", "bjet2_hhbtag")
        mask_fields(~has_fatjet, 0.0, "fatjet_px", "fatjet_py", "fatjet_pz", "fatjet_e")

        # combine daus
        cont.htt_e = cont.vis_tau1_e + cont.vis_tau2_e
        cont.htt_px = cont.vis_tau1_px + cont.vis_tau2_px
        cont.htt_py = cont.vis_tau1_py + cont.vis_tau2_py
        cont.htt_pz = cont.vis_tau1_pz + cont.vis_tau2_pz

        # combine bjets
        cont.hbb_e = cont.bjet1_e + cont.bjet2_e
        cont.hbb_px = cont.bjet1_px + cont.bjet2_px
        cont.hbb_py = cont.bjet1_py + cont.bjet2_py
        cont.hbb_pz = cont.bjet1_pz + cont.bjet2_pz
        mask_fields(~has_jet_pair, 0.0, "hbb_e", "hbb_px", "hbb_py", "hbb_pz")

        # htt + hbb
        cont.htthbb_e = cont.htt_e + cont.hbb_e
        cont.htthbb_px = cont.htt_px + cont.hbb_px
        cont.htthbb_py = cont.htt_py + cont.hbb_py
        cont.htthbb_pz = cont.htt_pz + cont.hbb_pz
        mask_fields(~has_jet_pair, 0.0, "htthbb_e", "htthbb_px", "htthbb_py", "htthbb_pz")

        # htt + fatjet
        cont.httfatjet_e = cont.htt_e + cont.fatjet_e
        cont.httfatjet_px = cont.htt_px + cont.fatjet_px
        cont.httfatjet_py = cont.htt_py + cont.fatjet_py
        cont.httfatjet_pz = cont.htt_pz + cont.fatjet_pz
        mask_fields(~has_fatjet, 0.0, "httfatjet_e", "httfatjet_px", "httfatjet_py", "httfatjet_pz")

    def define_event_mask(self, events: ak.Array, cat: DotDict, cont: DotDict) -> ak.Array:
        return (
            np.isin(cat.pair_type, self.embedding_expected_inputs["pair_type"]) &
            np.isin(cat.dm1, self.embedding_expected_inputs["decay_mode1"]) &
            np.isin(cat.dm2, self.embedding_expected_inputs["decay_mode2"]) &
            np.isin(cat.vis_tau1_charge, self.embedding_expected_inputs["charge1"]) &
            np.isin(cat.vis_tau2_charge, self.embedding_expected_inputs["charge2"]) &
            ((cat.has_jet_pair == 1) | (cat.has_fatjet == 1))
        )

    def aggregate_ensemble_output(self, scores: np.ndarray) -> np.ndarray:
        if self.ensemble_aggregation == "mean":
            return np.mean(scores, axis=1)

        if not self.ensemble_aggregation:
            return scores

        raise ValueError(f"invalid ensemble aggregation method: {self.ensemble_aggregation}")

    def validate_scores(self, scores: np.ndarray) -> np.ndarray:
        # in very rare cases (1 in 25k), the network output can be none, likely for numerical reasons,
        # so issue a warning and set them to a default value
        nan_mask = ~np.isfinite(scores)
        if np.any(nan_mask):
            nan_mask_event = nan_mask.any(axis=1)
            msg = f"{nan_mask_event.sum()} / {len(scores)} events ({100 * nan_mask_event.mean():.2f}%) have NaN scores"
            # raise when this happens too often
            if nan_mask_event.mean() >= 0.005:
                raise Exception(f"{msg}; this should not happen, so please debug")
            # raise when only some columns in events are nan, but not all
            uneven_nan_mask = nan_mask_event & ~nan_mask.all(axis=1)
            if uneven_nan_mask.any():
                raise Exception(
                    f"{msg}, of which {uneven_nan_mask.sum()} only have them in some output nodes; this should not "
                    "happen, so please debug",
                )
            # warn for the remainder of cases
            logger.warning(f"{msg}; setting them to {self.empty_value}")
            scores[nan_mask] = self.empty_value

        return scores

    def store_scores(self, events: ak.Array, scores: Any, event_mask: ak.Array) -> ak.Array:
        # prepare output columns with the shape of the original events and assign values into them
        for i, column in enumerate(self.output_columns):
            values = self.empty_value * np.ones(len(events), dtype=np.float32)
            values[event_mask] = scores[:, i]
            events = set_ak_column_f32(events, column, values)

        return events


class torch_test_dnn(_external_dnn):
    exposed = True


class torch_simple_kl01(_external_dnn):
    exposed = True


#
# end-to-end model tests
#

class _e2e_dnn(_external_dnn):

    latent_dim = 50

    def init_func(self, **kwargs) -> None:
        super().init_func(**kwargs)

        # store names of output columns for latent scores
        self.latent_output_columns = [
            f"{self.output_prefix}_bin{i}"
            for i in range(self.latent_dim)
        ]
        self.produces |= set(self.latent_output_columns)

    def validate_scores(self, scores: Any) -> Any:
        # scores is a tuple of two arrays of scores that have no softmax applied yet, so apply it first, then perform
        # the usual checks
        return type(scores)(
            super(_e2e_dnn, self).validate_scores(scipy.special.softmax(_scores, axis=1))
            for _scores in scores
        )

    def store_scores(self, events: ak.Array, scores: Any, event_mask: ak.Array) -> ak.Array:
        process_scores, latent_scores = scores

        # check the latent dimension
        if latent_scores.shape[1] != self.latent_dim:
            raise ValueError(
                f"expected latent scores to have dimension {self.latent_dim}, but got {latent_scores.shape[1]}",
            )

        # store the multi-class scores as usual
        events = super(_e2e_dnn, self).store_scores(events, process_scores, event_mask)

        # store latent scores
        for i, column in enumerate(self.latent_output_columns):
            values = self.empty_value * np.ones(len(events), dtype=np.float32)
            values[event_mask] = latent_scores[:, i]
            events = set_ak_column_f32(events, column, values)

        return events


class e2e_model1(_e2e_dnn):
    exposed = True
