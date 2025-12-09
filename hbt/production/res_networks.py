# coding: utf-8

"""
Producer for evaluating the pDNN developed for the resonant run 2 analysis.
See https://github.com/uhh-cms/tautauNN
"""

from __future__ import annotations

import functools

import law

from columnflow.production import Producer
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import (
    set_ak_column, attach_behavior, flat_np_view, EMPTY_FLOAT, default_coffea_collections, ak_concatenate_safe,
    layout_ak_array,
)
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.types import Any

from hbt.util import MET_COLUMN

np = maybe_import("numpy")
ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)

# helper functions
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)


def rotate_to_phi(ref_phi: ak.Array, px: ak.Array, py: ak.Array) -> tuple[ak.Array, ak.Array]:
    """
    Rotates a momentum vector extracted from *events* in the transverse plane to a reference phi
    angle *ref_phi*. Returns the rotated px and py components in a 2-tuple.
    """
    new_phi = np.arctan2(py, px, dtype=np.float64) - ref_phi
    pt = (px**2 + py**2)**0.5
    return pt * np.cos(new_phi), pt * np.sin(new_phi)


class _res_dnn_evaluation(Producer):
    """
    Base producer for dnn evaluations of the resonant run 2 analyses, whose models are considered external and thus part
    of producers rather than standalone ml model objects. The output scores are classifying if incoming events are HH,
    Drell-Yan or ttbar. The network uses continous, categorical and parametrized inputs. A list of all inputs in the
    correct order can be found in the tautauNN repo:
    https://github.com/uhh-cms/tautauNN/blob/f1ca194/evaluation/interface.py#L67
    """

    uses = {
        attach_coffea_behavior,
        "channel_id",
        "event",
        "Tau.{eta,phi,pt,mass,charge,decayMode}",
        "Electron.{eta,phi,pt,mass,charge}",
        "Muon.{eta,phi,pt,mass,charge}",
        "HHBJet.{pt,eta,phi,mass,hhbtag,btagDeepFlav*,btagPNet*}",
        "FatJet.{eta,phi,pt,mass}",
        MET_COLUMN("{pt,phi,covXX,covXY,covYY}"),
    }

    # whether to use pnet instead of deepflavor for btagging variables
    use_pnet: bool = False

    # whether the model is parameterized in mass, spin and year
    # (this is a slight forward declaration but simplifies the code reasonably well in our use case)
    parametrized: bool | None = None

    # directory of the unpacked model archive (no subdirectory is expected when None)
    dir_name: str | None = None

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

    def requires_func(self, task: law.Task, reqs: dict, **kwargs) -> None:
        super().requires_func(task=task, reqs=reqs, **kwargs)

        if "external_files" in reqs:
            return

        from columnflow.tasks.external import BundleExternalFiles
        reqs["external_files"] = BundleExternalFiles.req(task)

    def setup_func(self, task: law.Task, reqs: dict[str, DotDict[str, Any]], **kwargs) -> None:
        super().setup_func(task=task, reqs=reqs, **kwargs)

        from hbt.ml.evaluators import TFEvaluator
        if not getattr(task, "taf_tf_evaluator", None):
            task.taf_tf_evaluator = TFEvaluator()
        self.evaluator = task.taf_tf_evaluator

        # some checks
        if not isinstance(self.parametrized, bool):
            raise AttributeError("'parametrized' must be set in the producer configuration")

        # unpack the model archive
        bundle = reqs["external_files"]
        bundle.files
        model_dir = bundle.files_dir.child(f"{self.external_name}_unpacked", type="d")
        getattr(bundle.files, self.external_name).load(model_dir, formatter="tar")
        if self.dir_name:
            model_dir = model_dir.child(self.dir_name, type="d")

        # setup the evaluator
        self.evaluator.add_model(self.cls_name, model_dir.abspath, signature_key="serving_default")

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
            (2024, ""): 3,
        }[(self.config_inst.campaign.x.year, self.config_inst.campaign.x.postfix)]

    def teardown_func(self, task: law.Task, **kwargs) -> None:
        """
        Stops the TF evaluator.
        """
        if (evaluator := getattr(task, "taf_tf_evaluator", None)):
            evaluator.stop()
        task.taf_tf_evaluator = None
        self.evaluator = None

    def call_func(self, events: ak.Array, **kwargs) -> ak.Array:
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
        n_mask = ak.sum(event_mask)
        for n, v in cont.items():
            cont[n] = v[event_mask]
        for n, v in cat.items():
            cat[n] = v[event_mask]

        # build continuous inputs
        continuous_inputs = [
            np.asarray(t[..., None], dtype=np.float32) for t in [
                *cont.values(),
                (self.mass * np.ones(n_mask, dtype=np.float32)) if self.parametrized else None,
            ]
            if t is not None
        ]

        # build categorical inputs
        categorical_inputs = [
            np.asarray(t[..., None], dtype=np.int32) for t in [
                *cat.values(),
                (self.year_flag * np.ones(n_mask, dtype=np.int32)) if self.parametrized else None,
                (self.spin * np.ones(n_mask, dtype=np.int32)) if self.parametrized else None,
            ] if t is not None
        ]

        # evaluate the model
        scores = self.evaluator(
            self.cls_name,
            inputs=[
                np.concatenate(continuous_inputs, axis=1),
                np.concatenate(categorical_inputs, axis=1),
            ],
        )

        # in very rare cases (1 in 25k), the network output can be none, likely for numerical reasons,
        # so issue a warning and set them to a default value
        nan_mask = ~np.isfinite(scores)
        if np.any(nan_mask):
            logger.warning(
                f"{nan_mask.sum() // scores.shape[1]} out of {scores.shape[0]} events have NaN scores; setting them to "
                f"{self.empty_value}",
            )
            scores[nan_mask] = self.empty_value

        # prepare output columns with the shape of the original events and assign values into them
        for i, column in enumerate(self.output_columns):
            values = self.empty_value * np.ones(len(events), dtype=np.float32)
            values[event_mask] = scores[:, i]
            events = set_ak_column_f32(events, column, values)

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
        events = self[attach_coffea_behavior](
            events,
            collections={"HHBJet": default_coffea_collections["Jet"]},
        )

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
        cat.vis_tau1_charge = events.feat_vis_tau[:, 0].charge
        cat.vis_tau2_charge = events.feat_vis_tau[:, 1].charge

        # whether the events is resolved, boosted or neither
        cat.has_jet_pair = ak.num(events.HHBJet) >= 2
        cat.has_fatjet = ak.num(events.FatJet) >= 1

    def define_continuous_inputs(self, events: ak.Array, cont: DotDict, cat: DotDict) -> None:
        rot = functools.partial(rotate_to_phi, events.feat_dilep_phi)

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
        cont.bjet1_tag_b = bjets[:, 0]["btagPNetB" if self.use_pnet else "btagDeepFlavB"]
        cont.bjet1_tag_cvsb = bjets[:, 0]["btagPNetCvB" if self.use_pnet else "btagDeepFlavCvB"]
        cont.bjet1_tag_cvsl = bjets[:, 0]["btagPNetCvL" if self.use_pnet else "btagDeepFlavCvL"]
        cont.bjet1_hhbtag = bjets[:, 0].hhbtag

        # bjet 2
        cont.bjet2_px, cont.bjet2_py = rot(bjets[:, 1].px, bjets[:, 1].py)
        cont.bjet2_pz, cont.bjet2_e = bjets[:, 1].pz, bjets[:, 1].energy
        cont.bjet2_tag_b = bjets[:, 1]["btagPNetB" if self.use_pnet else "btagDeepFlavB"]
        cont.bjet2_tag_cvsb = bjets[:, 1]["btagPNetCvB" if self.use_pnet else "btagDeepFlavCvB"]
        cont.bjet2_tag_cvsl = bjets[:, 1]["btagPNetCvL" if self.use_pnet else "btagDeepFlavCvL"]
        cont.bjet2_hhbtag = bjets[:, 1].hhbtag

        # fatjet variables
        cont.fatjet_px, cont.fatjet_py = rot(fatjet.px, fatjet.py)
        cont.fatjet_pz, cont.fatjet_e = fatjet.pz, fatjet.energy

        # mask values of various fields as done during training of the network
        def mask_fields(mask, value, *fields):
            if not ak.any(mask):
                return
            for field in fields:
                arr = flat_np_view(ak.fill_none(cont[field], value, axis=0), copy=True)
                arr[flat_np_view(mask)] = value
                cont[field] = layout_ak_array(arr, cont[field]) if cont[field].ndim > 1 else arr

        mask_fields(~cat.has_jet_pair, 0.0, "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e")
        mask_fields(~cat.has_jet_pair, 0.0, "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e")
        mask_fields(~cat.has_jet_pair, -1.0, "bjet1_tag_b", "bjet1_tag_cvsb", "bjet1_tag_cvsl", "bjet1_hhbtag")
        mask_fields(~cat.has_jet_pair, -1.0, "bjet2_tag_b", "bjet2_tag_cvsb", "bjet2_tag_cvsl", "bjet2_hhbtag")
        mask_fields(~cat.has_fatjet, 0.0, "fatjet_px", "fatjet_py", "fatjet_pz", "fatjet_e")

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
        mask_fields(~cat.has_jet_pair, 0.0, "hbb_e", "hbb_px", "hbb_py", "hbb_pz")

        # htt + hbb
        cont.htthbb_e = cont.htt_e + cont.hbb_e
        cont.htthbb_px = cont.htt_px + cont.hbb_px
        cont.htthbb_py = cont.htt_py + cont.hbb_py
        cont.htthbb_pz = cont.htt_pz + cont.hbb_pz
        mask_fields(~cat.has_jet_pair, 0.0, "htthbb_e", "htthbb_px", "htthbb_py", "htthbb_pz")

        # htt + fatjet
        cont.httfatjet_e = cont.htt_e + cont.fatjet_e
        cont.httfatjet_px = cont.htt_px + cont.fatjet_px
        cont.httfatjet_py = cont.htt_py + cont.fatjet_py
        cont.httfatjet_pz = cont.htt_pz + cont.fatjet_pz
        mask_fields(~cat.has_fatjet, 0.0, "httfatjet_e", "httfatjet_px", "httfatjet_py", "httfatjet_pz")

    def define_event_mask(self, events: ak.Array, cat: DotDict, cont: DotDict) -> ak.Array:
        return (
            np.isin(cat.pair_type, self.embedding_expected_inputs["pair_type"]) &
            np.isin(cat.dm1, self.embedding_expected_inputs["decay_mode1"]) &
            np.isin(cat.dm2, self.embedding_expected_inputs["decay_mode2"]) &
            np.isin(cat.vis_tau1_charge, self.embedding_expected_inputs["charge1"]) &
            np.isin(cat.vis_tau2_charge, self.embedding_expected_inputs["charge2"]) &
            (cat.has_jet_pair | cat.has_fatjet) &
            (self.year_flag in self.embedding_expected_inputs["year"])
        )


#
# producers for classification-only networks
# (combined network)
#

class _res_dnn(_res_dnn_evaluation):

    dir_name = "model_fold0"
    output_prefix = "res_dnn"

    def init_func(self, **kwargs) -> None:
        super().init_func(**kwargs)

        # output column names (in this order)
        self.output_columns = [
            f"{self.output_prefix}_{name}"
            for name in ["hh", "tt", "dy"]
        ]

        # update produced columns
        self.produces |= set(self.output_columns)


class res_pdnn(_res_dnn):
    """
    Parameterized network, trained with Radion (spin 0) and Graviton (spin 2) samples up to mX = 3000 GeV in all run 2
    eras.
    """

    parametrized = True
    dir_name = "model_fold0"
    exposed = True
    mass = 500
    spin = 0

    def init_func(self, **kwargs) -> None:
        super().init_func(**kwargs)

        # check spin value and mass values
        if self.spin not in {0, 2}:
            raise ValueError(f"invalid spin value: {self.spin}")
        if self.mass < 250:
            raise ValueError(f"invalid mass value: {self.mass}")


class res_dnn(_res_dnn):
    """
    Non-parameterized network, trained only with Radion (spin 0) samples up to mX = 800 GeV across all run 2 eras.
    """

    parametrized = False
    dir_name = "model_fold0"
    exposed = True


class res_dnn_pnet(res_dnn):
    """
    Same as :py:class:`res_dnn` but using pnet btagging variables and storing inputs.
    """

    external_name = "res_dnn"
    use_pnet = True
    produce_features = True
    output_prefix = "res_dnn_pnet"


#
# producers for multi-output regression networks
# (tobi's regression)
#

class _reg_dnn(_res_dnn_evaluation):

    empty_value = 0.0
    parametrized = False

    def init_func(self, **kwargs) -> None:
        super().init_func(**kwargs)

        # output column names (in this order)
        self.output_columns = [
            f"{self.output_prefix}_nu{i}_{v}"
            for i in range(1, 2 + 1)
            for v in ["px", "py", "pz"]
        ]

        # update produced columns
        self.produces |= set(self.output_columns)


class reg_dnn(_reg_dnn):
    """
    Single regression network, trained with Radion samples and a flat mass range.
    """

    dir_name = "model_fold0_seed0"
    output_prefix = "reg_dnn"
    exposed = True


class reg_dnn_moe(_reg_dnn):
    """
    Mixture of experts regression network, trained with Radion samples and a flat mass range.
    """

    dir_name = "model_fold0_moe"
    output_prefix = "reg_dnn_moe"
    exposed = True


#
# producers for evaluating run 3 models trained with the legacy setup but on run 3 data
#

class _run3_dnn(_res_dnn):

    parametrized = False
    use_pnet = True
    dir_name = None
    fold = None
    n_folds = 5

    @property
    def output_prefix(self) -> str:
        return self.cls_name

    def define_event_mask(self, events: ak.Array, cat: DotDict, cont: DotDict) -> ak.Array:
        event_mask = super().define_event_mask(events, cat, cont)

        # when a fold is defined, select only events that match this fold
        # (all other events were potentially used for the training)
        if self.fold is not None:
            event_fold = events.event % self.n_folds
            event_mask = event_mask & (event_fold == self.fold)

        return event_mask


# derive evaluation producers for all folds
for fold in range(_run3_dnn.n_folds):
    _run3_dnn.derive(f"run3_dnn_fold{fold}_moe", cls_dict={
        "fold": fold,
        "external_name": f"run3_dnn_fold{fold}_moe",
        "exposed": True,
    })


class run3_dnn_simple(_run3_dnn):
    """
    Simple version of the run 3 dnn with a single fold for quick comparisons. Trained with kl 1 and 0.
    """

    fold = None
    external_name = "run3_dnn_simple"
    exposed = True


# same as :py:class:`run3_dnn_simple` but trained for different kl variations
for kl in ["kl1", "kl0", "allkl"]:
    run3_dnn_simple.derive(f"run3_dnn_simple_{kl}", cls_dict={"external_name": f"run3_dnn_simple_{kl}"})


#
# producer for combining the results of all folds
#

class run3_dnn_moe(Producer):

    # require ProduceColumns tasks per fold first when True, otherwise evaluate all folds in the same task
    require_folds = True

    # when require_folds is True, decide whether to remove their outputs after successful combination
    remove_folds = True

    # when require_folds is False, the sandbox must be set
    # sandbox = _run3_dnn.sandbox

    # used and produced columns
    # (used ones are updated dynamically in init_func)
    uses = {"event"}
    produces = {"run3_dnn_moe_{hh,tt,dy}"}

    def init_func(self, **kwargs) -> None:
        # store dnn evaluation classes
        self.dnn_classes = {
            f: _run3_dnn.get_cls(f"run3_dnn_fold{f}_moe")
            for f in range(_run3_dnn.n_folds)
        }

        # update used columns / dependencies
        for dnn_cls in self.dnn_classes.values():
            self.uses.add(f"{dnn_cls.cls_name}_{{hh,tt,dy}}" if self.require_folds else dnn_cls)

    @property
    def require_producers(self) -> list[str] | None:
        return (
            [dnn_cls.cls_name for dnn_cls in self.dnn_classes.values()]
            if self.require_folds
            else None
        )

    def setup_func(
        self,
        task: law.Task,
        reqs: dict,
        inputs: dict,
        reader_targets: law.util.InsertableDict,
        **kwargs,
    ) -> None:
        super().setup_func(task=task, reqs=reqs, inputs=inputs, reader_targets=reader_targets, **kwargs)

        # potentially store references to inputs to remove later on
        self.remove_fold_inputs = (
            [inp["columns"] for inp in inputs["required_producers"].values()]
            if self.require_folds and self.remove_folds
            else []
        )

    def call_func(self, events: ak.Array, **kwargs) -> ak.Array:
        event_fold = events.event % _run3_dnn.n_folds

        # invoke the evaluations of all folds when not requiring them as dedicated producers
        if not self.require_folds:
            for dnn_cls in self.dnn_classes.values():
                events = self[dnn_cls](events, **kwargs)

        for out in ["hh", "tt", "dy"]:
            # fill score from columns at positions with different folds
            score = EMPTY_FLOAT * np.ones(len(events), dtype=np.float32)
            for f in range(_run3_dnn.n_folds):
                score[event_fold == f] = events[f"run3_dnn_fold{f}_moe_{out}"][event_fold == f]

            # assign to new column
            events = set_ak_column_f32(events, f"run3_dnn_moe_{out}", score)

        return events

    def teardown_func(self, task: law.Task, **kwargs) -> None:
        super().teardown_func(task, **kwargs)

        # remove outputs of required fold producers
        for inp in self.remove_fold_inputs:
            inp.remove(silent=True)


#
# producers for vbf networks
#

class _vbf_dnn(_res_dnn_evaluation):
    """
    https://gitlab.cern.ch/cclubbtautau/AnalysisCore/-/blob/19cc9e8c02dda68a37c7d9ee308100a9e1821648/src/HHRun3DNNInterface.cc#L325-339
    """

    uses = {
        *_res_dnn_evaluation.uses,
        "Jet.{pt,eta,phi,mass,assignment_bits}",
        "VBFJet.{pt,eta,phi,mass,btagPNetQvG}",
        "reg_dnn_moe_nu{1,2}_{px,py,pz}",
    }
    require_producers = ["reg_dnn_moe"]

    parametrized = False

    @property
    def output_prefix(self) -> str:
        return self.cls_name

    def init_func(self, **kwargs) -> None:
        super().init_func(**kwargs)

        # output column names (in this order)
        self.output_columns = [
            f"{self.output_prefix}_{name}"
            for name in ["hh_ggf", "tt", "dy", "hh_vbf"]
        ]

        # update produced columns
        self.produces |= set(self.output_columns)

    def update_events(self, events: ak.Array) -> ak.Array:
        events = super().update_events(events)

        # ensure coffea behavior for VBFJets
        events = self[attach_coffea_behavior](
            events,
            collections={"VBFJet": default_coffea_collections["Jet"]},
        )

        padded_vbf_jets = ak.pad_none(events.VBFJet, 2, axis=1)
        events = set_ak_column(events, "padded_vbf_jets", padded_vbf_jets)

        return events

    def define_categorical_inputs(self, events: ak.Array, cat: DotDict) -> None:
        super().define_categorical_inputs(events, cat)

        # dm1 for e/mu is changed from -1 to -999
        cat.dm1[cat.dm1 == -1] = -999

        # add vbf jet pair presence
        cat.has_vbf_jets = ak.num(events.VBFJet) >= 2

    def define_continuous_inputs(self, events: ak.Array, cont: DotDict, cat: DotDict) -> None:
        super().define_continuous_inputs(events, cont, cat)

        # store original features as a shallow copy for reference below
        cont_orig = cont.copy()

        # remove leading met features
        for name in ["met_px", "met_py", "met_cov00", "met_cov01", "met_cov11"]:
            del cont[name]

        # the now leading features are preserved up to bjet2_hhbtag, drop features after that
        idx = list(cont.keys()).index("bjet2_hhbtag")
        for name in list(cont.keys())[idx + 1:]:
            del cont[name]

        # add regressed neutrino momenta
        cont.nu1_px = events.reg_dnn_moe_nu1_px
        cont.nu1_py = events.reg_dnn_moe_nu1_py
        cont.nu1_pz = events.reg_dnn_moe_nu1_pz
        cont.nu2_px = events.reg_dnn_moe_nu2_px
        cont.nu2_py = events.reg_dnn_moe_nu2_py
        cont.nu2_pz = events.reg_dnn_moe_nu2_pz

        # compute nu energies for later use
        nu1_e = (cont.nu1_px**2 + cont.nu1_py**2 + cont.nu1_pz**2)**0.5
        nu2_e = (cont.nu2_px**2 + cont.nu2_py**2 + cont.nu2_pz**2)**0.5

        # fatjet features (flipped order w.r.t. original)
        cont.fatjet_px = cont_orig.fatjet_px
        cont.fatjet_py = cont_orig.fatjet_py
        cont.fatjet_pz = cont_orig.fatjet_pz
        cont.fatjet_e = cont_orig.fatjet_e

        # htt features (with nu's, flipped order w.r.t. original)
        cont.htt_regr_px = cont_orig.htt_px + cont.nu1_px + cont.nu2_px
        cont.htt_regr_py = cont_orig.htt_py + cont.nu1_py + cont.nu2_py
        cont.htt_regr_pz = cont_orig.htt_pz + cont.nu1_pz + cont.nu2_pz
        cont.htt_regr_e = cont_orig.htt_e + nu1_e + nu2_e

        # hbb features (flipped order w.r.t. original)
        cont.hbb_px = cont_orig.hbb_px
        cont.hbb_py = cont_orig.hbb_py
        cont.hbb_pz = cont_orig.hbb_pz
        cont.hbb_e = cont_orig.hbb_e

        # htthbb features (with nu's, also note the non-flipped order)
        cont.htthbb_regr_e = cont_orig.htthbb_e + nu1_e + nu2_e
        cont.htthbb_regr_px = cont_orig.htthbb_px + cont.nu1_px + cont.nu2_px
        cont.htthbb_regr_py = cont_orig.htthbb_py + cont.nu1_py + cont.nu2_py
        cont.htthbb_regr_pz = cont_orig.htthbb_pz + cont.nu1_pz + cont.nu2_pz

        # httfatjet features (with nu's, flipped order w.r.t. original)
        cont.httfatjet_regr_px = cont_orig.httfatjet_px + cont.nu1_px + cont.nu2_px
        cont.httfatjet_regr_py = cont_orig.httfatjet_py + cont.nu1_py + cont.nu2_py
        cont.httfatjet_regr_pz = cont_orig.httfatjet_pz + cont.nu1_pz + cont.nu2_pz
        cont.httfatjet_regr_e = cont_orig.httfatjet_e + nu1_e + nu2_e

        # vbf jet features
        cont.vbfjet1_px = events.padded_vbf_jets[:, 0].px
        cont.vbfjet1_py = events.padded_vbf_jets[:, 0].py
        cont.vbfjet1_pz = events.padded_vbf_jets[:, 0].pz
        cont.vbfjet1_e = events.padded_vbf_jets[:, 0].energy
        cont.vbfjet1_tag_qvsg = events.padded_vbf_jets[:, 0].btagPNetQvG
        cont.vbfjet2_px = events.padded_vbf_jets[:, 1].px
        cont.vbfjet2_py = events.padded_vbf_jets[:, 1].py
        cont.vbfjet2_pz = events.padded_vbf_jets[:, 1].pz
        cont.vbfjet2_e = events.padded_vbf_jets[:, 1].energy
        cont.vbfjet2_tag_qvsg = events.padded_vbf_jets[:, 1].btagPNetQvG

        # mass chi
        # HH_mass - (Hbb_mass - 125.0) - (Htt_mass - 125.0);
        # definition from https://gitlab.cern.ch/cclubbtautau/AnalysisCore/-/blob/cclub_cmssw15010/src/HHRun3DNNInterface.cc?ref_type=heads#L517  # noqa: E501
        cont.m_chi = (
            (cont.htthbb_regr_e**2 - cont.htthbb_regr_px**2 - cont.htthbb_regr_py**2 - cont.htthbb_regr_pz**2)**0.5 -
            ((cont.hbb_e**2 - cont.hbb_px**2 - cont.hbb_py**2 - cont.hbb_pz**2)**0.5 - 125.0) -
            ((cont.htt_regr_e**2 - cont.htt_regr_px**2 - cont.htt_regr_py**2 - cont.htt_regr_pz**2)**0.5 - 125.0)
        )

        # vbf pair variables
        vbfjets = events.padded_vbf_jets[:, :2]
        cont.vbfjj_mass = vbfjets.sum(axis=-1).mass
        cont.vbfjj_delta_r = events.padded_vbf_jets[:, 0].delta_r(events.padded_vbf_jets[:, 1])

        # eta products
        bjets = ak.pad_none(events.HHBJet, 2, axis=1)
        cont.bb_eta_prod = bjets[:, 0].eta * bjets[:, 1].eta
        cont.vbfjj_eta_prod = vbfjets[:, 0].eta * vbfjets[:, 1].eta

        # fox-wolfram moments
        mask_hhbjets_vbfjets = events.Jet.assignment_bits == 0
        # TODO: once fatjets are fully defined, these jets should be cleaned from them with deltaR < 0.8
        central_jets = events.Jet[mask_hhbjets_vbfjets]
        # all central jets + hhbjets + vbfjets
        vbfcjets = ak_concatenate_safe((events.HHBJet, events.VBFJet, central_jets), axis=1)
        vbfc_m2 = (
            ak.sum(vbfcjets.energy, axis=1)**2 -
            ak.sum(vbfcjets.px, axis=1)**2 -
            ak.sum(vbfcjets.py, axis=1)**2 -
            ak.sum(vbfcjets.pz, axis=1)**2
        )
        sum_pt = ak.sum(vbfcjets.pt, axis=1)
        ijet, jjet = ak.unzip(ak.combinations(vbfcjets, 2, axis=1))
        omega_ij = (
            np.cos(ijet.theta) * np.cos(jjet.theta) +
            np.sin(ijet.theta) * np.sin(jjet.theta) * np.cos(ijet.phi - jjet.phi)
        )
        legendre_0 = np.polynomial.legendre.Legendre([1, 0, 0])(omega_ij)
        legendre_2 = np.polynomial.legendre.Legendre([0, 0, 1])(omega_ij)
        weight_s = (
            (ijet.px**2 + ijet.py**2 + ijet.pz**2)**0.5 *
            (jjet.px**2 + jjet.py**2 + jjet.pz**2)**0.5
        ) / vbfc_m2
        weight_t = (ijet.pt * jjet.pt) / (sum_pt**2)
        cont.fw_s_0 = ak.sum(weight_s * legendre_0, axis=1)
        cont.fw_t_0 = ak.sum(weight_t * legendre_0, axis=1)
        cont.fw_1_0 = ak.sum(legendre_0, axis=1)
        cont.fw_s_2 = ak.sum(weight_s * legendre_2, axis=1)
        # TODO: with proper fatjet definition, we should add them to the fw definitions

        # mask missing features with defaults
        def mask_values(mask, value, *fields):
            if not ak.any(mask):
                return
            for field in fields:
                arr = flat_np_view(ak.fill_none(cont[field], value, axis=0), copy=True)
                arr[flat_np_view(mask)] = value
                cont[field] = layout_ak_array(arr, cont[field]) if cont[field].ndim > 1 else arr

        mask_values(~cat.has_jet_pair, 0.0, "hbb_e", "hbb_px", "hbb_py", "hbb_pz")
        mask_values(~cat.has_jet_pair, 0.0, "htthbb_regr_e", "htthbb_regr_px", "htthbb_regr_py", "htthbb_regr_pz")
        mask_values(~cat.has_fatjet, 0.0, "httfatjet_regr_e", "httfatjet_regr_px", "httfatjet_regr_py", "httfatjet_regr_pz")  # noqa: E501
        mask_values(~cat.has_jet_pair, 0.0, "m_chi")
        mask_values(~cat.has_jet_pair, 0.0, "bb_eta_prod")

    def define_event_mask(self, events: ak.Array, cat: DotDict, cont: DotDict) -> ak.Array:
        event_mask = super().define_event_mask(events, cat, cont)

        # add vbf preselection and presence of vbf jets
        vbfjet1 = events.padded_vbf_jets[:, 0]
        vbfjet2 = events.padded_vbf_jets[:, 1]
        vbf_preselection_mask = (
            ak.fill_none((vbfjet1 + vbfjet2).mass > 500.0, False) &
            ak.fill_none(vbfjet1.delta_r(vbfjet2) > 2.5, False)
        )
        event_mask = event_mask & cat.has_vbf_jets & vbf_preselection_mask

        return event_mask


class _vbf_dnn_xvalid(_vbf_dnn):
    """
    Specialization of the VBF DNN producer for cross-validation folds.
    """

    n_folds = 5
    fold = None

    def define_event_mask(self, events: ak.Array, cat: DotDict, cont: DotDict) -> ak.Array:
        event_mask = super().define_event_mask(events, cat, cont)

        # select only events that match this fold
        # (all other events were potentially used for the training)
        event_fold = events.event % self.n_folds
        event_mask = event_mask & (event_fold == self.fold)

        return event_mask


for f in range(_vbf_dnn_xvalid.n_folds):
    _vbf_dnn_xvalid.derive(f"vbf_dnn_fold{f}", cls_dict={
        "fold": f,
        "exposed": True,
    })


class vbf_dnn_moe(Producer):

    # require ProduceColumns tasks per fold first when True, otherwise evaluate all folds in the same task
    require_folds = True

    # when require_folds is True, decide whether to remove their outputs after successful combination
    remove_folds = True

    # when require_folds is False, the sandbox must be set
    # sandbox = _vbf_dnn.sandbox

    # used and produced columns
    # (used ones are updated dynamically in init_func)
    uses = {"event"}
    produces = {"vbf_dnn_moe_{hh_ggf,tt,dy,hh_vbf}"}

    def init_func(self, **kwargs) -> None:
        # store dnn evaluation classes
        self.dnn_classes = {
            f: _vbf_dnn_xvalid.get_cls(f"vbf_dnn_fold{f}")
            for f in range(_run3_dnn.n_folds)
        }

        # update used columns / dependencies
        for dnn_cls in self.dnn_classes.values():
            self.uses.add(f"{dnn_cls.cls_name}_{{hh_ggf,tt,dy,hh_vbf}}" if self.require_folds else dnn_cls)

    @property
    def require_producers(self) -> list[str] | None:
        return (
            [dnn_cls.cls_name for dnn_cls in self.dnn_classes.values()]
            if self.require_folds
            else None
        )

    def setup_func(
        self,
        task: law.Task,
        reqs: dict,
        inputs: dict,
        reader_targets: law.util.InsertableDict,
        **kwargs,
    ) -> None:
        super().setup_func(task=task, reqs=reqs, inputs=inputs, reader_targets=reader_targets, **kwargs)

        # potentially store references to inputs to remove later on
        self.remove_fold_inputs = (
            [inp["columns"] for inp in inputs["required_producers"].values()]
            if self.require_producers and self.remove_folds
            else []
        )

    def call_func(self, events: ak.Array, **kwargs) -> ak.Array:
        event_fold = events.event % _run3_dnn.n_folds

        # invoke the evaluations of all folds when not requiring them as dedicated producers
        if not self.require_folds:
            for dnn_cls in self.dnn_classes.values():
                events = self[dnn_cls](events, **kwargs)

        for out in ["hh_ggf", "tt", "dy", "hh_vbf"]:
            # fill score from columns at positions with different folds
            score = EMPTY_FLOAT * np.ones(len(events), dtype=np.float32)
            for f in range(_vbf_dnn_xvalid.n_folds):
                score[event_fold == f] = events[f"vbf_dnn_fold{f}_{out}"][event_fold == f]

            # assign to new column
            events = set_ak_column_f32(events, f"vbf_dnn_moe_{out}", score)

        return events

    def teardown_func(self, task: law.Task, **kwargs) -> None:
        super().teardown_func(task, **kwargs)

        # remove outputs of required producers
        for inp in self.remove_fold_inputs:
            inp.remove(silent=True)
