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
    set_ak_column, attach_behavior, flat_np_view, EMPTY_FLOAT, default_coffea_collections,
)
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.types import Any

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


class _vbf_dnn_evaluation(Producer):
    """
    Base producer for the vbf dnn evaluation of the non-resonant run 3 analysis, whose models are considered external
    and thus part of producers rather than standalone ml model objects.
    The output scores are classifying if incoming events are VBF, GGF, Drell-Yan or ttbar.
    The network uses continous and categorical inputs. A list of all inputs in the
    correct order can be found in the Filip's eos space:
    https://cernbox.cern.ch/files/spaces/eos/user/f/fbilandz/meta.json
    """

    uses = {
        attach_coffea_behavior,
        # custom columns created upstream, probably by a selector
        "channel_id",
        # nano columns
        "event",
        "Tau.{eta,phi,pt,mass,charge,decayMode}",
        "Electron.{eta,phi,pt,mass,charge}",
        "Muon.{eta,phi,pt,mass,charge}",
        "HHBJet.{pt,eta,phi,mass,hhbtag,btagDeepFlav*,btagPNet*}",
        "VBFJet.{pt,eta,phi,mass,hhbtag,btagDeepFlav*,btagPNet*}",
        "Jet.{pt,eta,phi,mass,btagDeepFlav*,btagPNet*,assignment_bits}",
        "FatJet.{eta,phi,pt,mass}",
        # regressed neutrinos from reg_dnn_moe producer
        "reg_dnn_moe_nu{1,2}_{px,py,pz}",
    }

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

        # TODO: check whether needed here
        # update shifts dynamically
        self.shifts.add("minbias_xs_{up,down}")  # variations of minbias_xs used in met phi correction
        self.shifts.update({  # all calibrations that change jet and lepton momenta
            shift_inst.name
            for shift_inst in self.config_inst.shifts
            if shift_inst.has_tag({"jec", "jer", "tec", "eec", "eer"})
        })

    def requires_func(self, task: law.Task, reqs: dict, **kwargs) -> None:
        from columnflow.tasks.production import ProduceColumns
        reqs["reg_dnn_moe"] = {
            0: ProduceColumns.req_other_producer(task, producer="reg_dnn_moe"),
        }

        if "external_files" in reqs:
            return

        from columnflow.tasks.external import BundleExternalFiles
        reqs["external_files"] = BundleExternalFiles.req(task)

    def setup_func(
        self,
        task: law.Task,
        reqs: dict[str, DotDict[str, Any]],
        inputs: dict,
        reader_targets: law.util.InsertableDict,
        **kwargs,
    ) -> None:
        # add outputs of required producers to list of columnar files that are read in the producer loop
        reader_targets["reg_dnn_moe"] = inputs["reg_dnn_moe"][0]["columns"]
        # reader_targets.update({
        #     "reg_dnn_moe": inp["columns"]
        #     for inp in inputs["reg_dnn_moe"].items()
        # })

        from hbt.ml.tf_evaluator import TFEvaluator

        if not getattr(task, "taf_tf_evaluator", None):
            task.taf_tf_evaluator = TFEvaluator()
        self.evaluator = task.taf_tf_evaluator

        # TODO: check paths here
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
        # (names and values from training code that was aligned to CCLUB notation)
        self.embedding_expected_inputs = {
            "pair_type": [0, 1, 2],  # see mapping below
            "decay_mode1": [-999, 0, 1, 10, 11],  # -999 for e/mu
            "decay_mode2": [0, 1, 10, 11],
            "charge1": [-1, 1],
            "charge2": [-1, 1],
            "hasResolvedAK4": [0, 1],
            "hasBoostedAK8": [0, 1],
            "hasVBFAK4": [0, 1],
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

        # ensure coffea behavior
        events = self[attach_coffea_behavior](
            events,
            collections={"HHBJet": default_coffea_collections["Jet"], "VBFJet": default_coffea_collections["Jet"]},
            **kwargs,
        )

        # define the pair type (KLUBs channel id)
        pair_type = np.zeros(len(events), dtype=np.int32)
        for channel_id, pair_type_id in self.channel_id_to_pair_type.items():
            pair_type[events.channel_id == channel_id] = pair_type_id

        # TODO: add boosted taus
        # get visible tau decay products, consider them all as tau types
        vis_taus = attach_behavior(
            ak.concatenate((events.Electron, events.Muon, events.Tau), axis=1),
            type_name="Tau",
        )
        vis_tau1, vis_tau2 = vis_taus[:, 0], vis_taus[:, 1]

        # get decay mode of first lepton (e, mu or tau)
        tautau_mask = events.channel_id == self.config_inst.channels.n.tautau.id
        # -999 is the value used for electrons and muons
        dm1 = -999 * np.ones(len(events), dtype=np.int32)
        if ak.any(tautau_mask):
            dm1[tautau_mask] = events.Tau.decayMode[tautau_mask][:, 0]

        # get decay mode of second lepton (also a tau, but position depends on channel)
        leptau_mask = (
            (events.channel_id == self.config_inst.channels.n.etau.id) |
            (events.channel_id == self.config_inst.channels.n.mutau.id)
        )
        dm2 = -999 * np.ones(len(events), dtype=np.int32)
        if ak.any(leptau_mask):
            dm2[leptau_mask] = events.Tau.decayMode[leptau_mask][:, 0]
        if ak.any(tautau_mask):
            dm2[tautau_mask] = events.Tau.decayMode[tautau_mask][:, 1]

        # should not exist...
        # # the dnn treats dm 2 as 1, so we need to map it
        # dm1 = np.where(dm1 == 2, 1, dm1)
        # dm2 = np.where(dm2 == 2, 1, dm2)

        # whether the events is resolved, vbf, boosted or neither
        has_jet_pair = ak.num(events.HHBJet) >= 2
        has_fatjet = ak.num(events.FatJet) >= 1
        has_vbf_jets = ak.num(events.VBFJet) >= 2

        # before preparing the network inputs, define a mask of events which have categorical features
        # that are actually covered by the networks embedding layers; other events cannot be evaluated!
        event_mask = (
            np.isin(pair_type, self.embedding_expected_inputs["pair_type"]) &
            # TODO: check if dm still works with boosted Taus, but should since column exists
            np.isin(dm1, self.embedding_expected_inputs["decay_mode1"]) &
            np.isin(dm2, self.embedding_expected_inputs["decay_mode2"]) &  # removes ee, mumu and emu events
            np.isin(vis_tau1.charge, self.embedding_expected_inputs["charge1"]) &
            np.isin(vis_tau2.charge, self.embedding_expected_inputs["charge2"]) &
            (has_jet_pair | has_fatjet | has_vbf_jets)  # should always be the case, but fine...
        )

        # hook to update the event mask base on additional event info
        event_mask = self.update_event_mask(events, event_mask)

        # apply to all arrays needed until now
        _events = events[event_mask]
        pair_type = pair_type[event_mask]
        vis_tau1, vis_tau2 = vis_tau1[event_mask], vis_tau2[event_mask]
        tautau_mask = tautau_mask[event_mask]
        dm1, dm2 = dm1[event_mask], dm2[event_mask]
        has_jet_pair, has_fatjet, has_vbf_jets = has_jet_pair[event_mask], has_fatjet[event_mask], has_vbf_jets[event_mask]  # noqa: E501

        # prepare network inputs
        f = DotDict()

        # compute angle from visible mother particle of vis_tau1 and vis_tau2
        # used to rotate the kinematics of dau{1,2}, bjet{1,2}, vbfjet{1,2} and fatjets relative to it
        # used backwards to rotate neutrinos for htt calculation
        phi_lep = np.arctan2(vis_tau1.py + vis_tau2.py, vis_tau1.px + vis_tau2.px)

        # lepton 1
        f.vis_tau1_px, f.vis_tau1_py = rotate_to_phi(phi_lep, vis_tau1.px, vis_tau1.py)
        f.vis_tau1_pz, f.vis_tau1_e = vis_tau1.pz, vis_tau1.energy

        # lepton 2
        f.vis_tau2_px, f.vis_tau2_py = rotate_to_phi(phi_lep, vis_tau2.px, vis_tau2.py)
        f.vis_tau2_pz, f.vis_tau2_e = vis_tau2.pz, vis_tau2.energy

        # there might be less than two jets, vbfjets or no fatjet, so pad them
        bjets = ak.pad_none(_events.HHBJet, 2, axis=1)
        vbfjets = ak.pad_none(_events.VBFJet, 2, axis=1)
        fatjet = ak.pad_none(_events.FatJet, 1, axis=1)[:, 0]

        # bjet 1
        f.bjet1_px, f.bjet1_py = rotate_to_phi(phi_lep, bjets[:, 0].px, bjets[:, 0].py)
        f.bjet1_pz, f.bjet1_e = bjets[:, 0].pz, bjets[:, 0].energy
        f.bjet1_tag_b = bjets[:, 0]["btagPNetB"]
        f.bjet1_tag_cvsb = bjets[:, 0]["btagPNetCvB"]
        f.bjet1_tag_cvsl = bjets[:, 0]["btagPNetCvL"]
        f.bjet1_hhbtag = bjets[:, 0].hhbtag

        # bjet 2
        f.bjet2_px, f.bjet2_py = rotate_to_phi(phi_lep, bjets[:, 1].px, bjets[:, 1].py)
        f.bjet2_pz, f.bjet2_e = bjets[:, 1].pz, bjets[:, 1].energy
        f.bjet2_tag_b = bjets[:, 1]["btagPNetB"]
        f.bjet2_tag_cvsb = bjets[:, 1]["btagPNetCvB"]
        f.bjet2_tag_cvsl = bjets[:, 1]["btagPNetCvL"]
        f.bjet2_hhbtag = bjets[:, 1].hhbtag

        # fatjet variables
        f.fatjet_px, f.fatjet_py = rotate_to_phi(phi_lep, fatjet.px, fatjet.py)
        f.fatjet_pz, f.fatjet_e = fatjet.pz, fatjet.energy

        # vbf jet 1
        f.vbfjet1_px, f.vbfjet1_py = rotate_to_phi(phi_lep, vbfjets[:, 0].px, vbfjets[:, 0].py)
        f.vbfjet1_pz, f.vbfjet1_e = vbfjets[:, 0].pz, vbfjets[:, 0].energy
        f.vbfjet1_pnet_QvsG = vbfjets[:, 0]["btagPNetQvG"]

        # vbf jet 2
        f.vbfjet2_px, f.vbfjet2_py = rotate_to_phi(phi_lep, vbfjets[:, 1].px, vbfjets[:, 1].py)
        f.vbfjet2_pz, f.vbfjet2_e = vbfjets[:, 1].pz, vbfjets[:, 1].energy
        f.vbfjet2_pnet_QvsG = vbfjets[:, 1]["btagPNetQvG"]

        # neutrinos from regression
        f.nu1_px = _events.reg_dnn_moe_nu1_px
        f.nu1_py = _events.reg_dnn_moe_nu1_py
        f.nu1_pz = _events.reg_dnn_moe_nu1_pz
        f.nu2_px = _events.reg_dnn_moe_nu2_px
        f.nu2_py = _events.reg_dnn_moe_nu2_py
        f.nu2_pz = _events.reg_dnn_moe_nu2_pz

        # mask values as done during training of the network
        def mask_values(mask, value, *fields):
            if not ak.any(mask):
                print("No values to mask for fields:", fields)
                return
            for field in fields:
                arr = ak.fill_none(f[field], value, axis=0)
                flat_np_view(arr)[mask] = value
                f[field] = arr
        def mask_values_with_array(mask, value_array, *fields):
            if not ak.any(mask):
                print("No values to mask for fields:", fields)
                return
            for field in fields:
                if ak.sum(ak.is_none(f[field]) | mask) > ak.sum(mask):
                    raise RuntimeError("Still some None values in field: {} despite masking!".format(field))
                arr = ak.fill_none(f[field], 0., axis=0)
                flat_np_view(arr)[mask] = flat_np_view(value_array)[mask]
                f[field] = arr

        # TODO: check default values, should be rotated from -999 pt and -999 phi
        default_value_all = -999.0
        default_value_px_py = rotate_to_phi(phi_lep, default_value_all * np.cos(default_value_all), default_value_all * np.sin(default_value_all))  # noqa: E501
        mask_values_with_array(~has_jet_pair, default_value_px_py[0], "bjet1_px", "bjet2_px")
        mask_values_with_array(~has_jet_pair, default_value_px_py[1], "bjet1_py", "bjet2_py")
        mask_values(~has_jet_pair, default_value_all, "bjet1_pz", "bjet1_e", "bjet2_pz", "bjet2_e")
        mask_values(~has_jet_pair, default_value_all, "bjet1_tag_b", "bjet1_tag_cvsb", "bjet1_tag_cvsl", "bjet1_hhbtag")
        mask_values(~has_jet_pair, default_value_all, "bjet2_tag_b", "bjet2_tag_cvsb", "bjet2_tag_cvsl", "bjet2_hhbtag")
        mask_values_with_array(~has_fatjet, default_value_px_py[0], "fatjet_px")
        mask_values_with_array(~has_fatjet, default_value_px_py[1], "fatjet_py")
        mask_values(~has_fatjet, default_value_all, "fatjet_pz", "fatjet_e")
        mask_values_with_array(~has_vbf_jets, default_value_px_py[0], "vbfjet1_px", "vbfjet2_px")
        mask_values_with_array(~has_vbf_jets, default_value_px_py[1], "vbfjet1_py", "vbfjet2_py")
        mask_values(~has_vbf_jets, default_value_all, "vbfjet1_pz", "vbfjet1_e", "vbfjet2_pz", "vbfjet2_e", "vbfjet1_pnet_QvsG", "vbfjet2_pnet_QvsG")  # noqa: E501

        # define neutrino energy
        nu1_e = (f.nu1_px**2 + f.nu1_py**2 + f.nu1_pz**2)**0.5
        nu2_e = (f.nu2_px**2 + f.nu2_py**2 + f.nu2_pz**2)**0.5

        # combine regressed daus
        f.htt_regr_e = f.vis_tau1_e + f.vis_tau2_e + nu1_e + nu2_e
        f.htt_regr_px = f.vis_tau1_px + f.vis_tau2_px + f.nu1_px + f.nu2_px
        f.htt_regr_py = f.vis_tau1_py + f.vis_tau2_py + f.nu1_py + f.nu2_py
        f.htt_regr_pz = f.vis_tau1_pz + f.vis_tau2_pz + f.nu1_pz + f.nu2_pz

        # combine bjets
        f.hbb_e = f.bjet1_e + f.bjet2_e
        f.hbb_px = f.bjet1_px + f.bjet2_px
        f.hbb_py = f.bjet1_py + f.bjet2_py
        f.hbb_pz = f.bjet1_pz + f.bjet2_pz
        # TODO: modify default value -> -999 or the sum of the rotated -999s bjets?
        mask_values(~has_jet_pair, -999.0, "hbb_e", "hbb_px", "hbb_py", "hbb_pz")

        # htt + hbb
        f.htthbb_regr_e = f.htt_regr_e + f.hbb_e
        f.htthbb_regr_px = f.htt_regr_px + f.hbb_px
        f.htthbb_regr_py = f.htt_regr_py + f.hbb_py
        f.htthbb_regr_pz = f.htt_regr_pz + f.hbb_pz
        # TODO: modify default value
        mask_values(~has_jet_pair, -999.0, "htthbb_regr_e", "htthbb_regr_px", "htthbb_regr_py", "htthbb_regr_pz")

        # htt + fatjet
        f.httfatjet_regr_e = f.htt_regr_e + f.fatjet_e
        f.httfatjet_regr_px = f.htt_regr_px + f.fatjet_px
        f.httfatjet_regr_py = f.htt_regr_py + f.fatjet_py
        f.httfatjet_regr_pz = f.htt_regr_pz + f.fatjet_pz
        # TODO: modify default value
        mask_values(~has_fatjet, -999.0, "httfatjet_regr_e", "httfatjet_regr_px", "httfatjet_regr_py", "httfatjet_regr_pz")  # noqa: E501

        # vbf jets system variables
        f.VBFjj_mass = ((f.vbfjet1_e + f.vbfjet2_e)**2 -
                        (f.vbfjet1_px + f.vbfjet2_px)**2 -
                        (f.vbfjet1_py + f.vbfjet2_py)**2 -
                        (f.vbfjet1_pz + f.vbfjet2_pz)**2)**0.5

        vbfjet_1_phi = np.arctan2(f.vbfjet1_py, f.vbfjet1_px)
        vbfjet_2_phi = np.arctan2(f.vbfjet2_py, f.vbfjet2_px)
        # valid for pz > 0
        vbfjet1_eta = -np.log(np.tan(0.5 * np.arctan2(
            (f.vbfjet1_px**2 + f.vbfjet1_py**2)**0.5,
            (f.vbfjet1_px**2 + f.vbfjet1_py**2 + f.vbfjet1_pz**2)**0.5,
        )))
        vbfjet2_eta = -np.log(np.tan(0.5 * np.arctan2(
            (f.vbfjet2_px**2 + f.vbfjet2_py**2)**0.5,
            (f.vbfjet2_px**2 + f.vbfjet2_py**2 + f.vbfjet2_pz**2)**0.5,
        )))
        f.VBFdeltaR = ((vbfjet1_eta - vbfjet2_eta)**2 + (vbfjet_1_phi - vbfjet_2_phi)**2)**0.5
        f.etaprod_vbfjvbfj = vbfjet1_eta * vbfjet2_eta
        # default values
        # TODO: check default values: -999 or using the values needed for calculating?
        mask_values(~has_vbf_jets, -999.0, "VBFjj_mass", "VBFdeltaR", "etaprod_vbfjvbfj")

        # bb system variables
        bjet1_eta = -np.log(np.tan(0.5 * np.arctan2(
            (f.bjet1_px**2 + f.bjet1_py**2)**0.5,
            (f.bjet1_px**2 + f.bjet1_py**2 + f.bjet1_pz**2)**0.5,
        )))
        bjet2_eta = -np.log(np.tan(0.5 * np.arctan2(
            (f.bjet2_px**2 + f.bjet2_py**2)**0.5,
            (f.bjet2_px**2 + f.bjet2_py**2 + f.bjet2_pz**2)**0.5,
        )))
        f.etaprod_bb = bjet1_eta * bjet2_eta
        # default values
        # TODO: check default values: -999 or using the values needed for calculating?
        mask_values(~has_jet_pair, -999.0, "etaprod_bb")

        # M_chi
        # M_chi = HH_mass - (Hbb_mass - 125.0) - (Htt_mass - 125.0);
        # definition from  https://gitlab.cern.ch/cclubbtautau/AnalysisCore/-/blob/cclub_cmssw15010/src/HHRun3DNNInterface.cc?ref_type=heads#L517  # noqa: E501
        # TODO: add fatjet case for Hbb_mass for boosted events
        f.M_chi = (
            (f.htthbb_regr_e**2 - f.htthbb_regr_px**2 - f.htthbb_regr_py**2 - f.htthbb_regr_pz**2)**0.5 -
            ((f.hbb_e**2 - f.hbb_px**2 - f.hbb_py**2 - f.hbb_pz**2)**0.5 - 125.0) -
            ((f.htt_regr_e**2 - f.htt_regr_px**2 - f.htt_regr_py**2 - f.htt_regr_pz**2)**0.5 - 125.0)
        )
        # default values
        mask_values(~has_jet_pair, -999.0, "M_chi")
        # once fatjet in there:
        # mask_values(~(has_jet_pair | has_fatjet), -999.0, "M_chi")

        # fox wolfram moments
        # defined in https://gitlab.cern.ch/cclubbtautau/AnalysisCore/-/blob/cclub_cmssw15010/src/HHUtils.cc?ref_type=heads#L742-811  # noqa: E501
        # TODO: change hhbjets to fatjet for boosted events

        mask_hhbjets_vbfjets = (_events.Jet.assignment_bits == 0)
        # TODO: these jets should be cleaned from the fatjet witch deltaR < 0.8
        central_jets = _events.Jet[mask_hhbjets_vbfjets]
        # all central jets + hhbjets + vbfjets
        vbfcjets = ak.concatenate((_events.HHBJet, _events.VBFJet, central_jets), axis=1)
        sum_p_squared = (
            ak.sum(vbfcjets.energy, axis=1)**2 - ak.sum(vbfcjets.px, axis=1)**2 -
            ak.sum(vbfcjets.py, axis=1)**2 - ak.sum(vbfcjets.pz, axis=1)**2
        )
        sum_pt = ak.sum(vbfcjets.pt, axis=1)
        ijet, jjet = ak.unzip(ak.combinations(vbfcjets, 2, axis=1))
        omega_ij = (np.cos(ijet.theta) * np.cos(jjet.theta) +
                    np.sin(ijet.theta) * np.sin(jjet.theta) * np.cos(ijet.phi - jjet.phi))
        legendre_0 = np.polynomial.legendre.Legendre([1, 0, 0])(omega_ij)
        legendre_2 = np.polynomial.legendre.Legendre([0, 0, 1])(omega_ij)
        weight_s = (
            (ijet.px**2 + ijet.py**2 + ijet.pz**2)**0.5 *
            (jjet.px**2 + jjet.py**2 + jjet.pz**2)**0.5
        ) / (sum_p_squared)
        weight_T = (ijet.pt * jjet.pt) / (sum_pt**2)
        f.fwMoment_s_0 = ak.sum(weight_s * legendre_0, axis=1)
        f.fwMoment_T_0 = ak.sum(weight_T * legendre_0, axis=1)
        f.fwMoment_1_0 = ak.sum(legendre_0, axis=1)  # sum over 1s...
        f.fwMoment_s_2 = ak.sum(weight_s * legendre_2, axis=1)
        # TODO: add terms for j = fatjet

        # default values
        mask_values(
            ~(has_jet_pair & has_vbf_jets), -999.0,
            "fwMoment_s_0", "fwMoment_T_0", "fwMoment_1_0", "fwMoment_s_2",
        )
        # TODO: mask fatjet events too once included
        # mask_values(
        #     ~((has_jet_pair | has_fatjet) & has_vbf_jets), -999.0,
        #     "fwMoment_s_0", "fwMoment_T_0", "fwMoment_1_0", "fwMoment_s_2",
        # )

        # assign categorical inputs via names too
        f.pair_type = pair_type
        f.dm1 = dm1
        f.dm2 = dm2
        f.vis_tau1_charge = vis_tau1.charge
        f.vis_tau2_charge = vis_tau2.charge
        f.has_jet_pair = has_jet_pair
        f.has_fatjet = has_fatjet
        f.has_vbf_jets = has_vbf_jets

        # build continous inputs
        # (order exactly as documented in link above)
        continous_inputs = [
            np.asarray(t[..., None], dtype=np.float32) for t in [
                f.vis_tau1_px, f.vis_tau1_py, f.vis_tau1_pz, f.vis_tau1_e,
                f.vis_tau2_px, f.vis_tau2_py, f.vis_tau2_pz, f.vis_tau2_e,
                f.bjet1_px, f.bjet1_py, f.bjet1_pz, f.bjet1_e, f.bjet1_tag_b, f.bjet1_tag_cvsb, f.bjet1_tag_cvsl,
                f.bjet1_hhbtag,
                f.bjet2_px, f.bjet2_py, f.bjet2_pz, f.bjet2_e, f.bjet2_tag_b, f.bjet2_tag_cvsb, f.bjet2_tag_cvsl,
                f.bjet2_hhbtag,
                f.nu1_px, f.nu1_py, f.nu1_pz,
                f.nu2_px, f.nu2_py, f.nu2_pz,
                f.fatjet_px, f.fatjet_py, f.fatjet_pz, f.fatjet_e,
                f.htt_regr_px, f.htt_regr_py, f.htt_regr_pz, f.htt_regr_e,
                f.hbb_px, f.hbb_py, f.hbb_pz, f.hbb_e,
                f.httfatjet_regr_px, f.httfatjet_regr_py, f.httfatjet_regr_pz, f.httfatjet_regr_e,
                f.vbfjet1_px, f.vbfjet1_py, f.vbfjet1_pz, f.vbfjet1_e, f.vbfjet1_pnet_QvsG,
                f.vbfjet2_px, f.vbfjet2_py, f.vbfjet2_pz, f.vbfjet2_e, f.vbfjet2_pnet_QvsG,
                f.M_chi,
                f.VBFjj_mass, f.VBFdeltaR,
                f.etaprod_bb, f.etaprod_vbfjvbfj,
                f.fwMoment_s_0, f.fwMoment_T_0, f.fwMoment_1_0, f.fwMoment_s_2,
                f.htthbb_regr_e, f.htthbb_regr_px, f.htthbb_regr_py, f.htthbb_regr_pz,
            ]
            if t is not None
        ]
        # except Exception as e:
        #     print("Error while building continous inputs for VBF DNN:", e)
        #     from IPython import embed; embed(headers="in call_func of vbf_dnn_evaluation")  # noqa: E501
        #     raise e

        # build categorical inputs
        # (order exactly as documented in link above)
        categorical_inputs = [
            np.asarray(t[..., None], dtype=np.int32) for t in [
                f.pair_type,
                f.dm1, f.dm2,
                f.vis_tau1_charge, f.vis_tau2_charge,
                f.has_jet_pair, f.has_fatjet, f.has_vbf_jets,
            ] if t is not None
        ]

        # evaluate the model
        scores = self.evaluator(
            self.cls_name,
            inputs=[
                np.concatenate(continous_inputs, axis=1),
                np.concatenate(categorical_inputs, axis=1),
            ],
        )

        print(f"As VBF-classified events:{ak.sum(ak.argmax(scores, axis=1) == 0)} from {len(scores)} events")
        # TODO: check if still accurate
        if ak.sum(~np.isfinite(scores)) > 0:
            raise RuntimeError("NaN scores in VBF DNN evaluation!")
        # # in very rare cases (1 in 25k), the network output can be none, likely for numerical reasons,
        # # so issue a warning and set them to a default value
        # nan_mask = ~np.isfinite(scores)
        # if np.any(nan_mask):
        #     logger.warning(
        #         f"{nan_mask.sum() // scores.shape[1]} out of {scores.shape[0]} events have NaN scores; "
        #         f"setting them to {self.empty_value}",
        #     )
        #     scores[nan_mask] = self.empty_value

        # prepare output columns with the shape of the original events and assign values into them
        for i, column in enumerate(self.output_columns):
            values = self.empty_value * np.ones(len(events), dtype=np.float32)
            values[event_mask] = scores[:, i]
            events = set_ak_column_f32(events, column, values)

        if self.produce_features:
            # store input columns for sync
            cont_inputs_cols = [
                "vis_tau1_px", "vis_tau1_py", "vis_tau1_pz", "vis_tau1_e",
                "vis_tau2_px", "vis_tau2_py", "vis_tau2_pz", "vis_tau2_e",
                "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e", "bjet1_tag_b", "bjet1_tag_cvsb", "bjet1_tag_cvsl",
                "bjet1_hhbtag",
                "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e", "bjet2_tag_b", "bjet2_tag_cvsb", "bjet2_tag_cvsl",
                "bjet2_hhbtag",
                "nu1_px", "nu1_py", "nu1_pz",
                "nu2_px", "nu2_py", "nu2_pz",
                "fatjet_px", "fatjet_py", "fatjet_pz", "fatjet_e",
                "htt_regr_px", "htt_regr_py", "htt_regr_pz", "htt_regr_e",
                "hbb_px", "hbb_py", "hbb_pz", "hbb_e",
                "httfatjet_regr_px", "httfatjet_regr_py", "httfatjet_regr_pz", "httfatjet_regr_e",
                "vbfjet1_px", "vbfjet1_py", "vbfjet1_pz", "vbfjet1_e", "vbfjet1_pnet_QvsG",
                "vbfjet2_px", "vbfjet2_py", "vbfjet2_pz", "vbfjet2_e", "vbfjet2_pnet_QvsG",
                "M_chi",
                "VBFjj_mass", "VBFdeltaR",
                "etaprod_bb", "etaprod_vbfjvbfj",
                "fwMoment_s_0", "fwMoment_T_0", "fwMoment_1_0", "fwMoment_s_2",
                "htthbb_regr_e", "htthbb_regr_px", "htthbb_regr_py", "htthbb_regr_pz",
            ]
            cat_inputs_cols = [
                "pair_type", "dm1", "dm2", "vis_tau1_charge", "vis_tau2_charge", "has_jet_pair", "has_fatjet", "has_vbf_jets",  # noqa: E501
            ]
            for c in cont_inputs_cols + cat_inputs_cols:
                values = self.empty_value * np.ones(len(events), dtype=np.float32)
                values[event_mask] = ak.flatten(np.asarray(f[c][..., None], dtype=np.float32))
                events = set_ak_column_f32(events, f"{self.features_prefix}{self.cls_name}_{c}", values)

        return events

    def update_event_mask(self, events: ak.Array, event_mask: ak.Array) -> ak.Array:
        return event_mask


#
# producers for classification-only networks
# (combined network)
#

class _vbf_dnn(_vbf_dnn_evaluation):

    dir_name = "model_0"
    output_prefix = "vbf_dnn"

    def init_func(self, **kwargs) -> None:
        super().init_func(**kwargs)

        # output column names (in this order)
        self.output_columns = [
            f"{self.output_prefix}_{name}"
            for name in ["VBF", "ggF", "tt", "dy"]
        ]

        # update produced columns
        self.produces |= set(self.output_columns)


class _run3_vbf_dnn(_vbf_dnn):

    dir_name = None
    fold = None
    n_folds = 5

    @property
    def output_prefix(self) -> str:
        return self.cls_name

    def update_event_mask(self, events: ak.Array, event_mask: ak.Array) -> ak.Array:
        # when a fold is defined, select only events that match this fold
        # (all other events were potentially used for the training)
        if self.fold is not None:
            event_fold = events.event % self.n_folds
            event_mask = event_mask & (event_fold == self.fold)

        return event_mask


# derive evaluation producers for all folds
for fold in range(_run3_vbf_dnn.n_folds):
    # TODO: clarify dir_name, depending on unpacking, currently asuming no subdirectory inside the archive
    _run3_vbf_dnn.derive(f"run3_vbf_dnn_fold{fold}", cls_dict={
        "fold": fold,
        "external_name": f"run3_vbf_dnn_fold{fold}",
        "exposed": True,
    })

#
# producer for combining the results of all folds
#


class run3_vbf_dnn(Producer):

    # require ProduceColumns tasks per fold first when True, otherwise evaluate all folds in the same task
    require_producers = True

    # when require_producers is False, the sandbox must be set
    # sandbox = _run3_vbf_dnn.sandbox

    # when require_producers is True, decide whether to remove their outputs after successful combination
    remove_producers = True

    # used and produced columns
    # (used ones are updated dynamically in init_func)
    uses = {"event"}
    produces = {"run3_vbf_dnn_{VBF,ggF,tt,dy}"}
    exposed = True

    def init_func(self, **kwargs) -> None:
        # store dnn evaluation classes
        self.dnn_classes = {
            f: _run3_vbf_dnn.get_cls(f"run3_vbf_dnn_fold{f}")
            for f in range(_run3_vbf_dnn.n_folds)
        }

        # update used columns / dependencies
        for dnn_cls in self.dnn_classes.values():
            self.uses.add(f"{dnn_cls.cls_name}_{{VBF,ggF,tt,dy}}" if self.require_producers else dnn_cls)

    def requires_func(self, task: law.Task, reqs: dict, **kwargs) -> None:
        if not self.require_producers:
            return

        from columnflow.tasks.production import ProduceColumns
        reqs["run3_vbf_dnn_folds"] = {
            f: ProduceColumns.req_other_producer(task, producer=dnn_cls.cls_name)
            for f, dnn_cls in self.dnn_classes.items()
        }

    def setup_func(
        self,
        task: law.Task,
        reqs: dict,
        inputs: dict,
        reader_targets: law.util.InsertableDict,
        **kwargs,
    ) -> None:
        if not self.require_producers:
            return

        # add outputs of required producers to list of columnar files that are read in the producer loop
        reader_targets.update({
            f"run3_vbf_dnn_fold{f}": inp["columns"]
            for f, inp in inputs["run3_vbf_dnn_folds"].items()
        })

        # potentially store references to inputs to removal later on
        if self.remove_producers:
            self.remove_producer_inputs = [inp["columns"] for inp in inputs["run3_vbf_dnn_folds"].values()]

    def call_func(self, events: ak.Array, **kwargs) -> ak.Array:
        event_fold = events.event % _run3_vbf_dnn.n_folds

        # invoke the evaluations of all folds when not requiring them as dedicated producers
        if not self.require_producers:
            for dnn_cls in self.dnn_classes.values():
                events = self[dnn_cls](events, **kwargs)

        for out in ["VBF", "ggF", "tt", "dy"]:
            # fill score from columns at positions with different folds
            score = EMPTY_FLOAT * np.ones(len(events), dtype=np.float32)
            for f in range(_run3_vbf_dnn.n_folds):
                score[event_fold == f] = events[f"run3_vbf_dnn_fold{f}_{out}"][event_fold == f]

            # assign to new column
            events = set_ak_column_f32(events, f"run3_vbf_dnn_{out}", score)

        return events

    def teardown_func(self, task: law.Task, **kwargs) -> None:
        super().teardown_func(task, **kwargs)

        # remove outputs of required producers
        if self.require_producers and self.remove_producers:
            for inp in self.remove_producer_inputs:
                inp.remove()
