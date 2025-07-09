# coding: utf-8

"""
Producer to prepare DNN inputs for Run3 classifier.
"""

from __future__ import annotations
from collections.abc import Collection
import functools

import law

from columnflow.production import Producer, producer
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import (
    set_ak_column, attach_behavior, flat_np_view, EMPTY_FLOAT, default_coffea_collections,
    EMPTY_INT,
)
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.types import Any

np = maybe_import("numpy")
ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)

# helper function
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses=({
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
    }),
    produces=({
        "lepton{1,2}.*",
        "bjet{1,2}.*",
        "fatjet.*",
        "htt.*",
        "hbb.*",
        "htthbb.*",
        "year_flag",
        "pair_type",
        "has_jet_pair",
        "has_fatjet",
        "decay_mode{1,2}",
        "rotated_PuppiMET.*",

    }),
    # variable to configure whether to rotate continuous features against lepton system
    do_rotation=True,
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_dev.sh"),
)
def res_net_preprocessing(self, events: ak.Array, **kwargs) -> ak.Array:
    # ensure coffea behavior
    events = self[attach_coffea_behavior](
        events,
        collections={"HHBJet": default_coffea_collections["Jet"]},
        **kwargs,
    )

    # sanity masks for later usage
    # has_jet_pair = ak.num(events.HHBJet) >= 2
    # has_fatjet = ak.num(events.FatJet) >= 1

    # define the pair type (KLUBs channel id)
    pair_type = np.zeros(len(events), dtype=np.int32)
    for channel_id, pair_type_id in self.channel_id_to_pair_type.items():
        pair_type[events.channel_id == channel_id] = pair_type_id

    # first extract Leptons
    leptons: ak.Array = attach_behavior(
        ak.concatenate((events.Electron, events.Muon, events.Tau), axis=1),
        type_name="Tau",
    )

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

    # make sure to actually have two leptons
    has_lepton_pair = ak.num(leptons, axis=1) >= 2
    leptons = ak.mask(leptons, has_lepton_pair)
    lep1, lep2 = leptons[:, 0], leptons[:, 1]

    # whether the events is resolvede, boosted or neither
    has_jet_pair = ak.num(events.HHBJet) >= 2
    has_fatjet = ak.num(events.FatJet) >= 1

    # before preparing the network inputs, define a mask of events which have caregorical features
    # that are actually covered by the networks embedding layers; other events cannot be evaluated!
    event_mask = (
        np.isin(pair_type, self.embedding_expected_inputs["pair_type"]) &
        np.isin(dm1, self.embedding_expected_inputs["decay_mode1"]) &
        np.isin(dm2, self.embedding_expected_inputs["decay_mode2"]) &
        np.isin(lep1.charge, self.embedding_expected_inputs["lepton1.charge"]) &
        np.isin(lep2.charge, self.embedding_expected_inputs["lepton2.charge"]) &
        # (has_jet_pair | has_fatjet) &
        (self.year_flag in self.embedding_expected_inputs["year_flag"])
    )
    # if ak.any(~np.isfinite(event_mask) | ~event_mask):
    #     from IPython import embed
    #     embed(header="found NaN in event_mask")
    pair_type = ak.mask(pair_type, event_mask)
    leptons = ak.mask(leptons, event_mask)
    lep1, lep2 = leptons[:, 0], leptons[:, 1]
    tautau_mask = ak.mask(tautau_mask, event_mask)
    dm1, dm2 = ak.mask(dm1, event_mask), ak.mask(dm2, event_mask)
    has_jet_pair, has_fatjet = ak.mask(has_jet_pair, event_mask), ak.mask(has_fatjet, event_mask)

    # calculate phi of lepton system
    phi_lep = np.arctan2(lep1.py + lep2.py, lep1.px + lep2.px)

    def set_ak_column_nonull(events, name, values, placeholder, event_mask):
        try:
            arr = ak.ones_like(events.event, dtype=(np.int32 if placeholder == EMPTY_INT else np.float32)) * placeholder
            arr = ak.fill_none(arr, placeholder, axis=0)
            event_mask = ak.fill_none(event_mask, False, axis=0)
            flat_np_view(arr)[event_mask] = values[event_mask]
        except Exception as e:
            from IPython import embed
            embed(header=f"error when writing column {name}")
            raise e
        return set_ak_column(events, name, arr)

    def save_rotated_momentum(
        events: ak.Array,
        array: ak.Array,
        event_mask: ak.Array,
        target_field: str,
        additional_targets: Collection[str] | None = None,
        placeholder: float | int = EMPTY_FLOAT,
        rotation_angle=0.0,
    ) -> ak.Array:

        px, py = array.px, array.py
        if self.do_rotation:
            px, py = rotate_to_phi(phi_lep, array.px, array.py)

        # save px and py
        events = set_ak_column_nonull(events, f"{target_field}.px", px, placeholder=placeholder, event_mask=event_mask)
        events = set_ak_column_nonull(events, f"{target_field}.py", py, placeholder=placeholder, event_mask=event_mask)

        routes: set[str] = set()
        if additional_targets is not None:
            routes.update(additional_targets)
        for field in routes:
            events = set_ak_column_nonull(
                events,
                f"{target_field}.{field}",
                getattr(array, field),
                placeholder,
                event_mask=event_mask,
            )
        return events

    default_4momenta_cols: set[str] = set(("pz", "energy", "mass"))
    events = save_rotated_momentum(
        events,
        lep1,
        target_field="lepton1",
        additional_targets=(default_4momenta_cols | {"charge"}),
        event_mask=(event_mask & has_lepton_pair),
        rotation_angle=0.0,
    )

    events = save_rotated_momentum(
        events,
        lep2,
        target_field="lepton2",
        additional_targets=(default_4momenta_cols | {"charge"}),
        event_mask=(event_mask & has_lepton_pair),
        rotation_angle=0.0,
    )

    # there might be less than two jets or no fatjet, so pad them
    # bjets = ak.pad_none(_events.HHBJet, 2, axis=1)
    # fatjet = ak.pad_none(_events.FatJet, 1, axis=1)[:, 0]

    jet_columns = {
        "btagDeepFlavB", "hhbtag", "btagDeepFlavCvB", "btagDeepFlavCvL", "btagPNetB", "btagPNetCvB", "btagPNetCvL",
    } | default_4momenta_cols

    bjet_events = ak.mask(events, has_jet_pair)
    events = save_rotated_momentum(
        events,
        bjet_events.HHBJet[:, 0],
        target_field="bjet1",
        additional_targets=jet_columns,
        event_mask=(event_mask & has_jet_pair),
        rotation_angle=0.0,
    )
    events = save_rotated_momentum(
        events,
        bjet_events.HHBJet[:, 1],
        target_field="bjet2",
        additional_targets=jet_columns,
        event_mask=(event_mask & has_jet_pair),
        rotation_angle=0.0,
    )
    fatjet_events = ak.mask(events, has_fatjet)
    # fatjet variables
    events = save_rotated_momentum(
        events,
        fatjet_events.FatJet[:, 0],
        target_field="fatjet",
        additional_targets=default_4momenta_cols,
        event_mask=(event_mask & has_fatjet),
        rotation_angle=0.0,
    )

    # combine daus
    events = save_rotated_momentum(
        events,
        leptons[:, :2].sum(axis=-1),
        target_field="htt",
        additional_targets=default_4momenta_cols,
        event_mask=(event_mask & has_lepton_pair),
        rotation_angle=0.0,
    )
    # combine bjets
    events = save_rotated_momentum(
        events,
        events.HHBJet[:, :2].sum(axis=-1),
        target_field="hbb",
        additional_targets=default_4momenta_cols,
        event_mask=(event_mask & has_jet_pair),
        rotation_angle=0.0,
    )

    # htt + hbb
    events = save_rotated_momentum(
        events,
        leptons[:, :2].sum(axis=-1) + events.HHBJet[:, :2].sum(axis=-1),
        target_field="htthbb",
        additional_targets=default_4momenta_cols,
        event_mask=(event_mask & has_lepton_pair & has_jet_pair),
        rotation_angle=0.0,
    )
    # MET variables
    met_name = self.config_inst.x.met_name
    _met = events[met_name]

    events = save_rotated_momentum(
        events,
        _met,
        target_field=f"rotated_{met_name}",
        additional_targets={"covXX", "covXY", "covYY"},
        event_mask=event_mask,
        rotation_angle=0.0,
    )

    events = save_rotated_momentum(
        events,
        _met,
        target_field=f"{met_name}",
        additional_targets={"covXX", "covXY", "covYY"},
        event_mask=event_mask,
        rotation_angle=0.0,
    )

    events = set_ak_column_nonull(
        events,
        "year_flag",
        ak.ones_like(events.event) * self.year_flag,
        EMPTY_INT,
        event_mask=event_mask,
    )
    events = set_ak_column_nonull(
        events,
        "pair_type",
        pair_type,
        EMPTY_INT,
        event_mask=event_mask,
    )
    events = set_ak_column_nonull(
        events,
        "has_jet_pair",
        has_jet_pair,
        False,
        event_mask=event_mask,
    )
    events = set_ak_column_nonull(
        events,
        "has_fatjet",
        has_fatjet,
        False,
        event_mask=event_mask,
    )

    events = set_ak_column_nonull(
        events,
        "decay_mode1",
        dm1,
        EMPTY_INT,
        event_mask=(event_mask & (leptau_mask | tautau_mask)),
    )
    events = set_ak_column_nonull(
        events,
        "decay_mode2",
        dm2,
        EMPTY_INT,
        event_mask=(event_mask & (leptau_mask | tautau_mask)),
    )

    return events


@res_net_preprocessing.setup
def res_net_preprocessing_setup(
    self: Producer,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    **kwargs,
) -> None:
    from hbt.ml.torch_utils.utils import embedding_expected_inputs
    self.embedding_expected_inputs = embedding_expected_inputs
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
        (2022, ""): 4,
        (2022, "EE"): 5,
        (2023, ""): 6,
        (2023, "BPix"): 7,
    }[(self.config_inst.campaign.x.year, self.config_inst.campaign.x.postfix)]


@res_net_preprocessing.init
def res_net_preprocessing_init(self: Producer) -> None:
    self.uses.add(f"{self.config_inst.x.met_name}.{{pt,phi,covXX,covXY,covYY}}")
    self.produces.add(f"{self.config_inst.x.met_name}.{{px,py,covXX,covXY,covYY}}")


res_net_preprocessing_no_rotation = res_net_preprocessing.derive(
    "res_net_preprocessing_no_rotation", cls_dict={
        "do_rotation": False,
    },
)


def rotate_to_phi(ref_phi: ak.Array, px: ak.Array, py: ak.Array) -> tuple[ak.Array, ak.Array]:
    """
    Rotates a momentum vector extracted from *events* in the transverse plane to a reference phi
    angle *ref_phi*. Returns the rotated px and py components in a 2-tuple.
    """
    new_phi = np.arctan2(py, px) - ref_phi
    pt = (px**2 + py**2)**0.5
    return pt * np.cos(new_phi), pt * np.sin(new_phi)
