# coding: utf-8

"""
Custom event and object reducers.
"""
from functools import reduce
from operator import or_
from columnflow.reduction import Reducer, reducer
from columnflow.reduction.default import cf_default
from columnflow.production.cms.dy import gen_dilepton, recoil_corrected_met
from columnflow.production.cms.top_pt_weight import gen_parton_top as cf_gen_parton_top
from columnflow.util import maybe_import
from columnflow.config_util import get_events_from_categories

from hbt.util import IF_DATASET_HAS_TOP, IF_DATASET_IS_DY

ak = maybe_import("awkward")


gen_parton_top = cf_gen_parton_top.derive("gen_parton_top", cls_dict={"require_dataset_tag": None})


@reducer(
    uses={
        cf_default,
        IF_DATASET_HAS_TOP(gen_parton_top),
        IF_DATASET_IS_DY(gen_dilepton, recoil_corrected_met),
    },
    produces={
        cf_default,
        IF_DATASET_HAS_TOP(gen_parton_top),
        IF_DATASET_IS_DY(gen_dilepton, recoil_corrected_met),
    },
)
def default(self: Reducer, events: ak.Array, selection: ak.Array, **kwargs) -> ak.Array:
    # run cf's default reduction which handles event selection and collection creation
    events = self[cf_default](events, selection, **kwargs)

    # add generator particles, depending on the dataset
    if self.has_dep(gen_parton_top):
        events = self[gen_parton_top](events, **kwargs)
    if self.has_dep(gen_dilepton):
        events = self[gen_dilepton](events, **kwargs)

    # add recoil corrected met
    if self.has_dep(recoil_corrected_met):
        events = self[recoil_corrected_met](events, **kwargs)

    return events


@reducer(
    uses={default, "channel_id"},
    produces={default},
)
def signal_region_reducer(self, events, selection, **kwargs):
    events = self[default](events, selection, **kwargs)

    # only select signal-like channels
    channel_id = events.channel_id

    signal_ids = [
        self.config_inst.channels.n.mutau.id,
        self.config_inst.channels.n.etau.id,
        self.config_inst.channels.n.tautau.id,
    ]

    event_mask = reduce(or_, [channel_id == x for x in signal_ids])
    return events[event_mask]


@reducer(
    uses={signal_region_reducer, "channel_id"},
    produces={signal_region_reducer},
)
def signal_region_os_reducer(self, events, selection, **kwargs):
    events = self[signal_region_reducer](events, selection, **kwargs)

    events = get_events_from_categories(events=events, categories=["os"], config_inst=self.config_inst)

    return events
