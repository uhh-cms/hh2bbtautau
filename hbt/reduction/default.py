# coding: utf-8

"""
Custom event and object reducers.
"""

from columnflow.reduction import Reducer, reducer
from columnflow.reduction.default import cf_default
from columnflow.production.cms.dy import gen_dilepton, recoil_corrected_met
from columnflow.production.cms.top_pt_weight import gen_parton_top as cf_gen_parton_top
from columnflow.util import maybe_import

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
