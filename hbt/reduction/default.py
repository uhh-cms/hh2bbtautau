# coding: utf-8

"""
Custom event and object reducers.
"""

from columnflow.reduction import Reducer, reducer
from columnflow.reduction.default import cf_default
from columnflow.production.cms.dy import gen_dilepton, recoil_corrected_met
<<<<<<< HEAD
# from columnflow.production.cms.top_pt_weight import gen_parton_top as cf_gen_parton_top
=======
from columnflow.production.cms.gen_particles import gen_higgs_lookup, gen_top_lookup, gen_dy_lookup

>>>>>>> master
from columnflow.util import maybe_import

from hbt.util import IF_DATASET_HAS_HIGGS, IF_DATASET_HAS_TOP, IF_DATASET_IS_DY

ak = maybe_import("awkward")


@reducer(
    uses={
        cf_default,
        IF_DATASET_HAS_HIGGS(gen_higgs_lookup),
        IF_DATASET_HAS_TOP(gen_top_lookup),
        IF_DATASET_IS_DY(gen_dy_lookup, gen_dilepton, recoil_corrected_met),
    },
    produces={
        cf_default,
        IF_DATASET_HAS_HIGGS(gen_higgs_lookup),
        IF_DATASET_HAS_TOP(gen_top_lookup),
        IF_DATASET_IS_DY(gen_dy_lookup, gen_dilepton, recoil_corrected_met),
    },
    check_produced_columns=False,
)
def default(self: Reducer, events: ak.Array, selection: ak.Array, **kwargs) -> ak.Array:
    # run cf's default reduction which handles event selection and collection creation
    events = self[cf_default](events, selection, **kwargs)

    # when there are no events left, return immediately
    # (ReduceEvents would anyway not write this chunk to disk and skips it during merging)
    if len(events) == 0:
        return events

    # add generator particles, depending on the dataset
    if self.has_dep(gen_higgs_lookup):
        events = self[gen_higgs_lookup](events, **kwargs)
    if self.has_dep(gen_top_lookup):
        events = self[gen_top_lookup](events, **kwargs)
    if self.has_dep(gen_dy_lookup):
        events = self[gen_dy_lookup](events, **kwargs)
    # TODO: is gen_dilepton redundant to what gen_dy_lookup provides?
    if self.has_dep(gen_dilepton):
        events = self[gen_dilepton](events, **kwargs)

    # add recoil corrected met
    if self.has_dep(recoil_corrected_met):
        events = self[recoil_corrected_met](events, **kwargs)

    return events
