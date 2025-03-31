# coding: utf-8

"""
Custom event and object reducers.
"""

from columnflow.reduction import Reducer, reducer
from columnflow.reduction.default import cf_default
# from columnflow.production.cms.dy import gen_dilepton
from columnflow.util import maybe_import

# from hbt.util import IF_DATASET_IS_DY

ak = maybe_import("awkward")


@reducer(
    # uses={cf_default, IF_DATASET_IS_DY(gen_dilepton)},
    # produces={cf_default, IF_DATASET_IS_DY(gen_dilepton)},
    uses={cf_default},
    produces={cf_default},
)
def default(self: Reducer, events: ak.Array, selection: ak.Array, **kwargs) -> ak.Array:
    # run cf's default reduction which handles event selection and collection creation
    events = self[cf_default](events, selection, **kwargs)

    # add generator particles, depending on the dataset
    # if self.has_dep(gen_dilepton):
    #     events = self[gen_dilepton](events, **kwargs)

    return events
