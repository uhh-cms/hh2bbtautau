# coding: utf-8

"""
Selection methods.
"""

from operator import and_
from functools import reduce
from collections import defaultdict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.util import maybe_import, dev_sandbox

from hbt.selection.default import default

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        default, "Tau.decayMode", increment_stats,
    },
    produces={
        default,
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar_tf.sh"),
    exposed=True,
)
def tautauNN_default(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    # expand default selector
    events, results = self[default](events, stats, **kwargs)

    # ensure that only decay modes seen by tautauNN model are used
    # this is necessary since embedding layers are used
    expected_decay_mode = [-1, 0, 1, 10, 11]

    # filter out unseen decay modes
    decay_mode_filters = [ak.any(events.Tau.decayMode == decay_mode, axis=1)
        for decay_mode in expected_decay_mode]

    # combine filters, to create an event filter
    decay_mode_event_filter = ak.any(decay_mode_filters, axis=0)

    decay_mode_result = SelectionResult(
        steps={
            "decay_mode": decay_mode_event_filter,
        },
    )

    results += decay_mode_result

    # combine decay_mode mask with masks from default selection
    event_sel = reduce(and_, results.steps.values())
    results.events = event_sel

    return events, results
