# coding: utf-8

"""
Selection methods.
"""

from operator import and_
from functools import reduce
from collections import defaultdict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.selection.cms.json_filter import json_filter
from columnflow.selection.cms.met_filters import met_filters
from columnflow.production.processes import process_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.pileup import pu_weight
from columnflow.production.cms.pdf import pdf_weights
from columnflow.production.cms.scale import murmuf_weights
from columnflow.production.cms.btag import btag_weights
from columnflow.production.util import attach_coffea_behavior
from columnflow.util import maybe_import, dev_sandbox

from hbt.selection.trigger import trigger_selection
from hbt.selection.lepton import lepton_selection
from hbt.selection.jet import jet_selection
from hbt.production.features import cutflow_features
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

    events, results = self[default](events, stats, **kwargs)

    # remove decay modes not used within model
    decay_modes_used_by_model = [-1, 0, 1, 10, 11]

    # mask containing all events with values NOT seen by network
    events_mask_decay_modes_seen_by_network = ak.any(
        [ak.any(events.Tau.decayMode == decay_mode, axis=1)
            for decay_mode in decay_modes_used_by_model],
        axis=0,
    )

    from IPython import embed; embed(); globals().update(locals())

    decay_mode_result = SelectionResult(
        steps={
            "decay_mode": events_mask_decay_modes_seen_by_network,
        },
    )

    results += decay_mode_result

    return events, results
