# coding: utf-8

"""
Trigger selection methods.
"""

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, optional_column as opt

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        "run",
        "TrigObj.{id,pt,eta,phi,filterBits}",
    },
    produces={
        "fired_trigger_ids",
    },
    exposed=True,
)
def trigger_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    HLT trigger path selection.
    """
    any_fired = False
    trigger_data = []
    trigger_ids = []

    # index of TrigObj's to repeatedly convert masks to indices
    index = ak.local_index(events.TrigObj)

    for trigger in self.config_inst.x.triggers:
        # skip the trigger if it does not apply to the dataset
        if not trigger.applies_to_dataset(self.dataset_inst):
            continue

        # get bare decisions
        fired = events.HLT[trigger.hlt_field] == 1
        if trigger.run_range:
            fired = fired & (
                ((trigger.run_range[0] is None) | (trigger.run_range[0] <= events.run)) &
                ((trigger.run_range[1] is None) | (trigger.run_range[1] >= events.run))
            )
        any_fired = any_fired | fired

        # get trigger objects for fired events per leg
        leg_masks = {}
        all_legs_match = True
        for key, leg in trigger.legs.items():
            # start with a True mask
            leg_mask = abs(events.TrigObj.id) >= 0
            # pdg id selection
            if leg.pdg_id is not None:
                leg_mask = leg_mask & (abs(events.TrigObj.id) == leg.pdg_id)
            # pt cut
            if leg.min_pt is not None:
                leg_mask = leg_mask & (events.TrigObj.pt >= leg.min_pt)
            # trigger bits match
            if leg.trigger_bits is not None:
                # OR across bits themselves, AND between all decision in the list
                for bits in leg.trigger_bits:
                    leg_mask = leg_mask & ((events.TrigObj.filterBits & bits) > 0)
            leg_masks[key] = index[leg_mask]
            # at least one object must match this leg
            all_legs_match = all_legs_match & ak.any(leg_mask, axis=1)

        # final trigger decision
        fired_and_all_legs_match = fired & all_legs_match

        # store all intermediate results for subsequent selectors
        trigger_data.append((trigger, fired_and_all_legs_match, leg_masks))

        # store the trigger id
        ids = ak.where(fired_and_all_legs_match, np.float32(trigger.id), np.float32(np.nan))
        trigger_ids.append(ak.singletons(ak.nan_to_none(ids)))

    # store the fired trigger ids
    trigger_ids = ak.concatenate(trigger_ids, axis=1)
    events = set_ak_column(events, "fired_trigger_ids", trigger_ids, value_type=np.int32)

    return events, SelectionResult(
        steps={
            "trigger": any_fired,
        },
        aux={
            "trigger_data": trigger_data,
        },
    )


@trigger_selection.init
def trigger_selection_init(self: Selector, **kwargs) -> None:
    # full used columns
    self.uses |= {
        opt(trigger.name)
        for trigger in self.config_inst.x.triggers
        if trigger.applies_to_dataset(self.dataset_inst)
    }
