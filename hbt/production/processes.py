# coding: utf-8

"""
Process ID producer relevant for the stitching of the DY samples.
"""

import re

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from hbt.util import IF_DATASET_IS_DY

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={IF_DATASET_IS_DY("LHE.NpNLO", "LHE.Vpt", "LHE.*")},
    produces={"process_ids"},
)
def dy_process_ids(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Assigns each dy event a single process id, based on the number of jets and the di-lepton pt of the LHE record.
    This is used for the stitching of the DY samples.
    """
    # trivial case
    if len(self.dataset_inst.processes) != 1:
        raise NotImplementedError(
            f"dataset {self.dataset_inst.name} has {len(self.dataset_inst.processes)} processes "
            "assigned, which is not yet implemented",
        )
    process = self.dataset_inst.processes.get_first()
    jet_match = re.match(r"^.*(\d)j.*$", process.name)

    if process.is_leaf_process:
        process_id = process.id
        # store the column
        events = set_ak_column(events, "process_id", len(events) * [process_id], value_type=np.int32)
        return events

    # get the LHE Njets and Vpt
    njets = events.LHE.NpNLO
    pt = events.LHE.Vpt
    process_ids = np.zeros(len(events), dtype=np.int32)

    if jet_match:
        n = int(jet_match.groups()[0])
        for min_pt, max_pt in self.config_inst.x.dy_pt_stitching_ranges[n]:
            process_ids[(pt >= min_pt) & (pt < max_pt)] = (
                self.config_inst.get_process(f"dy_m50toinf_{n}j_pt{min_pt}to{max_pt}").id
            )

    else:
        process_ids[njets == 0] = self.config_inst.get_process("dy_m50toinf_0j").id
        process_ids[njets >= 3] = self.config_inst.get_process("dy_m50toinf_ge3j").id
        for n, pt_ranges in self.config_inst.x.dy_pt_stitching_ranges.items():
            for min_pt, max_pt in pt_ranges:
                process_ids[(njets == n) & (pt >= min_pt) & (pt < max_pt)] = (
                    self.config_inst.get_process(f"dy_m50toinf_{n}j_pt{min_pt}to{max_pt}").id
                )

    # store the column
    events = set_ak_column(events, "process_id", len(events) * [process_id], value_type=np.int32)
    return events
