# coding: utf-8

"""
Process ID producer relevant for the stitching of the DY samples.
"""
from __future__ import annotations
import functools

import law
import order

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, InsertableDict
from columnflow.columnar_util import set_ak_column, Route

from hbt.util import IF_DATASET_IS_DY
from columnflow.types import Callable

np = maybe_import("numpy")
ak = maybe_import("awkward")
sp = maybe_import("scipy")
maybe_import("scipy.sparse")


logger = law.logger.get_logger(__name__)

NJetsRange = tuple[int, int]
PtRange = tuple[float, float]

set_ak_column_i64 = functools.partial(set_ak_column, value_type=np.int64)

class stitched_process_ids(Producer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_func: Callable or None = None
        self.cross_check_func: Callable or None = self.cross_check_func
        self.stichting_observabes: list[str] or None = None
        self.cross_check_translation_dict: dict[str, str] or None = None

    def init_func(self, *args, **kwargs):
        self.uses |= {IF_DATASET_IS_DY(*self.stichting_observabes)}
        self.produces |= {IF_DATASET_IS_DY("process_id")}

    def call_func(self, events: ak.Array, **kwargs) -> ak.Array:
        """
        Assigns each event a single process id, based on the stitching observables of
        the LHE record. This is used for the stitching of the respective samples.
        """
        # as always, we assume that each dataset has exactly one process associated to it
        if len(self.dataset_inst.processes) != 1:
            raise NotImplementedError(
                f"dataset {self.dataset_inst.name} has {len(self.dataset_inst.processes)} processes "
                "assigned, which is not yet implemented",
            )
        process_inst = self.dataset_inst.processes.get_first()

        # # get the number of nlo jets and the di-lepton pt
        # njets = events.LHE.NpNLO
        # pt = events.LHE.Vpt
        # get stitching observables
        stitching_obs_values = [Route(obs).apply(events) for obs in self.stichting_observabes]

        if self.cross_check_translation_dict and callable(self.cross_check_func):
            self.cross_check_func(process_inst, stitching_obs_values)

        # lookup the id and check for invalid values
        process_ids = np.squeeze(np.asarray(self.id_table[self.key_func(*stitching_obs_values)].todense()))
        invalid_mask = process_ids == 0
        if ak.any(invalid_mask):
            raise ValueError(
                f"found {sum(invalid_mask)} events that could not be assigned to a process",
            )

        # store them
        events = set_ak_column_i64(events, "process_id", process_ids)

        return events


    def stitching_range_cross_check(
        self: Producer,
        process_inst: order.Process,
        stichting_values: list[ak.Array]
    ) -> None:
        # define lookup for stichting observable -> process auxiliary values to compare with
        # raise a warning if a datasets was already created for a specific "bin" (leaf process),
        # but actually does not fit
        for obs_name, obs_values in zip(self.stichting_observabes, stichting_values):
            aux_name = self.cross_check_translation_dict.get(obs_name, obs_name)
            aux_values = process_inst.x(aux_name, None)
            if aux_values is not None:
                outliers = (obs_values < aux_values[0]) | (obs_values >= aux_values[1])
                if ak.any(outliers):
                    logger.warning(
                        f"dataset {self.dataset_inst.name} is meant to contain {aux_name} "
                        f"values in the range [{aux_values[0]}, {aux_values[0]}), but found {ak.sum(outliers)} "
                        "events outside this range",
                    )

process_ids_dy = stitched_process_ids.derive(
    "process_ids_dy", cls_dict={
        "stichting_observabes": ["LHE.NpNLO", "LHE.Vpt"],
        "cross_check_translation_dict": {"LHE.NpNLO": "njets", "LHE.Vpt": "ptll"},
    },
)

# @producer(
#     uses={IF_DATASET_IS_DY("LHE.NpNLO", "LHE.Vpt")},
#     produces={IF_DATASET_IS_DY("process_id")},
# )
# def process_ids_dy(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
#     """
#     Assigns each dy event a single process id, based on the number of jets and the di-lepton pt of
#     the LHE record. This is used for the stitching of the DY samples.
#     """
#     # as always, we assume that each dataset has exactly one process associated to it
#     if len(self.dataset_inst.processes) != 1:
#         raise NotImplementedError(
#             f"dataset {self.dataset_inst.name} has {len(self.dataset_inst.processes)} processes "
#             "assigned, which is not yet implemented",
#         )
#     process_inst = self.dataset_inst.processes.get_first()

#     # get the number of nlo jets and the di-lepton pt
#     njets = events.LHE.NpNLO
#     pt = events.LHE.Vpt

#     # raise a warning if a datasets was already created for a specific "bin" (leaf process),
#     # but actually does not fit
#     njets_range = process_inst.x("njets", None)
#     if njets_range is not None:
#         outliers = (njets < njets_range[0]) | (njets >= njets_range[1])
#         if ak.any(outliers):
#             logger.warning(
#                 f"dataset {self.dataset_inst.name} is meant to contain njet values in the range "
#                 f"[{njets_range[0]}, {njets_range[0]}), but found {ak.sum(outliers)} events "
#                 "outside this range",
#             )
#     pt_range = process_inst.x("ptll", None)
#     if pt_range is not None:
#         outliers = (pt < pt_range[0]) | (pt >= pt_range[1])
#         if ak.any(outliers):
#             logger.warning(
#                 f"dataset {self.dataset_inst.name} is meant to contain ptll values in the range "
#                 f"[{pt_range[0]}, {pt_range[1]}), but found {ak.sum(outliers)} events outside this "
#                 "range",
#             )

#     # lookup the id and check for invalid values
#     process_ids = np.squeeze(np.asarray(self.id_table[self.key_func(njets, pt)].todense()))
#     invalid_mask = process_ids == 0
#     if ak.any(invalid_mask):
#         raise ValueError(
#             f"found {sum(invalid_mask)} dy events that could not be assigned to a process",
#         )

#     # store them
#     events = set_ak_column_i64(events, "process_id", process_ids)

#     return events


@process_ids_dy.setup
def process_ids_dy_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    # define stitching ranges for the DY datasets covered by this producer's dy_inclusive_dataset
    stitching_ranges: dict[NJetsRange, list[PtRange]] = {}
    for proc in self.dy_leaf_processes:
        njets = proc.x.njets
        stitching_ranges.setdefault(njets, [])
        if proc.has_aux("ptll"):
            stitching_ranges[njets].append(proc.x.ptll)

    # sort by the first element of the ptll range
    sorted_stitching_ranges: list[tuple[NJetsRange, list[PtRange]]] = [
        (nj_range, sorted(stitching_ranges[nj_range], key=lambda ptll_range: ptll_range[0]))
        for nj_range in sorted(stitching_ranges.keys(), key=lambda nj_range: nj_range[0])
    ]

    # define a key function that maps njets and pt to a unique key for use in a lookup table
    def key_func(njets, pt):
        # potentially convert single values into arrays
        single = False
        if isinstance(njets, int):
            assert isinstance(pt, (int, float))
            njets = np.array([njets], dtype=np.int32)
            pt = np.array([pt], dtype=np.float32)
            single = True

        # map into bins (index 0 means no binning)
        nj_bins = np.zeros(len(njets), dtype=np.int32)
        pt_bins = np.zeros(len(pt), dtype=np.int32)
        for nj_bin, (nj_range, pt_ranges) in enumerate(sorted_stitching_ranges, 1):
            # nj_bin
            nj_mask = (nj_range[0] <= njets) & (njets < nj_range[1])
            nj_bins[nj_mask] = nj_bin
            # pt_bin
            for pt_bin, (pt_min, pt_max) in enumerate(pt_ranges, 1):
                pt_mask = (pt_min <= pt) & (pt < pt_max)
                pt_bins[nj_mask & pt_mask] = pt_bin

        return (nj_bins[0], pt_bins[0]) if single else (nj_bins, pt_bins)

    self.key_func = key_func

    # define the lookup table
    max_nj_bin = len(sorted_stitching_ranges)
    max_pt_bin = max(map(len, stitching_ranges.values()))
    self.id_table = sp.sparse.lil_matrix((max_nj_bin + 1, max_pt_bin + 1), dtype=np.int64)

    # fill it
    for proc in self.dy_leaf_processes:
        key = key_func(proc.x.njets[0], proc.x("ptll", [-1])[0])
        self.id_table[key] = proc.id
