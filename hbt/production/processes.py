# coding: utf-8

"""
Process ID producer relevant for the stitching of the DY samples.
"""

import functools

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, InsertableDict
from columnflow.columnar_util import set_ak_column

from hbt.util import IF_DATASET_IS_DY

np = maybe_import("numpy")
ak = maybe_import("awkward")
sp = maybe_import("scipy")
maybe_import("scipy.sparse")


logger = law.logger.get_logger(__name__)

# helper
set_ak_column_i64 = functools.partial(set_ak_column, value_type=np.int64)


@producer(
    uses={IF_DATASET_IS_DY("LHE.NpNLO", "LHE.Vpt")},
    produces={IF_DATASET_IS_DY("process_ids")},
)
def process_ids_dy(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Assigns each dy event a single process id, based on the number of jets and the di-lepton pt of
    the LHE record. This is used for the stitching of the DY samples.
    """
    # as always, we assume that each dataset has exactly one process associated to it
    if len(self.dataset_inst.processes) != 1:
        raise NotImplementedError(
            f"dataset {self.dataset_inst.name} has {len(self.dataset_inst.processes)} processes "
            "assigned, which is not yet implemented",
        )

    # get the number of nlo jets and the di-lepton pt
    njets = events.LHE.NpNLO
    pt = events.LHE.Vpt

    # raise a warning if a datasets was already created for a specific "bin" (leaf process),
    # but actually does not fit
    njets_nominal = self.dataset_inst.x("njets")
    if njets_nominal is not None and ak.any(njets != njets_nominal):
        logger.warning(
            f"dataset {self.dataset_inst.name} is meant to contain only {njets_nominal} jets, "
            f"but the LHE record contains different jet multiplicities: {set(njets)}",
        )
    pt_range_nominal = self.dataset_inst.x("ptll")
    if pt_range_nominal is not None:
        outliers = (pt < pt_range_nominal[0]) | (pt >= pt_range_nominal[1])
        if ak.any(outliers):
            logger.warning(
                f"dataset {self.dataset_inst.name} is meant to contain ptZ values in the range "
                f"{pt_range_nominal[0]} to {pt_range_nominal[1]}, but found {ak.sum(outliers)} "
                "events outside this range",
            )

    # get the LHE Njets and Vpt to assign each event into a leaf process using the lookup table
    process_ids = np.array(self.id_table[0, self.key_func(njets, pt)].todense())[0]
    events = set_ak_column_i64(events, "process_id", process_ids)

    return events


@process_ids_dy.setup
def process_ids_dy_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    # define stitching ranges for the DY datasets covered by this producer's dy_inclusive_dataset
    # TODO: extract the following from the datasets' aux info
    stitching_ranges = {
        0: None,
        1: [(0, 40), (40, 100), (100, 200), (200, 400), (400, 600), (600, float("inf"))],
        2: [(0, 40), (40, 100), (100, 200), (200, 400), (400, 600), (600, float("inf"))],
    }

    # define a key function that maps njets and pt to a unique key for use in a lookup table
    def key_func(njets, pt):
        # potentially convert single values into arrays
        single = False
        if isinstance(njets, int) and isinstance(pt, (int, float)):
            njets = np.array([njets], dtype=np.int32)
            pt = np.array([pt], dtype=np.float32)
            single = True

        # map into pt bins (index 0 means no binning)
        pt_bin = np.zeros(len(pt), dtype=np.int32)
        for nj, pt_ranges in stitching_ranges.items():
            if not pt_ranges:
                continue
            nj_mask = njets == nj
            for i, (pt_min, pt_max) in enumerate(pt_ranges, 1):
                pt_mask = (pt_min <= pt) & (pt < pt_max)
                pt_bin[nj_mask & pt_mask] = i
        # values larger than the largest configured njet value are set to 0
        pt_bin[njets >= (nj + 1)] = 0

        # compute the key
        key = njets * 100 + pt_bin

        return key[0] if single else key

    # save it
    self.key_func = key_func

    # define the lookup table and fill it
    max_nj = max(stitching_ranges.keys()) + 1
    self.id_table = sp.sparse.lil_matrix((1, key_func(max_nj, 0) + 1), dtype=np.int64)
    # TODO: fill
