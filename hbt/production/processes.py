# coding: utf-8

"""
Process ID producer relevant for the stitching of the DY samples.
"""

from __future__ import annotations

import abc

import law
import order

from columnflow.production import Producer
from columnflow.production.cms.dy import gen_dilepton
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, Route
from columnflow.types import TYPE_CHECKING, Callable

from hbt.util import IF_DATASET_IS_DY_AMCATNLO, IF_DATASET_IS_DY_POWHEG, IF_DATASET_IS_W_LNU

np = maybe_import("numpy")
ak = maybe_import("awkward")
if TYPE_CHECKING:
    scipy = maybe_import("scipy")
    maybe_import("scipy.sparse")


logger = law.logger.get_logger(__name__)

LepId = int
NJetsRange = tuple[int, int]
PtRange = tuple[float, float]
MRange = tuple[int, int]


class stitched_process_ids(Producer):
    """
    General class to calculate process ids for stitched samples.

    Individual producers should derive from this class and set the following attributes:

    :param id_lut: scipy lookup table mapping processes variables (using compute_lut_index) to process ids
    :param compute_lut_index: function to generate keys for the lookup, receiving values of stitching columns
    :param stitching_columns: list of observables to use for stitching
    :param cross_check_translation_dict: dictionary to translate stitching columns to auxiliary
        fields of process objects, used for cross checking the validity of obtained ranges
    :param include_condition: condition for including stitching columns in used columns
    """

    recovery_thresholds = {}

    @abc.abstractproperty
    def stitching_columns(self) -> list[str | Route]:
        # must be overwritten by inheriting classes
        ...

    @property
    def uses_for_stitching(self) -> set[str | Route]:
        return self.stitching_columns

    @abc.abstractproperty
    def include_condition(self) -> Callable | None:
        # must be overwritten by inheriting classes
        ...

    @abc.abstractproperty
    def id_lut(self) -> scipy.sparse._lil.dok_matrix:
        # must be overwritten by inheriting classes
        ...

    @abc.abstractmethod
    def compute_lut_index(self, *values: ak.Array) -> int:
        # must be overwritten by inheriting classes
        ...

    @abc.abstractproperty
    def cross_check_translation_dict(self) -> dict[str, str]:
        # must be overwritten by inheriting classes
        ...

    def init_func(self, **kwargs) -> None:
        # if there is a include_condition set, apply it to both used and produced columns
        cond = lambda args: {self.include_condition(*args)} if self.include_condition else {*args}
        if (load := self.uses_for_stitching):
            self.uses |= cond(load)
        self.produces |= cond(["process_id"])

    def call_func(self, events: ak.Array, **kwargs) -> ak.Array:
        """
        Assigns each event a single process id, based on the stitching values extracted per event.
        This id can be used for the stitching of the respective datasets downstream.
        """
        # ensure that each dataset has exactly one process associated to it
        if len(self.dataset_inst.processes) != 1:
            raise NotImplementedError(
                f"dataset {self.dataset_inst.name} has {len(self.dataset_inst.processes)} processes "
                "assigned, which is not yet implemented",
            )
        process_inst = self.dataset_inst.processes.get_first()

        # get stitching observables
        stitching_values = [Route(obs).apply(events) for obs in self.stitching_columns]

        # run the cross check function if defined
        if callable(self.stitching_values_cross_check):
            self.stitching_values_cross_check(process_inst, stitching_values)

        # lookup the id and check for invalid values
        # (when the length of the events chunk is 1 by chance, the LUT access fails)
        process_ids = self.read_lut(stitching_values)
        invalid_mask = process_ids == 0
        if ak.any(invalid_mask):
            raise ValueError(f"found {sum(invalid_mask)} events that could not be assigned to a process")

        # when the assigned process ids contain the id of the main process of the dataset, it should be the only
        # identified process and there should be none other, as this would be logically inconsistent
        unique_ids = set(np.unique(process_ids))
        if process_inst.id in unique_ids and len(unique_ids) > 1:
            other_process_names = [
                self.config_inst.get_process(pid).name
                for pid in unique_ids - {process_inst.id}
            ]
            raise ValueError(
                f"dataset '{self.dataset_inst.name}' contains events that are assigned the main process "
                f"'{process_inst.name}' but also other processes '{','.join(other_process_names)}' which is "
                "logically inconsistent",
            )

        # store them
        events = set_ak_column(events, "process_id", process_ids, value_type=np.int64)

        return events

    def read_lut(self, values: list[np.array | ak.Array]) -> np.array:
        # compute the index where to access the LUT
        lut_index = self.compute_lut_index(*values)

        # when the length is exactly one, the sparse matrix access reduces and shape and fails, so artifically extend it
        extend = len(lut_index) == 1
        if extend:
            lut_index = np.tile(lut_index, 2)

        # lookup
        value = np.squeeze(np.asarray(self.id_lut[lut_index, 0].todense()))

        return value[:1] if extend else value

    def stitching_values_cross_check(
        self,
        process_inst: order.Process,
        stitching_values: list[ak.Array],
    ) -> None:
        # define lookup for stitching observable -> process auxiliary values to compare with
        # raise a warning if a datasets was already created for a specific "bin" (leaf process),
        # but actually does not fit
        for i, (column, values) in enumerate(zip(self.stitching_columns, stitching_values)):
            aux_name = self.cross_check_translation_dict[str(column)]
            if (aux_val := process_inst.x(aux_name, None)) is None:
                continue
            if isinstance(aux_val, (int, float)):
                unmatched = values != aux_val
                if ak.any(unmatched):
                    logger.warning(
                        f"dataset {self.dataset_inst.name} is meant to contain {aux_name} values of "
                        f"{aux_val}, but found {ak.sum(unmatched)} events with different values",
                    )
            elif isinstance(aux_val, (list, tuple)) and len(aux_val) == 2:
                aux_min, aux_max = aux_val
                min_outlier = values < aux_min
                max_outlier = values >= aux_max
                outliers = min_outlier | max_outlier
                if ak.any(outliers):
                    # exception or warning
                    msg = (
                        f"dataset {self.dataset_inst.name} is meant to contain {aux_name} values in the range "
                        f"[{aux_min}, {aux_max}), but found {ak.sum(outliers)} events outside this range"
                    )
                    if (recovery_threshold := self.recovery_thresholds.get(aux_name)) is None:
                        raise ValueError(msg)
                    else:
                        logger.warning(f"{msg}, trying to recover with treshold of {recovery_threshold}")
                    # cap values if they are within an acceptable range
                    if ak.any(min_outlier):
                        recover_mask = (aux_min - values[min_outlier]) <= recovery_threshold
                        # in case not all outliers can be recovered, do not deal with these cases but raise an error
                        if not ak.all(recover_mask):
                            raise ValueError(
                                f"dataset {self.dataset_inst.name} has {ak.sum(min_outlier)} events "
                                "with values below the minimum, but not all of them can be recovered",
                            )
                        stitching_values[i] = ak.where(min_outlier, aux_min, values)
                    if ak.any(max_outlier):
                        recover_mask = (values[max_outlier] - aux_max) <= recovery_threshold
                        # in case not all outliers can be recovered, do not deal with these cases but raise an error
                        if not ak.all(recover_mask):
                            raise ValueError(
                                f"dataset {self.dataset_inst.name} has {ak.sum(max_outlier)} events "
                                "with values below the maximum, but not all of them can be recovered",
                            )
                        stitching_values[i] = ak.where(max_outlier, aux_max - 1e-5, values)
            else:
                raise TypeError(
                    f"auxiliary value {aux_name} of process {process_inst.name} has unexpected type "
                    f"{type(aux_val).__name__}, no stitching value cross check can be performed",
                )


class stitched_process_ids_nj_pt(stitched_process_ids):
    """
    Process identifier for subprocesses spanned by a jet multiplicity and an optional pt range, such
    as DY or W->lnu, which have (e.g.) "*_1j" as well as "*_1j_pt100to200" subprocesses.
    """

    # id table is set during setup, create a non-abstract class member in the meantime
    id_lut = None

    # required aux fields
    njets_aux = "njets"
    pt_aux = "ptll"

    recovery_thresholds = {
        "ptll": 1.0,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # filled during setup
        self.stitching_ranges: list[tuple[NJetsRange, PtRange]] = []

        # check that aux fields are present in cross_check_translation_dict
        for field in (self.njets_aux, self.pt_aux):
            if field not in self.cross_check_translation_dict.values():
                raise ValueError(f"field {field} must be present in cross_check_translation_dict")

    @abc.abstractproperty
    def leaf_processes(self) -> list[order.Process]:
        # must be overwritten by inheriting classes
        ...

    def setup_func(self, task: law.Task, **kwargs) -> None:
        import scipy.sparse

        # fill stitching ranges
        for proc in self.leaf_processes:
            njets = proc.x(self.njets_aux, (0, np.inf))
            pt = proc.x(self.pt_aux, (0.0, np.inf))
            self.stitching_ranges.append((njets, pt))

        # make unique and sort
        self.stitching_ranges = sorted(set(self.stitching_ranges))

        # define the lookup table
        self.id_lut = scipy.sparse.dok_matrix((len(self.stitching_ranges), 1), dtype=np.int64)

        # fill it
        for proc in self.leaf_processes:
            index = self.compute_lut_index(proc.x(self.njets_aux, [0])[0], proc.x(self.pt_aux, [0])[0])
            self.id_lut[index, 0] = proc.id

    def compute_lut_index(
        self,
        njets: int | np.ndarray,
        pt: int | float | np.ndarray,
    ) -> int | np.ndarray:
        # potentially convert single values into arrays
        single = False
        if isinstance(njets, int):
            assert isinstance(pt, (int, float))
            njets = np.array([njets], dtype=np.int32)
            pt = np.array([pt], dtype=np.float32)
            single = True

        # map into bins (-1 means no binning and should raise errors)
        indices = -np.ones(len(njets), dtype=np.int32)
        for index, (nj_range, pt_range) in enumerate(self.stitching_ranges):
            nj_mask = (nj_range[0] <= njets) & (njets < nj_range[1])
            pt_mask = (pt_range[0] <= pt) & (pt < pt_range[1])
            mask = nj_mask & pt_mask
            if np.any(indices[mask] != -1):
                raise RuntimeError(
                    f"found misconfigured leaf process definitions while assigning process ids with {self.cls_name} "
                    f"producer in dataset {self.dataset_inst.name} (hint: check 'stitching_ranges')",
                )
            indices[mask] = index

        return indices[0] if single else indices


class stitched_process_ids_nj(stitched_process_ids):
    """
    Process identifier for subprocesses spanned by a jet multiplicity, such
    as DY or W->lnu, which have (e.g.) "*_1j" subprocesses.
    """

    # id table is set during setup, create a non-abstract class member in the meantime
    id_lut = None

    # required aux fields
    njets_aux = "njets"

    recovery_thresholds = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # filled during setup
        self.stitching_ranges: list[NJetsRange] = []

        # check that aux fields are present in cross_check_translation_dict
        if self.njets_aux not in self.cross_check_translation_dict.values():
            raise ValueError(f"field {self.njets_aux} must be present in cross_check_translation_dict")

    @abc.abstractproperty
    def leaf_processes(self) -> list[order.Process]:
        # must be overwritten by inheriting classes
        ...

    def setup_func(self, task: law.Task, **kwargs) -> None:
        import scipy.sparse

        # fill stitching ranges
        for proc in self.leaf_processes:
            njets = proc.x(self.njets_aux, (0, np.inf))
            self.stitching_ranges.append(njets)

        # make unique and sort
        self.stitching_ranges = sorted(set(self.stitching_ranges))

        # define the lookup table
        self.id_lut = scipy.sparse.dok_matrix((len(self.stitching_ranges), 1), dtype=np.int64)

        # fill it
        for proc in self.leaf_processes:
            index = self.compute_lut_index(proc.x(self.njets_aux, [0])[0])
            self.id_lut[index, 0] = proc.id

    def compute_lut_index(
        self,
        njets: int | np.ndarray,
    ) -> int | np.ndarray:
        # potentially convert single values into arrays
        single = False
        if isinstance(njets, int):
            njets = np.array([njets], dtype=np.int32)
            single = True

        # map into bins (-1 means no binning and should raise errors)
        indices = -np.ones(len(njets), dtype=np.int32)
        for index, nj_range in enumerate(self.stitching_ranges):
            mask = (nj_range[0] <= njets) & (njets < nj_range[1])
            if np.any(indices[mask] != -1):
                raise RuntimeError(
                    f"found misconfigured leaf process definitions while assigning process ids with {self.cls_name} "
                    f"producer in dataset {self.dataset_inst.name} (hint: check 'stitching_ranges')",
                )
            indices[mask] = index

        return indices[0] if single else indices


class stitched_process_ids_lep_nj_pt(stitched_process_ids):
    """
    Same as :py:class:`stitched_process_ids_nj_pt`, but with an additional generator-level lepton pair identification.
    """

    # id table is set during setup, create a non-abstract class member in the meantime
    id_lut = None

    # required aux fields
    lep_aux = "lep_id"
    njets_aux = "njets"
    pt_aux = "ptll"

    recovery_thresholds = {
        "ptll": 1.0,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # filled during setup
        self.stitching_ranges: list[tuple[LepId, NJetsRange, PtRange]] = []

        # check that aux fields are present in cross_check_translation_dict
        for field in (self.lep_aux, self.njets_aux, self.pt_aux):
            if field not in self.cross_check_translation_dict.values():
                raise ValueError(f"field {field} must be present in cross_check_translation_dict")

    @abc.abstractproperty
    def leaf_processes(self) -> list[order.Process]:
        # must be overwritten by inheriting classes
        ...

    def init_func(self, **kwargs) -> None:
        super().init_func(**kwargs)
        self.uses.add(gen_dilepton)

    def call_func(self, events: ak.Array, **kwargs) -> ak.Array:
        events = self[gen_dilepton](events, **kwargs)
        return super().call_func(events, **kwargs)

    def setup_func(self, task: law.Task, **kwargs) -> None:
        import scipy.sparse

        # fill stitching ranges
        for proc in self.leaf_processes:
            lep = proc.x(self.lep_aux, 0)
            njets = proc.x(self.njets_aux, (0, np.inf))
            pt = proc.x(self.pt_aux, (0.0, np.inf))
            self.stitching_ranges.append((lep, njets, pt))

        # make unique and sort
        self.stitching_ranges = sorted(set(self.stitching_ranges))

        # define the lookup table
        self.id_lut = scipy.sparse.dok_matrix((len(self.stitching_ranges), 1), dtype=np.int64)

        # fill it
        for proc in self.leaf_processes:
            index = self.compute_lut_index(
                proc.x(self.lep_aux, 0),
                proc.x(self.njets_aux, [0])[0],
                proc.x(self.pt_aux, [0])[0],
            )
            self.id_lut[index, 0] = proc.id

    def compute_lut_index(
        self,
        lep: int | np.ndarray,
        njets: int | np.ndarray,
        pt: int | float | np.ndarray,
    ) -> int | np.ndarray:
        # potentially convert single values into arrays
        single = False
        if isinstance(njets, int):
            assert isinstance(lep, int)
            assert isinstance(pt, (int, float))
            lep = np.array([lep], dtype=np.int32)
            njets = np.array([njets], dtype=np.int32)
            pt = np.array([pt], dtype=np.float32)
            single = True

        # map into bins (-1 means no binning and should raise errors)
        indices = -np.ones(len(njets), dtype=np.int32)
        for index, (_lep, nj_range, pt_range) in enumerate(self.stitching_ranges):
            lep_mask = (_lep == 0) | (_lep == lep)
            nj_mask = (nj_range[0] <= njets) & (njets < nj_range[1])
            pt_mask = (pt_range[0] <= pt) & (pt < pt_range[1])
            mask = lep_mask & nj_mask & pt_mask
            if np.any(indices[mask] != -1):
                raise RuntimeError(
                    f"found misconfigured leaf process definitions while assigning process ids with {self.cls_name} "
                    f"producer in dataset {self.dataset_inst.name} (hint: check 'stitching_ranges')",
                )
            indices[mask] = index

        return indices[0] if single else indices


class stitched_process_ids_m(stitched_process_ids):
    """
    Process identifier for subprocesses spanned by the mll mass.
    """

    # id table is set during setup, create a non-abstract class member in the meantime
    id_lut = None

    # required aux fields
    var_aux = "mll"

    recovery_thresholds = {
        "mll": 1.0,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # setup during setup
        self.sorted_stitching_ranges: list[tuple[MRange]]

        # check that aux field is present in cross_check_translation_dict
        for field in (self.var_aux,):
            if field not in self.cross_check_translation_dict.values():
                raise ValueError(f"field {field} must be present in cross_check_translation_dict")

    @abc.abstractproperty
    def leaf_processes(self) -> list[order.Process]:
        # must be overwritten by inheriting classes
        ...

    def init_func(self, **kwargs) -> None:
        # if there is a include_condition set, apply it to both used and produced columns
        cond = lambda args: {self.include_condition(*args)} if self.include_condition else {*args}
        self.uses |= cond(["LHEPart.{pt,eta,mass,phi,status,pdgId}"])
        self.produces |= cond(["process_id"])

    def setup_func(self, task: law.Task, **kwargs) -> None:
        import scipy.sparse

        # define stitching ranges for the DY datasets covered by this producer's dy_inclusive_dataset
        stitching_ranges = [
            proc.x(self.var_aux)
            for proc in self.leaf_processes
        ]

        # sort by the first element of the range
        self.sorted_stitching_ranges = sorted(stitching_ranges, key=lambda mll_range: mll_range[0])

        # define the lookup table
        max_var_bin = len(self.sorted_stitching_ranges)
        self.id_lut = scipy.sparse.dok_matrix((max_var_bin + 1, 1), dtype=np.int64)

        # fill it
        for proc in self.leaf_processes:
            key = self.compute_lut_index(proc.x(self.var_aux)[0])
            self.id_lut[key] = proc.id

    def compute_lut_index(
        self,
        mll: int | float | np.ndarray,
    ) -> tuple[int] | tuple[np.ndarray]:
        # potentially convert single values into arrays
        single = False
        if isinstance(mll, (int, float)):
            mll = np.array([mll], dtype=(np.int32))
            single = True

        # map into bins (index 0 means no binning)
        mll_bins = np.zeros(len(mll), dtype=np.int32)
        for mll_bin, (mll_min, mll_max) in enumerate(self.sorted_stitching_ranges, 1):
            mll_mask = (mll_min <= mll) & (mll < mll_max)
            mll_bins[mll_mask] = mll_bin

        return (mll_bins[0],) if single else (mll_bins,)

    def call_func(self, events: ak.Array, **kwargs) -> ak.Array:
        # produce the mass variable and save it as LHEmll, then call the super class
        abs_pdg_id = abs(events.LHEPart.pdgId)
        leps = events.LHEPart[(abs_pdg_id >= 11) & (abs_pdg_id <= 16) & (events.LHEPart.status == 1)]
        if ak.any((num_leps := ak.num(leps)) != 2):
            raise ValueError(f"expected exactly two leptons in the event, but found {set(num_leps)}")
        mll = leps.sum(axis=-1).mass
        events = set_ak_column(events, "LHEmll", mll, value_type=np.float32)

        return super().call_func(events, **kwargs)


process_ids_dy_amcatnlo_2223 = stitched_process_ids_nj_pt.derive("process_ids_dy_amcatnlo_2223", cls_dict={
    "stitching_columns": ["LHE.NpNLO", "LHE.Vpt"],
    "cross_check_translation_dict": {"LHE.NpNLO": "njets", "LHE.Vpt": "ptll"},
    "include_condition": IF_DATASET_IS_DY_AMCATNLO,
    # still misses leaf_processes, must be set dynamically
})

process_ids_dy_lep_amcatnlo_2223 = stitched_process_ids_lep_nj_pt.derive("process_ids_dy_lep_amcatnlo_2223", cls_dict={
    "stitching_columns": ["gen_dilepton_pdgid", "LHE.NpNLO", "LHE.Vpt"],
    "uses_for_stitching": ["LHE.NpNLO", "LHE.Vpt"],  # gen_dilepton_pdgid must haven been produced dynamically
    "cross_check_translation_dict": {"gen_dilepton_pdgid": "lep_id", "LHE.NpNLO": "njets", "LHE.Vpt": "ptll"},
    "include_condition": IF_DATASET_IS_DY_AMCATNLO,
    # still misses leaf_processes, must be set dynamically
})

process_ids_dy_powheg_2223 = stitched_process_ids_m.derive("process_ids_dy_powheg_2223", cls_dict={
    "stitching_columns": ["LHEmll"],
    "cross_check_translation_dict": {"LHEmll": "mll"},
    "include_condition": IF_DATASET_IS_DY_POWHEG,
    # still misses leaf_processes, must be set dynamically
})

process_ids_dy_mumu_amcatnlo_24 = stitched_process_ids_nj.derive("process_ids_dy_mumu_amcatnlo_24", cls_dict={
    "stitching_columns": ["LHE.NpNLO"],
    "cross_check_translation_dict": {"LHE.NpNLO": "njets"},
    "include_condition": IF_DATASET_IS_DY_AMCATNLO,
    # still misses leaf_processes, must be set dynamically
})

process_ids_dy_ee_amcatnlo_24 = stitched_process_ids_nj.derive("process_ids_dy_ee_amcatnlo_24", cls_dict={
    "stitching_columns": ["LHE.NpNLO"],
    "cross_check_translation_dict": {"LHE.NpNLO": "njets"},
    "include_condition": IF_DATASET_IS_DY_AMCATNLO,
    # still misses leaf_processes, must be set dynamically
})

process_ids_dy_tautau_amcatnlo_24 = stitched_process_ids_nj.derive("process_ids_dy_tautau_amcatnlo_24", cls_dict={
    "stitching_columns": ["LHE.NpNLO"],
    "cross_check_translation_dict": {"LHE.NpNLO": "njets"},
    "include_condition": IF_DATASET_IS_DY_AMCATNLO,
    # still misses leaf_processes, must be set dynamically
})

process_ids_w_lnu_amcatnlo_2223 = stitched_process_ids_nj_pt.derive("process_ids_w_lnu_amcatnlo_2223", cls_dict={
    "stitching_columns": ["LHE.NpNLO", "LHE.Vpt"],
    "cross_check_translation_dict": {"LHE.NpNLO": "njets", "LHE.Vpt": "ptll"},
    "include_condition": IF_DATASET_IS_W_LNU,
    # still misses leaf_processes, must be set dynamically
})
