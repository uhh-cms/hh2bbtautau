# coding: utf-8

"""
Tasks that create files for synchronization efforts with other frameworks.
"""

from __future__ import annotations

from functools import reduce
from operator import or_

import luigi
import law

from columnflow.tasks.framework.base import Requirements, DatasetTask
from columnflow.tasks.framework.mixins import ProducersMixin, MLModelsMixin, ChunkedIOMixin
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.external import GetDatasetLFNs
from columnflow.tasks.reduction import ReducedEventsUser
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.ml import MLEvaluation
from columnflow.util import dev_sandbox

from hbt.tasks.base import HBTTask
from hbt.util import hash_events


class CheckExternalLFNOverlap(
    HBTTask,
    DatasetTask,
):

    lfn = luigi.Parameter(
        description="local path to an external LFN to check for overlap with the dataset",
        # fetched via nanogen's FetchLFN
        default="/pnfs/desy.de/cms/tier2/store/user/bwieders/nanogen_store/FetchLFN/store/sync/skim_2024_v2/Run3_2022/GluGlutoHHto2B2Tau_kl_1p00_kt_1p00_c2_0p00/nano_0.root",  # noqa
    )

    # no versioning required
    version = None

    # default sandbox, might be overwritten by calibrator function
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        GetDatasetLFNs=GetDatasetLFNs,
    )

    def requires(self):
        return self.reqs.GetDatasetLFNs.req(self)

    def output(self):
        return {
            "overlap": self.target("lfn_overlap.json"),
            "index_variables": self.target("index_variables.parquet"),
        }

    def run(self):
        import awkward as ak
        import numpy as np

        # load the index columns of the reference lfn
        output = self.output()

        with self.publish_step("loading reference ids"):
            ref_hashes = hash_events(self.load_nano_index(law.LocalFileTarget(self.lfn)))

        # loop over all lfns in the dataset
        n_files = self.dataset_inst.n_files
        lfns_task = self.requires()
        relative_overlap = {"rel_to_file": {}, "rel_to_ref": {}}
        overlapping_identifier = []

        for i, lfn_target in lfns_task.iter_nano_files(self, lfn_indices=list(range(n_files))):
            with self.publish_step(f"loading ids of file {i}"):

                file_arr = self.load_nano_index(lfn_target)
                file_hashes = hash_events(file_arr)
            # find unique hashes in the reference and the file
            # faster than np.
            overlapping_mask = np.isin(
                file_hashes,
                ref_hashes,
                assume_unique=True,
                kind="sort",
            )
            num_overlapping = np.sum(overlapping_mask)
            if num_overlapping:
                # calculate the relative overlaps
                # relative to file indicates how many events of the reference are within the files
                relative_overlap["rel_to_file"][str(i)] = np.sum(num_overlapping) / len(file_hashes)
                relative_overlap["rel_to_ref"][str(i)] = np.sum(num_overlapping) / len(ref_hashes)
                # apply mask and store the overlapping identifiers
                overlapping_identifier.extend(file_arr[overlapping_mask])

        # sanity checks
        assert relative_overlap["rel_to_file"], "no overlap between reference and files found"

        reference_sum = np.sum([ref_value for ref_value in relative_overlap["rel_to_ref"].values()])
        assert reference_sum.item() == 1, f"reference sum is not 1 but {reference_sum.item()}"

        output["overlap"].dump(
            relative_overlap,
            formatter="json",
        )

        output["index_variables"].dump(
            ak.Array(overlapping_identifier),
            formatter="awkward",
        )

    @classmethod
    def load_nano_index(cls, lfn_target: law.FileSystemFileTarget) -> set[int]:
        fields = ["event", "luminosityBlock", "run"]
        arr = lfn_target.load(formatter="uproot")["Events"].arrays(fields)
        return arr


class CreateSyncFiles(
    HBTTask,
    ReducedEventsUser,
    ChunkedIOMixin,
    ProducersMixin,
    MLModelsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    filter_file = luigi.Parameter(
        description="local path to a file containing event unique identifier to filter them out",
        default="",
    )

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        ReducedEventsUser.reqs,
        RemoteWorkflow.reqs,
        ProduceColumns=ProduceColumns,
        MLEvaluation=MLEvaluation,
    )

    def workflow_requires(self):
        reqs = super().workflow_requires()

        # require the full merge forest
        reqs["events"] = self.reqs.ProvideReducedEvents.req(self)

        if not self.pilot:
            if self.producer_insts:
                reqs["producers"] = [
                    self.reqs.ProduceColumns.req(self, producer=producer_inst.cls_name)
                    for producer_inst in self.producer_insts
                    if producer_inst.produced_columns
                ]
            if self.ml_model_insts:
                reqs["ml"] = [
                    self.reqs.MLEvaluation.req(self, ml_model=m)
                    for m in self.ml_models
                ]

        return reqs

    def requires(self):
        reqs = {"events": self.reqs.ProvideReducedEvents.req(self)}

        if self.producer_insts:
            reqs["producers"] = [
                self.reqs.ProduceColumns.req(self, producer=producer_inst.cls_name)
                for producer_inst in self.producer_insts
                if producer_inst.produced_columns
            ]
        if self.ml_model_insts:
            reqs["ml"] = [
                self.reqs.MLEvaluation.req(self, ml_model=m)
                for m in self.ml_models
            ]

        return reqs

    workflow_condition = ReducedEventsUser.workflow_condition.copy()

    @workflow_condition.output
    def output(self):
        return {
            "normal": self.target(f"sync_{self.dataset_inst.name}_{self.branch}.csv"),
            "hhbtag": self.target(f"sync_{self.dataset_inst.name}_{self.branch}_hhbtag.csv"),
            "resonant": self.target(f"sync_{self.dataset_inst.name}_{self.branch}_resonant.csv"),
        }

    @law.decorator.log
    @law.decorator.localize
    def run(self):
        import awkward as ak
        import numpy as np
        from columnflow.columnar_util import EMPTY_FLOAT, EMPTY_INT, update_ak_array, set_ak_column

        # prepare inputs and outputs
        inputs = self.input()
        output = self.output()

        # iterate over chunks of events and diffs
        files = [inputs["events"]["events"].abspath]
        if self.producer_insts:
            files.extend([inp["columns"].abspath for inp in inputs["producers"]])
        if self.ml_model_insts:
            files.extend([inp["mlcolumns"].abspath for inp in inputs["ml"]])

        # helper to replace our internal empty placeholders with a custom ones
        # dtype -> (our, custom)
        empty = {
            np.float32: (EMPTY_FLOAT, EMPTY_FLOAT),
            np.float64: (EMPTY_FLOAT, EMPTY_FLOAT),
            np.int32: (EMPTY_INT, EMPTY_INT),
            np.int64: (EMPTY_INT, EMPTY_INT),
            np.uint64: (EMPTY_INT, EMPTY_INT),
        }
        def replace_empty(arr, dtype=np.float32):
            default, custom = empty[dtype]
            if custom != default:
                arr = ak.where(arr == default, custom, arr)
            return arr

        # helper to pad nested fields with an empty float if missing
        def pad_nested(arr, n, *, axis=1, dtype=np.float32):
            padded = ak.pad_none(arr, n, axis=axis)
            return ak.values_astype(ak.fill_none(padded, empty[dtype][1]), dtype)

        # helper to pad and select the last element on the first inner axis
        def select(arr, idx, dtype=np.float32):
            padded = pad_nested(arr, idx + 1, axis=-1, dtype=dtype)
            return replace_empty((padded if arr.ndim == 1 else padded[:, idx]), dtype=dtype)

        # helper to select leptons
        def select_leptons(events: ak.Array, common_fields: dict[str, int | float]) -> ak.Array:
            # ensure all lepton arrays have the same common fields
            leptons = [events.Electron, events.Muon, events.Tau]
            for i in range(len(leptons)):
                lepton = leptons[i]
                for field, default in common_fields.items():
                    if field not in lepton.fields:
                        lepton = set_ak_column(lepton, field, default)
                leptons[i] = lepton
            # concatenate (first event any lepton, second alsways tau) and add to events
            return set_ak_column(events, "Lepton", ak.concatenate(leptons, axis=1))

        def uint64_to_str(array: ak.Array) -> ak.Array:
            # -99999 casted to uint64
            empty_uint64_str = str(np.iinfo(np.uint64).max + empty[np.uint64][0] + 1)
            array = np.asarray(array, dtype=np.str_)
            empty_mask = array == empty_uint64_str
            array = np.where(empty_mask, str(empty[np.uint64][0]), array)
            return array

        def get_category_id(category_id: ak.Array, config_inst, category_replacement_map, axis=-1) -> ak.Array:
            # Helper function to map leaf category ids ids specified in *category_replacement_map*.
            def get_mapping(config_inst, demanded_categories):
                # get all leaf ids for the required categories
                all_categories = config_inst.get_category(-1).categories
                categories = {category.name: category.id for category in all_categories}
                mapping = {}
                for cat_name in demanded_categories:
                    leaves = config_inst.get_category(categories[cat_name]).get_leaf_categories()
                    mapping[cat_name] = [category.id for category in leaves]
                return mapping

            root_category_map = get_mapping(config_inst, category_replacement_map.keys())
            flat_category_view = np.asarray(ak.flatten(category_id, axis=axis))
            output_array = np.zeros(shape=(len(category_id)), dtype=np.int32)
            ak_layout = ak.num(category_id)

            # replace ids with category ids by replacement
            for cat_name, replacement_value in category_replacement_map.items():
                # events can have multiple categories,
                ids = root_category_map[cat_name]
                mask = ak.any(
                    ak.unflatten(np.isin(flat_category_view, ids), ak_layout), axis=axis,
                )
                output_array[mask] = replacement_value
            return output_array

        # event chunk loop
        # optional filter to get only the events that overlap with given external LFN
        if self.filter_file:
            with self.publish_step("loading reference ids"):
                events_to_filter = law.LocalFileTarget(self.filter_file).load(formatter="awkward")
                filter_events = hash_events(events_to_filter)

        for (events, *columns), pos in self.iter_chunked_io(
            files,
            source_type=len(files) * ["awkward_parquet"],
            pool_size=1,
        ):
            # optional check for overlapping inputs
            if self.check_overlapping_inputs:
                self.raise_if_overlapping([events] + list(columns))
            # add additional columns
            events = update_ak_array(events, *columns)
            # apply mask if optional filter is given
            # calculate mask by using 1D hash values
            if self.filter_file:
                mask = np.isin(
                    hash_events(events),
                    filter_events,
                    assume_unique=True,
                    kind="sort",
                )

                # apply mask
                events = events[mask]

            if len(events) == 0:
                raise ValueError(
                    """
                    No events left after filtering.
                    "Check if correct overlap file is used
                    """,
                )

            #
            # create new columns, do mapping
            #

            # map category ids to given values
            category_id = get_category_id(
                category_id=events.category_ids,
                config_inst=self.config_inst,
                category_replacement_map={
                    "res1b": 0,
                    "res2b": 1,
                    "boosted": 2},
                axis=-1,
            )

            # insert leptons
            events = select_leptons(events, {"rawDeepTau2018v2p5VSjet": empty[np.float32][1]})
            met_name = self.config_inst.x.met_name

            index_variables = {
                "event": events.event,
                "run": events.run,
                "lumi": events.luminosityBlock,
            }

            #
            # combine all columns in a pandas df and save it
            #
            df_res = ak.to_dataframe({
                **index_variables,
                "dnn_dy": events.res_dnn_dy,
                "dnn_hh": events.res_dnn_hh,
                "dnn_tt": events.res_dnn_tt,
                **{
                    field.replace("sync_", ""): events[field]
                    for field in events.fields
                    if field.startswith("sync_res_dnn")},
            })
            df_hhb = ak.to_dataframe({
                **index_variables,
                **{
                    f"hhbtag_score{i + 1}": select(events.Jet.hhbtag, i)
                    for i in range(3)
                },
                **{
                    f"hhbtag{i + 1}_{field.replace('sync_hhbtag', '')}": select(events[field], i)
                    for field in events.fields if field.startswith("sync_hhbtag")
                    for i in range(3)
                },
            })

            df = ak.to_dataframe({
                **index_variables,
                # high-level events variables
                "channel_id": events.channel_id,
                "os": events.leptons_os * 1,
                "iso": events.tau2_isolated * 1,
                "category_id": category_id,
                "deterministic_seed": uint64_to_str(events.deterministic_seed),
                # jets
                **reduce(or_, (
                    {
                        f"jet{i + 1}_pt": select(events.Jet.pt, i),
                        f"jet{i + 1}_eta": select(events.Jet.eta, i),
                        f"jet{i + 1}_phi": select(events.Jet.phi, i),
                        f"jet{i + 1}_mass": select(events.Jet.mass, i),
                        f"jet{i + 1}_deterministic_seed": uint64_to_str(
                            select(events.Jet.deterministic_seed, i, np.uint64),
                        ),
                    }
                    for i in range(2)
                )),
                # electron specific variables
                "e1_deterministic_seed": uint64_to_str(
                    select(events.Electron.deterministic_seed, 0, np.uint64),
                ),
                "e2_deterministic_seed": uint64_to_str(
                    select(events.Electron.deterministic_seed, 1, np.uint64),
                ),
                # combined leptons
                **reduce(or_, (
                    {
                        f"lep{i + 1}_pt": select(events.Lepton.pt, i),
                        f"lep{i + 1}_phi": select(events.Lepton.phi, i),
                        f"lep{i + 1}_eta": select(events.Lepton.eta, i),
                        f"lep{i + 1}_charge": select(events.Lepton.charge, i),
                        f"lep{i + 1}_deeptauvsjet": select(events.Lepton.rawDeepTau2018v2p5VSjet, i),
                    }
                    for i in range(2)
                )),
                # met
                "met_pt": events[met_name].pt,
                "met_phi": events[met_name].phi,
                **({} if self.config_inst.campaign.x.version < 14 else {
                    "met_significance": select(events[met_name].significance, 0),
                    "met_covXX": select(events[met_name].covXX, 0),
                    "met_covXY": select(events[met_name].covXY, 0),
                    "met_covYY": select(events[met_name].covYY, 0),
                }),
                # fatjets
                **reduce(or_, (
                    {
                        f"fatjet{i + 1}_pt": select(events.FatJet.pt, i),
                        f"fatjet{i + 1}_eta": select(events.FatJet.eta, i),
                        f"fatjet{i + 1}_phi": select(events.FatJet.phi, i),
                        f"fatjet{i + 1}_mass": select(events.FatJet.mass, i),
                    }
                    for i in range(2)
                )),
                "dnn_dy": events.res_dnn_dy,
                "dnn_hh": events.res_dnn_hh,
                "dnn_tt": events.res_dnn_tt,
            })
            # save as csv in output, append if necessary
            output["normal"].dump(
                df,
                formatter="pandas",
                index=False,
                header=pos.index == 0,
                mode="w" if pos.index == 0 else "a",
            )

            output["resonant"].dump(
                df_res,
                formatter="pandas",
                index=False,
                header=pos.index == 0,
                mode="w" if pos.index == 0 else "a",
            )

            output["hhbtag"].dump(
                df_hhb,
                formatter="pandas",
                index=False,
                header=pos.index == 0,
                mode="w" if pos.index == 0 else "a",
            )


check_overlap_tasks = law.config.get_expanded("analysis", "check_overlapping_inputs", [], split_csv=True)
CreateSyncFiles.check_overlapping_inputs = ChunkedIOMixin.check_overlapping_inputs.copy(
    default=CreateSyncFiles.task_family in check_overlap_tasks,
    add_default_to_description=True,
)
