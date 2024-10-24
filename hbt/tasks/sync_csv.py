# coding: utf-8

"""
Task to create a single file csv file for framework sync with other frameworks.
"""

import luigi
import law


from columnflow.tasks.framework.base import Requirements, AnalysisTask, wrapper_factory
from columnflow.tasks.framework.mixins import ProducersMixin, MLModelsMixin, ChunkedIOMixin, SelectorMixin
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.reduction import ReducedEventsUser
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.ml import MLEvaluation
from columnflow.tasks.selection import MergeSelectionMasks
from columnflow.util import dev_sandbox, DotDict


class CreateSyncFile(
    MLModelsMixin,
    ProducersMixin,
    ChunkedIOMixin,
    ReducedEventsUser,
    SelectorMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    file_type = luigi.ChoiceParameter(
        default="csv",
        choices=("csv",),
        description="the file type to create; choices: csv; default: csv",
    )

    # upstream requirements
    reqs = Requirements(
        ReducedEventsUser.reqs,
        RemoteWorkflow.reqs,
        ProduceColumns=ProduceColumns,
        MLEvaluation=MLEvaluation,
        MergeSelectionMasks=MergeSelectionMasks,
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
        return self.target(f"sync_file_{self.branch}.{self.file_type}")

    @law.decorator.log
    @law.decorator.localize(input=True, output=True)
    @law.decorator.safe_output
    def run(self):
        from columnflow.columnar_util import (
            Route, RouteFilter, mandatory_coffea_columns, update_ak_array,
            sort_ak_fields, EMPTY_FLOAT,
        )
        import awkward as ak
        import pandas as pd
        def sync_columns():
            columns = {
                "physics_objects": {
                    "Jet.{pt,eta,phi,mass}": 2,
                },
                "flat_objects": [
                    "event",
                    "run",
                    "channel_id",
                    "leptons_os",
                    "luminosityBlock",
                    "lep*",
                ],
            }
            mapping = {
                "luminosityBlock": "lumi",
                "leptons_os": "is_os",
            }
            return columns, mapping

        def lepton_selection(events, attributes=["pt", "eta", "phi", "charge"]):
            first_lepton = ak.concatenate([events["Electron"], events["Muon"]], axis=1)
            second_lepton = events["Tau"]
            for attribute in attributes:
                events[f"lep1_{attribute}"] = first_lepton[attribute]
                events[f"lep2_{attribute}"] = second_lepton[attribute]
            return events

        def awkward_to_pandas(events, physics_objects, flat_objects=["event"]):
            """Helper function to convert awkward arrays to pandas dataframes.

            Args:
                events (ak.array): Awkward array with nested structure.
                physics_objects (Dict[str]): Dict of physics objects to consider, with value representing padding.
                attributes (List[str]): List of attributes to consider.
                flat_objects (List[str]): List of additional columns to consider, these are not padded.

            Returns:
                pd.DataFrame: Pandas dataframe with flattened structure of the awkward array.
            """
            events = sort_ak_fields(events)
            f = DotDict()
            # add meta columns (like event number, ...)
            # these columns do not need padding

            # columns that need no padding (like event number, ...)
            # resolve glob patterns
            for flat_object_pattern in flat_objects:
                for field in events.fields:
                    if law.util.fnmatch.fnmatch(field, flat_object_pattern):
                        f[field] = events[field]

            # add columns of physics objects
            # columns are padded to given pad_length
            for physics_pattern, pad_length in physics_objects.items():
                physics_objects = law.util.brace_expand(physics_pattern)
                for physics_object in physics_objects:
                    physics_route = Route(physics_object)
                    parts = physics_route.fields
                    physics_array = physics_route.apply(events)
                    physics_array = ak.pad_none(physics_array, pad_length)
                    physics_array = ak.fill_none(physics_array, EMPTY_FLOAT)
                    for pad_number in range(0, pad_length):
                        full_name = f"{parts[0].lower()}{pad_number}_{parts[1].lower()}"
                        f[full_name] = physics_array[:, pad_number]
            return ak.to_dataframe(f)

        # prepare inputs and outputs
        inputs = self.input()

        # create a temp dir for saving intermediate files
        tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
        tmp_dir.touch()

        # define columns that will be written
        write_columns: set[Route] = set()
        skip_columns: set[str] = set()
        for c in self.config_inst.x.keep_columns.get(self.task_family, ["*"]):
            for r in self._expand_keep_column(c):
                if r.has_tag("skip"):
                    skip_columns.add(r.column)
                else:
                    write_columns.add(r)
        write_columns = {
            r for r in write_columns
            if not law.util.multi_match(r.column, skip_columns, mode=any)
        }
        route_filter = RouteFilter(write_columns)

        # define columns that need to be read
        read_columns = write_columns | set(mandatory_coffea_columns)
        read_columns = {Route(c) for c in read_columns}

        # iterate over chunks of events and diffs
        files = [inputs["events"]["events"].abspath]
        if self.producer_insts:
            files.extend([inp["columns"].abspath for inp in inputs["producers"]])
        if self.ml_model_insts:
            files.extend([inp["mlcolumns"].abspath for inp in inputs["ml"]])

        pandas_frameworks = []
        for (events, *columns), pos in self.iter_chunked_io(
            files,
            source_type=len(files) * ["awkward_parquet"],
            read_columns=len(files) * [read_columns],
        ):
            # optional check for overlapping inputs
            if self.check_overlapping_inputs:
                self.raise_if_overlapping([events] + list(columns))

            # add additional columns
            events = update_ak_array(events, *columns)

            # remove columns
            events = route_filter(events)

            # optional check for finite values
            if self.check_finite_output:
                self.raise_if_not_finite(events)

            # construct first, second lepton columns
            events = lepton_selection(
                events,
                attributes=["pt", "eta", "phi", "mass", "charge"],
            )

            # convert to pandas dataframe
            keep_columns, mapping = sync_columns()

            events_pd = awkward_to_pandas(
                events=events,
                physics_objects=keep_columns["physics_objects"],
                flat_objects=keep_columns["flat_objects"],
            )

            pandas_frameworks.append(events_pd)

        # merge output files
        merged_pandas_framework = pd.concat(
            pandas_frameworks,
            ignore_index=True,
        ).rename(columns=mapping, inplace=False)
        self.output().dump(merged_pandas_framework, index=False, formatter="pandas")


# overwrite class defaults
check_finite_tasks = law.config.get_expanded("analysis", "check_finite_output", [], split_csv=True)
CreateSyncFile.check_finite_output = ChunkedIOMixin.check_finite_output.copy(
    default=CreateSyncFile.task_family in check_finite_tasks,
    add_default_to_description=True,
)

check_overlap_tasks = law.config.get_expanded("analysis", "check_overlapping_inputs", [], split_csv=True)
CreateSyncFile.check_overlapping_inputs = ChunkedIOMixin.check_overlapping_inputs.copy(
    default=CreateSyncFile.task_family in check_overlap_tasks,
    add_default_to_description=True,
)


CreateSyncFileWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=CreateSyncFile,
    enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"],
)
