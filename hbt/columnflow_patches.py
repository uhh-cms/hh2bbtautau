# coding: utf-8

"""
Collection of patches of underlying columnflow tasks.
"""

import os
import getpass

import luigi
import law
import order as od

from columnflow.util import memoize


logger = law.logger.get_logger(__name__)


@memoize
def patch_bundle_repo_exclude_files():
    """
    Patches the exclude_files attribute of the existing BundleRepo task to exclude files specific to _this_ analysis
    project.
    """
    from columnflow.tasks.framework.remote import BundleRepo

    # get the relative path to CF_BASE
    cf_rel = os.path.relpath(os.environ["CF_BASE"], os.environ["HBT_BASE"])

    # amend exclude files to start with the relative path to CF_BASE
    exclude_files = [os.path.join(cf_rel, path) for path in BundleRepo.exclude_files]

    # add additional files
    exclude_files.extend([
        "docs", "tests", "data", "assets", ".law", ".setups", ".data", ".github",
    ])

    # overwrite them
    BundleRepo.exclude_files[:] = exclude_files

    logger.debug(f"patched exclude_files of {BundleRepo.task_family}")


@memoize
def patch_remote_workflow_poll_interval():
    """
    Patches the HTCondorWorkflow and SlurmWorkflow tasks to change the default value of the poll_interval parameter to 1
    minute.
    """
    from columnflow.tasks.framework.remote import HTCondorWorkflow, SlurmWorkflow

    HTCondorWorkflow.poll_interval._default = 1.0  # minutes
    SlurmWorkflow.poll_interval._default = 1.0  # minutes

    logger.debug(f"patched poll_interval._default of {HTCondorWorkflow.task_family} and {SlurmWorkflow.task_family}")


@memoize
def patch_htcondor_workflow_naf_resources():
    """
    Patches the HTCondorWorkflow task to declare user-specific resources when running on the NAF.
    """
    from columnflow.tasks.framework.remote import HTCondorWorkflow

    def htcondor_job_resources(self, job_num, branches):
        # one "naf_<username>" resource per job, indendent of the number of branches in the job
        return {f"naf_{getpass.getuser()}": 1}

    HTCondorWorkflow.htcondor_job_resources = htcondor_job_resources

    # also disable the memory summary plot by default
    HTCondorWorkflow.show_memory_summary_hist = False

    logger.debug(f"patched htcondor_job_resources of {HTCondorWorkflow.task_family}")


@memoize
def patch_merge_reduction_stats_inputs():
    """
    Patches the MergeReductionStats task to set the default value of n_inputs to -1, so as to use all files to infer
    merging factors with full statistical precision.
    """
    from columnflow.tasks.reduction import MergeReductionStats

    MergeReductionStats.n_inputs._default = -1

    logger.debug(f"patched n_inputs default value of {MergeReductionStats.task_family}")


@memoize
def patch_unite_columns_keep_columns_key_default():
    """
    Patches the keep_columns_key parameter of the UniteColumns task to have a custum default value of "all".
    """
    from columnflow.tasks.union import UniteColumns

    UniteColumns.keep_columns_key = UniteColumns.keep_columns_key.copy(
        default="all",
        add_default_to_description=True,
    )

    logger.debug(f"patched keep_columns_key parameter of {UniteColumns.task_family}")


@memoize
def patch_unite_columns_events_filter():
    """
    Patches the UniteColumns task to use a custom event filter function to only keep events whose "keep_in_union" is
    true-ish.
    """
    from columnflow.tasks.union import UniteColumns

    UniteColumns.filter_events = lambda self, events: (
        (events.keep_in_union == 1)
        if "keep_in_union" in events.fields
        else (events.event >= 0)
    )

    logger.debug(f"patched filter_events method of {UniteColumns.task_family}")


@memoize
def patch_merge_shifted_histograms():
    """
    Patches the MergeShiftedHistograms task to add several analysis-specific performance improvements:

        - Add a parameter "--trigger-only" that causes its run method to complete without writing an output in order to
        only trigger the tasks requirement resolution.
        - Add a parameter "--filter-categories" that causes the early removal of unwanted categories to save memory. The
        store path of output targets is updated accordingly.
    """
    from columnflow.tasks.framework.mixins import CategoriesMixin
    from columnflow.tasks.histograms import MergeShiftedHistograms

    # add parameters
    MergeShiftedHistograms.trigger_only = luigi.BoolParameter(
        default=False,
        description="if set, the task will not write an output but only trigger its requirements; default: False",
    )
    MergeShiftedHistograms.filter_categories = law.CSVParameter(
        default=(),
        description="comma-separated category names or patterns of categories to keep; empty default",
        brace_expand=True,
        parse_empty=True,
    )

    # store original methods
    store_parts_orig = MergeShiftedHistograms.store_parts
    init_orig = MergeShiftedHistograms.__init__
    run_orig = MergeShiftedHistograms.run
    modify_input_hist_orig = MergeShiftedHistograms.modify_input_hist

    # define patched methods
    def init(self, *args, **kwargs):
        init_orig(self, *args, **kwargs)

        # resolve and sort filter categories
        if self.filter_categories:
            self.filter_categories = tuple(sorted(self.find_config_objects(
                names=self.filter_categories,
                container=self.config_inst,
                object_cls=od.Category,
                groups_str="category_groups",
                deep=True,
            )))

    def store_parts(self):
        parts = store_parts_orig(self)
        if self.filter_categories:
            categories_repr = CategoriesMixin._categories_repr(self.filter_categories)
            parts.insert_after("version", "filter_categories", f"categories_{categories_repr}")
        return parts

    def run(self):
        if self.trigger_only:
            self.logger.warning(f"{self.task_family} invoked with '--trigger-only', skipping actual run method")
            return

        return run_orig(self)

    def modify_input_hist(self, shift, variable, h):
        if not self.filter_categories:
            return modify_input_hist_orig(self, shift, variable, h)

        import hist
        h = h[{"category": [
            hist.loc(category) for category in self.filter_categories
            if category in h.axes["category"]
        ]}]

        return h

    # store patched methods
    MergeShiftedHistograms.__init__ = init
    MergeShiftedHistograms.run = run
    MergeShiftedHistograms.store_parts = store_parts
    MergeShiftedHistograms.modify_input_hist = modify_input_hist

    logger.debug(f"patched {MergeShiftedHistograms.task_family}")


@memoize
def patch_serialize_inference_model_base():
    """
    Patches the SerializeInferenceModelBase task to request MergeShiftedHistograms with category filtering applied.
    """
    from columnflow.tasks.framework.inference import SerializeInferenceModelBase

    hist_requirement_orig = SerializeInferenceModelBase._hist_requirement

    def _hist_requirement(self, **kwargs):
        # if there is at least one shift source required, only filter necessary leaf categories
        if kwargs.get("shift_sources") and set(kwargs["shift_sources"]) != {"nominal"}:
            config_inst = self.analysis_inst.get_config(kwargs["config"])
            categories = self.combined_config_data[config_inst]["categories"]
            leaf_categories = set.union(*(
                set(category_inst.get_leaf_categories() or [category_inst])
                for category_inst in (config_inst.get_category(c, deep=True) for c in categories)
            ))
            kwargs["filter_categories"] = tuple(sorted(cat_inst.name for cat_inst in leaf_categories))

        return hist_requirement_orig(self, **kwargs)

    SerializeInferenceModelBase._hist_requirement = _hist_requirement

    logger.debug(f"patched {SerializeInferenceModelBase.task_family}")


@memoize
def patch_all():
    patch_bundle_repo_exclude_files()
    patch_remote_workflow_poll_interval()
    patch_htcondor_workflow_naf_resources()
    patch_merge_reduction_stats_inputs()
    patch_unite_columns_keep_columns_key_default()
    patch_unite_columns_events_filter()
    patch_merge_shifted_histograms()
    patch_serialize_inference_model_base()
