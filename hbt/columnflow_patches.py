# coding: utf-8

"""
Collection of patches of underlying columnflow tasks.
"""

import os
import getpass

import law
from columnflow.util import memoize


logger = law.logger.get_logger(__name__)


@memoize
def patch_bundle_repo_exclude_files():
    """
    Patches the exclude_files attribute of the existing BundleRepo task to exclude files
    specific to _this_ analysis project.
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
    Patches the HTCondorWorkflow and SlurmWorkflow tasks to change the default value of the
    poll_interval parameter to 1 minute.
    """
    from columnflow.tasks.framework.remote import HTCondorWorkflow, SlurmWorkflow

    HTCondorWorkflow.poll_interval._default = 1.0  # minutes
    SlurmWorkflow.poll_interval._default = 1.0  # minutes

    logger.debug(
        f"patched poll_interval._default of {HTCondorWorkflow.task_family} and "
        f"{SlurmWorkflow.task_family}",
    )


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

    logger.debug(f"patched htcondor_job_resources of {HTCondorWorkflow.task_family}")


@memoize
def patch_all():
    patch_bundle_repo_exclude_files()
    patch_remote_workflow_poll_interval()
    patch_htcondor_workflow_naf_resources()
