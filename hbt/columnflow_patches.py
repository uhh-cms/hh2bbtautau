# coding: utf-8

"""
Collection of patches of underlying columnflow tasks.
"""

import os

import law
from columnflow.util import memoize, dev_sandbox


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
def patch_create_pileup_weights_sandbox():
    """
    Patches the sandox attribute of the existing CreatePileupWeights task to use a different one.
    """
    from columnflow.tasks.cms.external import CreatePileupWeights

    CreatePileupWeights.sandbox = dev_sandbox("bash::$HBT_BASE/sandboxes/cmssw_default.sh")

    logger.debug(f"patched sandbox of {CreatePileupWeights.task_family}")


@memoize
def patch_all():
    patch_bundle_repo_exclude_files()
    patch_create_pileup_weights_sandbox()
