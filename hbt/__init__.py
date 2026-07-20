# coding: utf-8

from __future__ import annotations

__all__ = ["env_is_desy", "env_is_cern"]

import os
import socket

import law
import order as od

from hbt.columnflow_patches import patch_all


law.contrib.load("pandas")


# the analysis is not intended to run in optimized mode, so that assert's remain active
if not __debug__:
    raise RuntimeError("analysis should not run in optimized mode")


#: Boolean denoting whether the environment is on DESY resources.
_hostname = socket.gethostname()
env_is_desy = _hostname.endswith(".desy.de")

#: Boolean denoting whether the environment is on CERN resources.
env_is_cern = _hostname.endswith(".cern.ch")

#: Boolean denoting whether to keep using DESY resources when when the env is different.
force_desy_resources = law.util.flag_to_bool(os.getenv("HBT_FORCE_DESY", "0"))

# apply cf patches once
patch_all()


def get_config(name_or_id: str | int) -> od.Config:
    """
    Helper to load a config from the analysis by *name_or_id* and return it.
    """
    from hbt.config.analysis_hbt import analysis_hbt
    return analysis_hbt.get_config(name_or_id)
