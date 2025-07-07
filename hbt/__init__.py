# coding: utf-8

from __future__ import annotations

__all__ = ["env_is_desy", "env_is_cern"]

import os
import socket

import law

from hbt.columnflow_patches import patch_all


law.contrib.load("pandas")

# apply cf patches once
patch_all()

#: Boolean denoting whether the environment is on DESY resources.
_hostname = socket.gethostname()
env_is_desy = _hostname.endswith(".desy.de")

#: Boolean denoting whether the environment is on CERN resources.
env_is_cern = _hostname.endswith(".cern.ch")

#: Boolean denoting whether to keep using DESY resources when when the env is different.
force_desy_resources = law.util.flag_to_bool(os.getenv("HBT_FORCE_DESY", "0"))
