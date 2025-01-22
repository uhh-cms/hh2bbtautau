# coding: utf-8

import law

from hbt.columnflow_patches import patch_all


law.contrib.load("pandas")

# apply cf patches once
patch_all()
