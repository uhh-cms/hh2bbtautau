# coding: utf-8

"""
Configuration of the HH → bb𝜏𝜏 analysis.
"""

import order as od


#
# the main analysis object
#

analysis_hbt = od.Analysis(
    name="analysis_hbt",
    id=1,
)

# analysis-global versions
analysis_hbt.x.versions = {}

# files of bash sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
analysis_hbt.x.bash_sandboxes = [
    "$CF_BASE/sandboxes/cf.sh",
    "$CF_BASE/sandboxes/venv_columnar.sh",
    "$HBT_BASE/sandboxes/venv_columnar_tf.sh",
]

# files of cmssw sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
analysis_hbt.x.cmssw_sandboxes = [
    "$CF_BASE/sandboxes/cmssw_default.sh",
]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
analysis_hbt.x.config_groups = {}


#
# load configs
#

# 2017
from hbt.config.configs_run2ul import add_config as add_config_run2ul
from cmsdb.campaigns.run2_2017_nano_v9 import campaign_run2_2017_nano_v9
from cmsdb.campaigns.run2_2017_nano_uhh_v11 import campaign_run2_2017_nano_uhh_v11


# default v9 config
add_config_run2ul(
    analysis_hbt,
    campaign_run2_2017_nano_v9.copy(),
    config_name=campaign_run2_2017_nano_v9.name,
    config_id=2,
)

# v9 config with limited number of files for faster prototyping
add_config_run2ul(
    analysis_hbt,
    campaign_run2_2017_nano_v9.copy(),
    config_name=f"{campaign_run2_2017_nano_v9.name}_limited",
    config_id=12,
    limit_dataset_files=2,
)

# default v11 uhh config
add_config_run2ul(
    analysis_hbt,
    campaign_run2_2017_nano_uhh_v11.copy(),
    config_name=campaign_run2_2017_nano_uhh_v11.name,
    config_id=31,
)

# v11 uhh config with limited number of files for faster prototyping
add_config_run2ul(
    analysis_hbt,
    campaign_run2_2017_nano_uhh_v11.copy(),
    config_name=f"{campaign_run2_2017_nano_uhh_v11.name}_limited",
    config_id=32,
    limit_dataset_files=2,
)
