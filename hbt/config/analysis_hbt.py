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
# (empty since we use the lookup from the law.cfg instead)
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
    # "$CF_BASE/sandboxes/cmssw_default.sh",
]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
analysis_hbt.x.config_groups = {}

# named function hooks that can modify store_parts of task outputs if needed
analysis_hbt.x.store_parts_modifiers = {}


#
# load configs
#

#
# Run 2 configs
#

from hbt.config.configs_run2ul import add_config as add_config_run2ul

# 2016 HIPM
from cmsdb.campaigns.run2_2016_HIPM_nano_uhh_v12 import campaign_run2_2016_HIPM_nano_uhh_v12

# default v12 config
add_config_run2ul(
    analysis_hbt,
    campaign_run2_2016_HIPM_nano_uhh_v12.copy(),
    config_name=campaign_run2_2016_HIPM_nano_uhh_v12.name,
    config_id=6,  # random number here that is not repeated ?
)

# default v12 config with limited number of files for faster prototyping
add_config_run2ul(
    analysis_hbt,
    campaign_run2_2016_HIPM_nano_uhh_v12.copy(),
    config_name=f"{campaign_run2_2016_HIPM_nano_uhh_v12.name}_limited",
    config_id=16,  # random number here that is not repeated ?
    limit_dataset_files=2,
)

# 2016 post
from cmsdb.campaigns.run2_2016_nano_uhh_v12 import campaign_run2_2016_nano_uhh_v12

# v12 uhh config with full datasets
add_config_run2ul(
    analysis_hbt,
    campaign_run2_2016_nano_uhh_v12.copy(),
    config_name=campaign_run2_2016_nano_uhh_v12.name,
    config_id=3,
)

# v12 uhh config with limited number of files for faster prototyping
add_config_run2ul(
    analysis_hbt,
    campaign_run2_2016_nano_uhh_v12.copy(),
    config_name=f"{campaign_run2_2016_nano_uhh_v12.name}_limited",
    config_id=13,
    limit_dataset_files=2,
)

# 2017
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


#
# Run 3 configs
#

# 2022
from hbt.config.configs_run3 import add_config as add_config_run3
from cmsdb.campaigns.run3_2022_preEE_nano_uhh_v12 import campaign_run3_2022_preEE_nano_uhh_v12

# preEE v12 config
add_config_run3(
    analysis_hbt,
    campaign_run3_2022_preEE_nano_uhh_v12.copy(),
    config_name="run3_2022_preEE_limited",
    config_id=4,
    limit_dataset_files=2,
)

# preEE v12 config
add_config_run3(
    analysis_hbt,
    campaign_run3_2022_preEE_nano_uhh_v12.copy(),
    config_name="run3_2022_preEE",
    config_id=14,
)
