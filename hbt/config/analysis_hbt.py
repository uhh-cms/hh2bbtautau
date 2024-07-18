# coding: utf-8

"""
Configuration of the HH → bb𝜏𝜏 analysis.
"""

from __future__ import annotations

import importlib

import order as od

from hbt.config.configs_hbt import add_config


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
# define configs
#

def add_lazy_config(
    campaign_module: str,
    campaign_attr: str,
    config_name: str,
    config_id: int,
    **kwargs,
):
    def create_factory(
        config_id: int,
        config_name_postfix: str = "",
        limit_dataset_files: int | None = None,
    ):
        def factory(configs: od.UniqueObjectIndex):
            # import the campaign
            mod = importlib.import_module(campaign_module)
            campaign = getattr(mod, campaign_attr)

            return add_config(
                analysis_hbt,
                campaign.copy(),
                config_name=config_name + config_name_postfix,
                config_id=config_id,
                **kwargs,
            )
        return factory

    analysis_hbt.configs.add_lazy_factory(config_name, create_factory(config_id))
    analysis_hbt.configs.add_lazy_factory(f"{config_name}_limited", create_factory(config_id + 200, "_limited", 2))


#
# Run 2 configs
#

# 2016 HIPM (also known as APV or preVFP)
add_lazy_config(
    campaign_module="cmsdb.campaigns.run2_2016_HIPM_nano_uhh_v12",
    campaign_attr="campaign_run2_2016_HIPM_nano_uhh_v12",
    config_name="run2_2016_HIPM_nano_uhh_v12",
    config_id=1,
)

# 2016 (also known postVFP)
add_lazy_config(
    campaign_module="cmsdb.campaigns.run2_2016_nano_uhh_v12",
    campaign_attr="campaign_run2_2016_nano_uhh_v12",
    config_name="run2_2016_nano_uhh_v12",
    config_id=2,
)

# 2017
add_lazy_config(
    campaign_module="cmsdb.campaigns.run2_2017_nano_uhh_v11",
    campaign_attr="campaign_run2_2017_nano_uhh_v11",
    config_name="run2_2017_nano_uhh_v11",
    config_id=3,
)

#
# Run 3 configs
#

# 2022, preEE
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2022_preEE_nano_uhh_v12",
    campaign_attr="campaign_run3_2022_preEE_nano_uhh_v12",
    config_name="run3_2022_preEE",
    config_id=5,
)
