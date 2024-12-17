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
                limit_dataset_files=limit_dataset_files,
                **kwargs,
            )
        return factory

    analysis_hbt.configs.add_lazy_factory(config_name, create_factory(config_id))
    analysis_hbt.configs.add_lazy_factory(f"{config_name}_limited", create_factory(config_id + 200, "_limited", 2))


#
# Run 2 configs
#

# 2016 HIPM (also known as APV or preVFP), TODO: campaign needs consistency and content check
# add_lazy_config(
#     campaign_module="cmsdb.campaigns.run2_2016_HIPM_nano_uhh_v12",
#     campaign_attr="campaign_run2_2016_HIPM_nano_uhh_v12",
#     config_name="run2_2016_HIPM_nano_uhh_v12",
#     config_id=1,
# )

# 2016 (also known postVFP), TODO: campaign needs consistency and content check
# add_lazy_config(
#     campaign_module="cmsdb.campaigns.run2_2016_nano_uhh_v12",
#     campaign_attr="campaign_run2_2016_nano_uhh_v12",
#     config_name="run2_2016_nano_uhh_v12",
#     config_id=2,
# )

# 2017, old nano version, TODO: needs re-processing
# add_lazy_config(
#     campaign_module="cmsdb.campaigns.run2_2017_nano_uhh_v11",
#     campaign_attr="campaign_run2_2017_nano_uhh_v11",
#     config_name="run2_2017_nano_uhh_v11",
#     config_id=3,
# )

# 2018, TODO: not processed yet
# add_lazy_config(
#     campaign_module="cmsdb.campaigns.run2_2018_nano_uhh_v12",
#     campaign_attr="campaign_run2_2018_nano_uhh_v12",
#     config_name="run2_2018_nano_uhh_v12",
#     config_id=4,
# )

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

# 2022, postEE
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2022_postEE_nano_uhh_v12",
    campaign_attr="campaign_run3_2022_postEE_nano_uhh_v12",
    config_name="run3_2022_postEE",
    config_id=6,
)

# 2023, preBPix
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2023_preBPix_nano_uhh_v14",
    campaign_attr="campaign_run3_2023_preBPix_nano_uhh_v14",
    config_name="run3_2023_preBPix",
    config_id=7,
)

# 2023, postBPix
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2023_postBPix_nano_uhh_v14",
    campaign_attr="campaign_run3_2023_postBPix_nano_uhh_v14",
    config_name="run3_2023_postBPix",
    config_id=8,
)

#
# sync configs
#

# 2022, preEE
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2022_preEE_nano_v12",
    campaign_attr="campaign_run3_2022_preEE_nano_v12",
    config_name="run3_2022_preEE_sync",
    config_id=5005,
    sync_mode=True,
)

# 2022, postEE
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2022_postEE_nano_v12",
    campaign_attr="campaign_run3_2022_postEE_nano_v12",
    config_name="run3_2022_postEE_sync",
    config_id=5006,
    sync_mode=True,
)

# 2023, preBPix
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2023_preBPix_nano_v13",
    campaign_attr="campaign_run3_2023_preBPix_nano_v13",
    config_name="run3_2023_preBPix_sync",
    config_id=5007,
    sync_mode=True,
)

# 2023, postBPix
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2023_postBPix_nano_v13",
    campaign_attr="campaign_run3_2023_postBPix_nano_v13",
    config_name="run3_2023_postBPix_sync",
    config_id=5008,
    sync_mode=True,
)
