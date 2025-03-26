# coding: utf-8

"""
Configuration of the HH → bb𝜏𝜏 analysis.
"""

from __future__ import annotations

import importlib

import order as od

from columnflow.util import DotDict

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

################################################################################################
# analysis-wide groups and defaults
################################################################################################

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
analysis_hbt.x.config_groups = {}

# named function hooks that can modify store_parts of task outputs if needed
analysis_hbt.x.store_parts_modifiers = {}

################################################################################################
# hist hooks
################################################################################################

analysis_hbt.x.hist_hooks = DotDict()

# simple blinding
from hbt.hist_hooks.blinding import add_hooks as add_blinding_hooks
add_blinding_hooks(analysis_hbt)

# qcd estimation
from hbt.hist_hooks.qcd import add_hooks as add_qcd_hooks
add_qcd_hooks(analysis_hbt)

# binning
from hbt.hist_hooks.binning import add_hooks as add_binning_hooks
add_binning_hooks(analysis_hbt)


#
# define configs
#

def add_lazy_config(
    *,
    campaign_module: str,
    campaign_attr: str,
    config_name: str,
    config_id: int,
    add_limited: bool = True,
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
    if add_limited:
        analysis_hbt.configs.add_lazy_factory(f"{config_name}_limited", create_factory(config_id + 200, "_limited", 2))


# 2022, preEE
# TODO: remove after move to v14
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2022_preEE_nano_uhh_v12",
    campaign_attr="campaign_run3_2022_preEE_nano_uhh_v12",
    config_name="22pre_v12",
    config_id=5012,
)
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2022_preEE_nano_uhh_v14",
    campaign_attr="campaign_run3_2022_preEE_nano_uhh_v14",
    config_name="22pre_v14",
    config_id=5014,
)

# 2022, postEE
# TODO: remove after move to v14
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2022_postEE_nano_uhh_v12",
    campaign_attr="campaign_run3_2022_postEE_nano_uhh_v12",
    config_name="22post_v12",
    config_id=6012,
)
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2022_postEE_nano_uhh_v14",
    campaign_attr="campaign_run3_2022_postEE_nano_uhh_v14",
    config_name="22post_v14",
    config_id=6014,
)

# 2023, preBPix
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2023_preBPix_nano_uhh_v14",
    campaign_attr="campaign_run3_2023_preBPix_nano_uhh_v14",
    config_name="23pre_v14",
    config_id=7014,
)

# 2023, postBPix
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2023_postBPix_nano_uhh_v14",
    campaign_attr="campaign_run3_2023_postBPix_nano_uhh_v14",
    config_name="23post_v14",
    config_id=8014,
)

#
# sync configs
#

# 2022, preEE
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2022_preEE_nano_v12",
    campaign_attr="campaign_run3_2022_preEE_nano_v12",
    config_name="22pre_v12_sync",
    config_id=5112,
    add_limited=False,
    sync_mode=True,
)

# 2022, postEE
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2022_postEE_nano_v12",
    campaign_attr="campaign_run3_2022_postEE_nano_v12",
    config_name="22post_v12_sync",
    config_id=6112,
    add_limited=False,
    sync_mode=True,
)

# 2023, preBPix
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2023_preBPix_nano_v13",
    campaign_attr="campaign_run3_2023_preBPix_nano_v13",
    config_name="23pre_v13_sync",
    config_id=7113,
    add_limited=False,
    sync_mode=True,
)

# 2023, postBPix
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2023_postBPix_nano_v13",
    campaign_attr="campaign_run3_2023_postBPix_nano_v13",
    config_name="23post_v13_sync",
    config_id=8113,
    add_limited=False,
    sync_mode=True,
)

# 2022, preEE, v14
add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2022_postEE_nano_uhh_v14",
    campaign_attr="campaign_run3_2022_postEE_nano_uhh_v14",
    config_name="22post_v14_sync",
    config_id=6114,
    add_limited=False,
    sync_mode=True,
)

add_lazy_config(
    campaign_module="cmsdb.campaigns.run3_2022_preEE_nano_uhh_v14",
    campaign_attr="campaign_run3_2022_preEE_nano_uhh_v14",
    config_name="22pre_v14_sync",
    config_id=5114,
    add_limited=False,
    sync_mode=True,
)
