# coding: utf-8

"""
Inference base models with common functionality.
"""

import re
import abc

import order as od

from columnflow.inference import InferenceModel


class HBTInferenceModelBase(InferenceModel):
    """
    Base class for statistical models with support for a single or and multiple configs. In the latter case, each set of
    processes is created per config and will thus have different names, resulting in a "stacking" of histograms.
    """

    def __init__(self, *args, **kwargs) -> None:
        # members that are set in init_objects
        self.single_config: bool
        self.campaign_keys: dict[od.Config, str] = {}
        self.campaign_key: str
        self.proc_map: dict[str, dict[od.Config, str]] = {}

        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def init_proc_map(self) -> None:
        # should setup self.proc_map
        ...

    @abc.abstractmethod
    def init_categories(self) -> None:
        # should setup inference model cateogries
        ...

    @abc.abstractmethod
    def init_processes(self) -> None:
        # should setup inference model processes
        ...

    @abc.abstractmethod
    def init_parameters(self) -> None:
        # should setup inference model parameters
        ...

    def init_func(self) -> None:
        # the default initialization is split into logical parts
        self.init_objects()
        self.init_categories()
        self.init_processes()
        self.init_parameters()
        self.init_cleanup()

    def inject_era(self, config_inst: od.Config, combine_name: str) -> str:
        # helper to inject era info into combine process names
        campaign_key = self.campaign_keys[config_inst]
        # for HH, inject the key before the ecm value
        if (m := re.match(r"^((ggHH|qqHH)_.+)_(13p(0|6)TeV_hbbhtt)$", combine_name)):
            return f"{m.group(1)}_{campaign_key}_{m.group(3)}"
        # for single H, inject the key before the higgs decay
        if (m := re.match(r"^(.+)_(hbb|htt)$", combine_name)):
            return f"{m.group(1)}_{campaign_key}_{m.group(2)}"
        # for all other processes, just append the campaign key
        return f"{combine_name}_{campaign_key}"

    def init_objects(self) -> None:
        # gather campaign identifier keys per config
        self.single_config = len(self.config_insts) == 1
        for config_inst in self.config_insts:
            year2 = config_inst.campaign.x.year % 100
            self.campaign_keys[config_inst] = f"{year2}{config_inst.campaign.x.postfix}"

        # overall campaign key
        self.campaign_key = "_".join(self.campaign_keys.values())

        # setup the process_map
        self.init_proc_map()

    def init_cleanup(self) -> None:
        self.cleanup(keep_parameters="THU_HH")
