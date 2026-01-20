# coding: utf-8

"""
Inference base models with common functionality.
"""

from __future__ import annotations

import re
import abc
import functools
import dataclasses

import law
import order as od

from columnflow.inference import InferenceModel, FlowStrategy
from columnflow.config_util import get_datasets_from_process
from columnflow.util import DotDict


logger = law.logger.get_logger(__name__)

get_all_datasets_from_process = functools.partial(
    get_datasets_from_process,
    strategy="all",
    only_first=False,
)


class HBTInferenceModelBase(InferenceModel):
    """
    Base class for statistical models with support for a single and multiple configs. In the latter case, each set of
    processes is created per config and will thus have different names, resulting in a "stacking" of histograms.

    This class provides the general structure for building models but is otherwise agnostic of the actual analysis
    details. For the concrete analysis specific base class, see :py:class:`HBTInferenceModel`.
    """

    def __init__(self, *args, **kwargs) -> None:
        # members that are set in init_objects
        self.single_config: bool
        self.campaign_keys: dict[od.Config, str] = {}
        self.campaign_key: str
        self.proc_map: dict[str, dict[od.Config, str]] = {}

        super().__init__(*args, **kwargs)

    def init_func(self) -> None:
        # the default initialization is split into logical parts
        self.init_objects()
        self.init_categories()
        self.init_processes()
        self.init_parameters()
        self.init_cleanup()

    def init_objects(self) -> None:
        # gather campaign identifier keys per config
        self.single_config = len(self.config_insts) == 1
        for config_inst in self.config_insts:
            assert config_inst.campaign.x.year in {2022, 2023}
            self.campaign_keys[config_inst] = f"{config_inst.campaign.x.year}{config_inst.campaign.x.postfix}"

        # overall campaign key
        self.campaign_key = "_".join(self.campaign_keys.values())

        # setup the process_map
        self.init_proc_map()

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

    @abc.abstractmethod
    def init_cleanup(self) -> None:
        # should cleanup the inference model after initialization
        ...

    @abc.abstractmethod
    def inject_era(self, config_inst: od.Config, combine_name: str) -> str:
        # should inject era info into combine process names
        ...

    def fill_proc_map(self, name_map: dict[str, str]) -> None:
        # (same process name for all configs for now)
        for combine_name, proc_name in name_map.items():
            # same process name for all configs for now
            for config_inst in self.config_insts:
                _combine_name = self.inject_era(config_inst, combine_name)
                self.proc_map.setdefault(_combine_name, {})[config_inst] = proc_name


class HBTInferenceModel(HBTInferenceModelBase):
    """
    Base class for statistical models for the HH -> bbtautau analysis with opinionated content, structure and defaults.
    """

    # whether to fake data, by default from the sum of background processes
    fake_data = True

    # whether to add QCD to the process map, leading to dynamic modeling of QCD in the categories
    add_qcd = True

    # process tag that identifies signal processes
    signal_process_tag = "signal"

    # filling of bins with yield <= 0 (setting this to 0 disables empty bin filling)
    empty_bin_value = 1.0e-5

    # threshold for autoMCStats
    auto_mc_stats_threshold = 10.0

    @dataclasses.dataclass
    class CategoryInfo:
        combine_category: str
        config_category: str
        config_variable: str
        config_data_datasets: list[str]

    def create_category_combinations(self) -> list[DotDict[str, str]]:
        # should return a list of dictionaries that will be passed as keyword arguments to get_category_variable and
        # create_category_info during init_categories to create datacard caregories
        ...

    @abc.abstractmethod
    def create_category_info(self, **kwargs) -> CategoryInfo:
        # should return category info objects given keyword arguments with fields from category_combinations
        ...

    def init_categories(self) -> None:
        combis = self.create_category_combinations()
        if not combis:
            raise RuntimeError(f"no category combinations defined for inference model '{self.cls_name}'")

        for combi in combis:
            info = self.create_category_info(**combi)

            # gather fake processes to model data when needed
            fake_processes = []
            if self.fake_data:
                fake_processes = list(set.union(*(
                    {
                        combine_name
                        for config_inst, proc_name in proc_map.items()
                        if not config_inst.get_process(proc_name).has_tag(self.signal_process_tag)
                    }
                    for combine_name, proc_map in self.proc_map.items()
                )))

            # add the category
            self.add_category(
                info.combine_category,
                config_data={
                    config_inst.name: self.category_config_spec(
                        category=info.config_category,
                        variable=info.config_variable,
                        data_datasets=info.config_data_datasets,
                    )
                    for config_inst in self.config_insts
                },
                data_from_processes=fake_processes,
                mc_stats=self.auto_mc_stats_threshold,
                empty_bin_value=self.empty_bin_value,
                flow_strategy=FlowStrategy.move,
            )

    def init_processes(self) -> None:
        # while setting up processes, remember which ones have at least one dataset with LHE weights
        self.processes_with_lhe_weights = set()

        # loop through process map and add process objects
        for combine_name, proc_map in self.proc_map.items():
            for config_inst, proc_name in proc_map.items():
                proc_inst = config_inst.get_process(proc_name)
                dataset_names = []

                # when the process is not dynmically built, collect datasets contributing to it
                is_dynamic = proc_name == "qcd"
                if not is_dynamic:
                    dataset_names = [dataset.name for dataset in get_all_datasets_from_process(config_inst, proc_name)]
                    if not dataset_names:
                        logger.debug(
                            f"skipping process {proc_name} in inference model {self.cls_name}, no matching datasets "
                            f"found in config {config_inst.name}",
                        )
                        continue

                # add the process
                self.add_process(
                    name=combine_name,
                    config_data={
                        config_inst.name: self.process_config_spec(
                            process=proc_name,
                            mc_datasets=dataset_names,
                        ),
                    },
                    is_signal=proc_inst.has_tag(self.signal_process_tag),
                    is_dynamic=is_dynamic,
                )

                # store whether there is at least one dataset contributing to this process with lhe weights
                if not all(config_inst.get_dataset(d).has_tag("no_lhe_weights") for d in dataset_names):
                    self.processes_with_lhe_weights.add(combine_name)

    def init_cleanup(self) -> None:
        self.cleanup(keep_parameters="THU_HH")

    def inject_era(self, config_inst: od.Config, combine_name: str) -> str:
        # helper to inject era info into combine process names
        campaign_key = self.campaign_keys[config_inst]
        # for HH, inject the key before the ecm value
        if (m := re.match(r"^((ggHH|qqHH)_.+_13p(0|6)TeV)_(hbbhtt)$", combine_name)):
            return f"{m.group(1)}_{campaign_key}_{m.group(4)}"
        # for single H, inject the key before the higgs decay
        if (m := re.match(r"^(.+)_(hbb|htt)$", combine_name)):
            return f"{m.group(1)}_{campaign_key}_{m.group(2)}"
        # for all other processes, just append the campaign key
        return f"{combine_name}_{campaign_key}"

    def inject_all_eras(self, *names: str) -> list[str]:
        # helper to select processes across multiple configs
        gen = (
            {self.inject_era(config_inst, name) for config_inst in self.config_insts}
            for name in names
        )
        return list(set.union(*gen))

    def process_matches(
        self,
        *,
        processes: str | list[str] | None = None,
        configs: od.Config | list[od.Config] | None = None,
        skip_qcd: bool = False,
    ) -> list[str] | None:
        # helper to create process patterns to match specific rules
        patterns = []
        # build a single regexp that matches processes and configs
        name_parts = []
        if processes:
            name_parts.append(law.util.make_list(processes))
        if configs:
            name_parts.append([self.campaign_keys[c] for c in law.util.make_list(configs)])
        if name_parts:
            re_parts = [f"({'|'.join(n)})" for n in name_parts]
            patterns.append(rf"^.*{'.*'.join(re_parts)}.*$")
        if skip_qcd:
            patterns.append("!QCD*")
        return patterns or None
