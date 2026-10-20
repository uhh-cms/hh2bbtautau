# coding: utf-8

"""
Inference models with the in-situ QCD estimation.

Instead of a single, pre-computed QCD template, the signal region category of the datacard receives additional
processes that are filled by the "qcd_in_situ" hist hook (see hbt/hist_hooks/qcd_in_situ.py) from the non-isolated
control region B, scaled by the transfer factor TF from the same-sign regions C/D:

  - QCD_data_<era>_QCDCR  : TF * data_B
  - <bkg>_<era>_QCDCR     : TF * bkg_B, for each background process <bkg>

The "_QCDCR" marker is placed directly after the era, so that process names keep their trailing decay channel, e.g.
"ttH_13p6TeV_2023BPix_QCDCR_hbb" (single higgs processes must end in "_hbb" or "_htt").

A frozen rate parameter "QCD_scale" of -1 is attached to the background copies, so that the sum of all these processes
corresponds to TF * (data_B - sum(bkg_B)) in the fit. All nuisance parameters acting on a background process also act
on its control region copy, which correlates the background subtraction with the background modeling in the signal
region. The statistical uncertainty of TF is a rate (lnN) nuisance per era and category, derived from the
integral effect of the "qcd_tf" shift templates written by the hook.
"""

from __future__ import annotations

import re
import copy

import law
import order as od

from columnflow.inference import ParameterType, ParameterTransformation
from columnflow.util import DotDict
from columnflow.types import Any

from hbt.inference.default import default, remove_shift_parameters


logger = law.logger.get_logger(__name__)


class default_qcd_in_situ(default):
    """
    Default statistical model for the HH -> bbtautau analysis with the in-situ QCD estimation.
    """

    # no pre-computed qcd template
    add_qcd = False

    # whether to add the control region processes for the in-situ qcd estimation
    add_qcd_in_situ = True

    # marker of combine process names of control region copies, inserted directly after the era
    qcd_cr_suffix = "_QCDCR"

    # combine process name (before era injection and suffix) of the data in the control region
    qcd_cr_data_name = "QCD_data"

    # name of the shift source carrying the statistical uncertainty of the transfer factor
    qcd_tf_shift_source = "qcd_tf"

    @property
    def qcd_cr_pattern(self) -> str:
        # regex matching all control region copies (data and backgrounds) across eras
        return rf"^.+{self.qcd_cr_suffix}(_.+)?$"

    @property
    def qcd_cr_data_pattern(self) -> str:
        # pattern matching the data process in the control region across eras
        return f"{self.qcd_cr_data_name}_*{self.qcd_cr_suffix}"

    def is_qcd_cr_name(self, combine_name: str) -> bool:
        # whether a (era injected) combine process name refers to a control region copy
        return self.qcd_cr_suffix in combine_name

    def to_qcd_cr_name(self, combine_name: str, config_inst: od.Config | None = None) -> str | None:
        """
        Returns the name of the control region copy of an era injected *combine_name*, i.e., with the marker inserted
        directly after the era, or *None* if no era is found. When *config_inst* is given, only its era is considered.
        """
        config_insts = [config_inst] if config_inst else self.config_insts
        for _config_inst in config_insts:
            key = self.campaign_keys[_config_inst]
            cr_name, n = re.subn(rf"_{re.escape(key)}(?=_|$)", f"_{key}{self.qcd_cr_suffix}", combine_name, count=1)
            if n:
                return cr_name
        return None

    def inject_era(self, config_inst: od.Config, combine_name: str) -> str:
        # for control region copies, inject the era into the base name and insert the marker after it
        if combine_name.endswith(self.qcd_cr_suffix):
            base_name = combine_name[:-len(self.qcd_cr_suffix)]
            cr_name = self.to_qcd_cr_name(super().inject_era(config_inst, base_name), config_inst=config_inst)
            if cr_name is None:
                raise ValueError(f"could not inject era into control region process name '{combine_name}'")
            return cr_name
        return super().inject_era(config_inst, combine_name)

    def create_proc_name_map(self) -> dict[str, str]:
        proc_name_map = super().create_proc_name_map()
        if not self.add_qcd_in_situ:
            return proc_name_map

        # helper to check that a process exists in all configs
        def check_process(proc_name: str, reason: str) -> None:
            for config_inst in self.config_insts:
                if not config_inst.get_process(proc_name, default=None):
                    raise ValueError(
                        f"process '{proc_name}' required for {reason} in inference model '{self.cls_name}' not found "
                        f"in config '{config_inst.name}'",
                    )

        # data in the control region
        check_process("qcd_cr_data", "the in-situ qcd estimation")
        proc_name_map[f"{self.qcd_cr_data_name}{self.qcd_cr_suffix}"] = "qcd_cr_data"

        # control region copies of all background processes
        for combine_name, proc_name in list(proc_name_map.items()):
            proc_inst = self.config_insts[0].get_process(proc_name)
            if proc_inst.has_tag(self.signal_process_tag) or proc_inst.has_tag("qcd_cr"):
                continue
            cr_proc_name = f"qcd_cr_{proc_name}"
            check_process(cr_proc_name, f"the control region copy of '{proc_name}'")
            proc_name_map[f"{combine_name}{self.qcd_cr_suffix}"] = cr_proc_name

        return proc_name_map

    def add_parameter(
        self,
        *args,
        qcd_cr_data: bool = False,
        **kwargs,
    ) -> DotDict:
        """
        Same as the base method, but with two additions for the in-situ qcd estimation:

          - after the parameter was added, every process carrying it also passes it on to its control region copy in
            the same category, so that a parameter acting on a background process also acts on the copy (independent
            of whether the process patterns match the copy's name)
          - unless *qcd_cr_data* is *True*, rate parameters are removed from the control region data process; shape
            parameters are kept since the transfer factor applied to the data template depends on the shifted mc in
            the C and D regions (see :py:meth:`sync_qcd_cr_data_parameters` for shape parameters that were not matched)
        """
        parameter = super().add_parameter(*args, **kwargs)

        if not self.add_qcd_in_situ:
            return parameter

        # pass the parameter on to control region copies
        for cat_obj in self.categories:
            proc_names = {proc_obj.name for proc_obj in cat_obj.processes}
            for proc_obj in list(cat_obj.processes):
                if self.is_qcd_cr_name(proc_obj.name):
                    continue
                cr_name = self.to_qcd_cr_name(proc_obj.name)
                if cr_name is None or cr_name not in proc_names:
                    continue
                param = self.get_parameter(parameter.name, process=proc_obj.name, category=cat_obj.name, silent=True)
                if param is None or self.has_parameter(parameter.name, process=cr_name, category=cat_obj.name):
                    continue
                cr_proc_obj = self.get_process(cr_name, category=cat_obj.name)
                cr_proc_obj.parameters.append(copy.deepcopy(param))

        # remove rate parameters from the data copy
        if not qcd_cr_data and parameter.type.is_rate:
            self.remove_parameter(parameter.name, process=self.qcd_cr_data_pattern)

        return parameter

    def sync_qcd_cr_data_parameters(self) -> None:
        """
        Attaches every shape parameter acting on a control region background copy also to the control region data
        process of the same category and era, as the hist hook fills the data template for all mc shifts with the
        corresponding transfer factor.
        """
        for cat_obj in self.categories:
            for config_inst in self.config_insts:
                key = self.campaign_keys[config_inst]
                data_name = self.inject_era(config_inst, f"{self.qcd_cr_data_name}{self.qcd_cr_suffix}")
                data_proc = self.get_process(data_name, category=cat_obj.name, silent=True)
                if not data_proc:
                    continue
                existing = {param.name for param in data_proc.parameters}
                for proc_obj in cat_obj.processes:
                    if proc_obj is data_proc or not self.is_qcd_cr_name(proc_obj.name):
                        continue
                    # only copies of the same era
                    if f"_{key}{self.qcd_cr_suffix}" not in proc_obj.name:
                        continue
                    for param in proc_obj.parameters:
                        if param.type.is_shape and param.name not in existing:
                            data_proc.parameters.append(copy.deepcopy(param))
                            existing.add(param.name)

    def init_parameters(self) -> None:
        super().init_parameters()
        if not self.add_qcd_in_situ:
            return

        self.add_parameter_group("qcd_in_situ")

        # frozen factor of -1 for background copies in the control region, implementing the subtraction
        self.add_parameter(
            "QCD_scale",
            type=ParameterType.rate_unconstrained,
            process=[self.qcd_cr_pattern, f"!{self.qcd_cr_data_pattern}"],
            process_match_mode=all,
            effect="-1 [-1,-1]",
            group=["qcd_in_situ"],
        )

        # statistical uncertainty of the transfer factor, decorrelated across eras and categories; the up/down
        # templates written by the hook only differ in normalization, so the effect is converted to a rate (lnN)
        for combi in self.create_category_combinations():
            info = self.create_category_info(**combi)
            for config_inst in self.config_insts:
                key = self.campaign_keys[config_inst]
                self.add_parameter(
                    f"CMS_bbtt_QCD_TF_{key}_{'_'.join(combi.values())}",
                    type=ParameterType.rate_gauss,
                    transformations=[ParameterTransformation.effect_from_shape],
                    config_data={
                        config_inst.name: self.parameter_config_spec(shift_source=self.qcd_tf_shift_source),
                    },
                    category=[info.combine_category],
                    process=[rf"^.+_{key}{self.qcd_cr_suffix}(_.+)?$"],
                    qcd_cr_data=True,
                    group=["experiment", "rate_nuisances", "qcd_in_situ"],
                )

        # shape parameters of background copies also act on the data copy through the transfer factor
        self.sync_qcd_cr_data_parameters()


@default_qcd_in_situ.inference_model
def default_qcd_in_situ_no_shifts(self) -> None:
    """
    Variant without mc shape nuisances, keeping the statistical uncertainty of the transfer factor.
    """
    super(default_qcd_in_situ_no_shifts, self).init_func()
    # (the transfer factor uncertainty is a rate parameter and therefore not removed)
    remove_shift_parameters(self)
    self.init_cleanup()
