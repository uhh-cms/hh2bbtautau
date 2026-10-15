# coding: utf-8

"""
Inference models with the in-situ QCD estimation.

Instead of a single, pre-computed QCD template, the signal region category of the datacard receives additional
processes that are filled by the "qcd_in_situ" hist hook (see hbt/hist_hooks/qcd_in_situ.py) from the non-isolated
control region B, scaled by the transfer factor TF from the same-sign regions C/D:

  - QCD_data_<era>_QCDCR  : TF * data_B
  - <bkg>_<era>_QCDCR     : TF * bkg_B, for each background process <bkg>

A frozen rate parameter "QCD_scale" of -1 is attached to the background copies, so that the sum of all these processes
corresponds to TF * (data_B - sum(bkg_B)) in the fit. All nuisance parameters acting on a background process also act
on its control region copy, which correlates the background subtraction with the background modeling in the signal
region. The statistical uncertainty of TF is a shape nuisance per era and category ("qcd_tf" shift source).
"""

from __future__ import annotations

import law
import order as od

from columnflow.inference import ParameterType
from columnflow.util import DotDict
from columnflow.types import Any

from hbt.inference.default import default_no_shifts


logger = law.logger.get_logger(__name__)


class default_qcd_in_situ(default_no_shifts):
    """
    Default statistical model for the HH -> bbtautau analysis with the in-situ QCD estimation.
    """

    # no pre-computed qcd template
    add_qcd = False

    # whether to add the control region processes for the in-situ qcd estimation
    add_qcd_in_situ = True

    # suffix of combine process names of control region copies
    qcd_cr_suffix = "_QCDCR"

    # combine process name (before era injection and suffix) of the data in the control region
    qcd_cr_data_name = "QCD_data"

    # name of the shift source carrying the statistical uncertainty of the transfer factor
    qcd_tf_shift_source = "qcd_tf"

    @property
    def qcd_cr_data_pattern(self) -> str:
        # pattern matching the data process in the control region across eras
        return f"{self.qcd_cr_data_name}_*{self.qcd_cr_suffix}"

    def inject_era(self, config_inst: od.Config, combine_name: str) -> str:
        # for control region copies, inject the era into the base name and keep the suffix at the end
        if combine_name.endswith(self.qcd_cr_suffix):
            base_name = combine_name[:-len(self.qcd_cr_suffix)]
            return super().inject_era(config_inst, base_name) + self.qcd_cr_suffix
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
        process: str | list[str] | None = None,
        process_match_mode: Any = any,
        qcd_cr_data: bool = False,
        **kwargs,
    ) -> DotDict:
        """
        Same as the base method, but with two additions for the in-situ qcd estimation:

          - plain (non-regex, non-negated) process patterns are extended by their control region counterparts, so that
            a parameter acting on a background process also acts on its copy in the control region
          - unless *qcd_cr_data* is *True*, the parameter is removed from the control region data process which is not
            subject to any modeling uncertainty
        """
        if self.add_qcd_in_situ and process is not None and process_match_mode is any:
            patterns = law.util.make_list(process)
            # (same regex detection as in columnflow.util.pattern_matcher)
            is_regex = lambda pattern: "^" in pattern and "$" in pattern
            process = patterns + [
                f"{pattern}{self.qcd_cr_suffix}"
                for pattern in patterns
                if not (pattern.startswith("!") or is_regex(pattern))
            ]

        parameter = super().add_parameter(*args, process=process, process_match_mode=process_match_mode, **kwargs)

        if self.add_qcd_in_situ and not qcd_cr_data:
            self.remove_parameter(parameter.name, process=self.qcd_cr_data_pattern)

        return parameter

    def init_parameters(self) -> None:
        super().init_parameters()
        if not self.add_qcd_in_situ:
            return

        self.add_parameter_group("qcd_in_situ")

        # frozen factor of -1 for background copies in the control region, implementing the subtraction
        self.add_parameter(
            "QCD_scale",
            type=ParameterType.rate_unconstrained,
            process=[f"*{self.qcd_cr_suffix}", f"!{self.qcd_cr_data_pattern}"],
            process_match_mode=all,
            effect="-1 [-1,-1]",
            group=["qcd_in_situ"],
        )

        # statistical uncertainty of the transfer factor, decorrelated across eras and categories
        for combi in self.create_category_combinations():
            info = self.create_category_info(**combi)
            for config_inst in self.config_insts:
                key = self.campaign_keys[config_inst]
                self.add_parameter(
                    f"CMS_bbtt_QCD_TF_{key}_{'_'.join(combi.values())}",
                    type=ParameterType.shape,
                    config_data={
                        config_inst.name: self.parameter_config_spec(shift_source=self.qcd_tf_shift_source),
                    },
                    category=[info.combine_category],
                    process=[rf"^.+_{key}(_.+)?{self.qcd_cr_suffix}$"],
                    qcd_cr_data=True,
                    group=["experiment", "shape_nuisances", "qcd_in_situ"],
                )
