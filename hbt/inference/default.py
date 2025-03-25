# coding: utf-8

"""
Default inference model.
"""

from __future__ import annotations

import law

from columnflow.inference import ParameterType, FlowStrategy
from columnflow.config_util import get_datasets_from_process

from hbt.inference.base import HBTInferenceModelBase


logger = law.logger.get_logger(__name__)


class default(HBTInferenceModelBase):
    """
    Default statistical model for the HH -> bbtautau analysis.
    """

    add_qcd = False
    fake_data = True

    def init_proc_map(self) -> None:
        # mapping of process names in the datacard ("combine name") to configs and process names in a dict
        name_map = dict([
            *[
                (f"ggHH_kl_{kl}_kt_1_13p6TeV_hbbhtt", f"hh_ggf_hbb_htt_kl{kl}_kt1")
                for kl in ["0", "1", "2p45", "5"]
            ],
            # ("ttbar", "tt"),
            ("ttbarV", "ttv"),
            # ("ttbarVV", "ttvv"),
            # ("singlet", "st"),
            # ("DY", "dy"),
            # # ("EWK", "z"),  # currently not use
            # ("W", "w"),
            # ("VV", "vv"),
            # ("VVV", "vvv"),
            # ("WH_htt", "wh"),
            # ("ZH_hbb", "zh"),
            # ("ggH_htt", "h_ggf"),
            # ("qqH_htt", "h_vbf"),
            # ("ttH_hbb", "tth"),
        ])
        if self.add_qcd:
            name_map["QCD"] = "qcd"

        # insert into proc_map
        # (same process name for all configs for now)
        for combine_name, proc_name in name_map.items():
            # same process name for all configs for now
            for config_inst in self.config_insts:
                _combine_name = self.inject_era(config_inst, combine_name)
                self.proc_map.setdefault(_combine_name, {})[config_inst] = proc_name

    def init_categories(self) -> None:
        for ch in ["etau", "mutau", "tautau"]:
            for cat in ["res1b", "res2b", "boosted"]:
                # gather fake processes to model data when needed
                fake_processes = []
                if self.fake_data:
                    fake_processes = list(set.union(*(
                        {
                            combine_name
                            for config_inst, proc_name in proc_map.items()
                            if (
                                not config_inst.get_process(proc_name).has_tag("nonresonant_signal") and
                                proc_name != "qcd"
                            )
                        }
                        for combine_name, proc_map in self.proc_map.items()
                    )))
                # add the category
                self.add_category(
                    f"cat_{self.campaign_key}_{ch}_{cat}",
                    config_data={
                        config_inst.name: self.category_config_spec(
                            category=f"{ch}__{cat}__os__iso",
                            # variable="res_dnn_hh_fine",
                            variable="jet1_pt",
                            data_datasets=["data_*"],
                        )
                        for config_inst in self.config_insts
                    },
                    data_from_processes=fake_processes,
                    mc_stats=10.0,
                    flow_strategy=FlowStrategy.move,
                )

    def init_processes(self) -> None:
        for combine_name, proc_map in self.proc_map.items():
            for config_inst, proc_name in proc_map.items():
                proc_inst = config_inst.get_process(proc_name)
                is_dynamic = proc_name == "qcd"
                dataset_names = []
                if not is_dynamic:
                    dataset_names = [
                        dataset.name
                        for dataset in get_datasets_from_process(config_inst, proc_name, strategy="all")
                    ]
                    if not dataset_names:
                        logger.debug(
                            f"skipping process {proc_name} in inference model {self.cls_name}, no matching datasets "
                            f"found in config {config_inst.name}",
                        )
                        continue
                self.add_process(
                    name=combine_name,
                    config_data={
                        config_inst.name: self.process_config_spec(
                            process=proc_name,
                            mc_datasets=dataset_names,
                        ),
                    },
                    is_signal=proc_inst.has_tag("nonresonant_signal"),
                    is_dynamic=is_dynamic,
                )

    def init_parameters(self) -> None:
        # general groups
        self.add_parameter_group("experiment")
        self.add_parameter_group("theory")

        # groups that contain parameters that solely affect the signal cross section and/or br
        self.add_parameter_group("signal_norm_xs")
        self.add_parameter_group("signal_norm_xsbr")

        # parameter that is added by the HH physics model, representing kl-dependent QCDscale + mtop
        # uncertainties on the ggHH cross section
        self.add_parameter_to_group("THU_HH", "theory")
        self.add_parameter_to_group("THU_HH", "signal_norm_xs")
        self.add_parameter_to_group("THU_HH", "signal_norm_xsbr")

        # helper to select processes across multiple configs
        def inject_all_eras(*names: str) -> list[str]:
            gen = (
                {self.inject_era(config_inst, name) for config_inst in self.config_insts}
                for name in names
            )
            return list(set.union(*gen))

        # theory uncertainties
        self.add_parameter(
            "BR_hbb",
            type=ParameterType.rate_gauss,
            process=["*_hbb", "*_hbbhtt"],
            effect=(0.9874, 1.0124),
            group=["theory", "signal_norm_xsbr"],
        )
        self.add_parameter(
            "BR_htt",
            type=ParameterType.rate_gauss,
            process=["*_htt", "*_hbbhtt"],
            effect=(0.9837, 1.0165),
            group=["theory", "signal_norm_xsbr"],
        )
        self.add_parameter(
            "pdf_gg",  # contains alpha_s
            type=ParameterType.rate_gauss,
            process=inject_all_eras("TT"),
            effect=1.042,
            group=["theory"],
        )
        self.add_parameter(
            "pdf_Higgs_ggHH",  # contains alpha_s
            type=ParameterType.rate_gauss,
            process="ggHH_*",
            effect=1.023,
            group=["theory", "signal_norm_xs", "signal_norm_xsbr"],
        )
        # self.add_parameter(
        #     "pdf_Higgs_qqHH",  # contains alpha_s
        #     type=ParameterType.rate_gauss,
        #     process="qqHH_*",
        #     effect=1.027,
        #     group=["theory", "signal_norm_xs", "signal_norm_xsbr"],
        # )
        self.add_parameter(
            "QCDscale_ttbar",
            type=ParameterType.rate_gauss,
            process=inject_all_eras("TT"),
            effect=(0.965, 1.024),
            group=["theory"],
        )
        # self.add_parameter(
        #     "QCDscale_qqHH",
        #     type=ParameterType.rate_gauss,
        #     process="qqHH_*",
        #     effect=(0.9997, 1.0005),
        #     group=["theory", "signal_norm_xs", "signal_norm_xsbr"],
        # )

        # lumi
        for config_inst in self.config_insts:
            ckey = self.campaign_keys[config_inst]
            lumi = config_inst.x.luminosity
            for unc_name in lumi.uncertainties:
                self.add_parameter(
                    unc_name,
                    type=ParameterType.rate_gauss,
                    effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
                    process=[f"*{ckey}*", "!QCD*"],
                    process_match_mode=all,
                    group="experiment",
                )

        # pileup
        # for config_inst in self.config_insts:
        #     ckey = self.campaign_keys[config_inst]
        #     self.add_parameter(
        #         f"CMS_pileup_20{ckey}",
        #         type=ParameterType.shape,
        #         config_data={
        #             config_inst.name: self.parameter_config_spec(shift_source="minbias_xs"),
        #         },
        #         process=[f"*{ckey}*", "!QCD*"],
        #         process_match_mode=all,
        #         group="experiment",
        #     )

        # btag
        # TODO: adapt for multi-config and jec correlation
        # for name in self.config_inst.x.btag_unc_names:
        #     self.add_parameter(
        #         f"CMS_btag_{name}",
        #         type=ParameterType.shape,
        #         config_data={
        #             self.config_inst.name: self.parameter_config_spec(shift_source=f"btag_{name}"),
        #         },
        #         group="experiment",
        #     )


@default.inference_model
def default_no_shifts(self):
    super(default_no_shifts, self).init_func()

    # remove all parameters that require a shift source other than nominal
    for category_name, process_name, parameter in self.iter_parameters():
        if parameter.type.is_shape or any(trafo.from_shape for trafo in parameter.transformations):
            self.remove_parameter(parameter.name, process=process_name, category=category_name)

    # repeat the cleanup
    self.init_cleanup()
