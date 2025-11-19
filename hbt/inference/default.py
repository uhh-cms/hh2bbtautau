# coding: utf-8

"""
Default inference model.
"""

from __future__ import annotations

import re
import functools
import collections

import law
import order as od

from columnflow.inference import ParameterType, FlowStrategy  # , ParameterTransformation
from columnflow.config_util import get_datasets_from_process

from hbt.inference.base import HBTInferenceModelBase


logger = law.logger.get_logger(__name__)

get_all_datasets_from_process = functools.partial(
    get_datasets_from_process,
    strategy="all",
    only_first=False,
)


class default(HBTInferenceModelBase):
    """
    Default statistical model for the HH -> bbtautau analysis.
    """

    add_qcd = True
    fake_data = True
    variable_name = "run3_dnn_moe_hh_fine"
    channel_names = ["etau", "mutau", "tautau"]
    category_names = ["res1b", "res2b", "boosted"]

    def init_proc_map(self) -> None:
        # mapping of process names in the datacard ("combine name") to configs and process names in a dict
        name_map = {
            **{
                f"ggHH_kl_{kl}_kt_1_13p6TeV_hbbhtt": f"hh_ggf_hbb_htt_kl{kl}_kt1"
                for kl in ["0", "1", "2p45", "5"]
            },
            **{
                f"qqHH_CV_{kv}_C2V_{k2v}_kl_{kl}_13p6TeV_hbbhtt": f"hh_vbf_hbb_htt_kv{kv}_k2v{k2v}_kl{kl}"
                for kv, k2v, kl in [
                    ("1", "1", "1"),
                    ("1", "0", "1"),
                    ("1p74", "1p37", "14p4"),
                    ("2p12", "3p87", "m5p96"),
                    ("m0p012", "0p03", "10p2"),
                    ("m0p758", "1p44", "m19p3"),
                    ("m0p962", "0p959", "m1p43"),
                    ("m1p21", "1p94", "m0p94"),
                    ("m1p6", "2p72", "m1p36"),
                    ("m1p83", "3p57", "m3p39"),
                ]
            },
            "ttbar": "tt",
            "ttbarV": "ttv",
            "ttbarVV": "ttvv",
            "singlet": "st",
            "DY": "dy",
            # "EWK": "z",  # currently not used
            "W": "w",
            "VV": "vv",
            "VVV": "vvv",
            "WH_htt": "wh",
            "ZH_hbb": "zh",
            "ggH_htt": "h_ggf",
            "qqH_htt": "h_vbf",
            "ttH_hbb": "tth",
        }
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
        for ch in self.channel_names:
            for cat in self.category_names:
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
                            variable=self.variable_name,
                            data_datasets=["data_*"],
                        )
                        for config_inst in self.config_insts
                    },
                    data_from_processes=fake_processes,
                    mc_stats=10,
                    empty_bin_value=1e-5,  # setting this to 0 disables empty bin filling
                    flow_strategy=FlowStrategy.move,
                )

    def init_processes(self) -> None:
        # loop through process map and add process objects
        self.processes_with_lhe_weights = set()
        for combine_name, proc_map in self.proc_map.items():
            for config_inst, proc_name in proc_map.items():
                proc_inst = config_inst.get_process(proc_name)
                is_dynamic = proc_name == "qcd"
                dataset_names = []
                if not is_dynamic:
                    dataset_names = [
                        dataset.name
                        for dataset in get_all_datasets_from_process(config_inst, proc_name)
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
                # store whether there is at least one dataset contributing to this process with lhe weights
                if not all(config_inst.get_dataset(d).has_tag("no_lhe_weights") for d in dataset_names):
                    self.processes_with_lhe_weights.add(combine_name)

    def init_parameters(self) -> None:
        # general groups
        self.add_parameter_group("experiment")
        self.add_parameter_group("theory")
        self.add_parameter_group("rate_nuisances")
        self.add_parameter_group("shape_nuisances")

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

        # helper to create process patterns to match specific rules
        def process_matches(
            *,
            processes: str | list[str] | None = None,
            configs: od.Config | list[od.Config] | None = None,
            skip_qcd: bool = False,
        ) -> list[str] | None:
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

        #
        # simple rate parameters
        #

        # theory uncertainties
        self.add_parameter(
            "BR_hbb",
            type=ParameterType.rate_gauss,
            process=["*_hbb", "*_hbbhtt"],
            effect=(0.9874, 1.0124),
            group=["theory", "signal_norm_xsbr", "rate_nuisances"],
        )
        self.add_parameter(
            "BR_htt",
            type=ParameterType.rate_gauss,
            process=["*_htt", "*_hbbhtt"],
            effect=(0.9837, 1.0165),
            group=["theory", "signal_norm_xsbr", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_gg",  # contains alpha_s
            type=ParameterType.rate_gauss,
            process=inject_all_eras("ttbar"),
            effect=1.042,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_Higgs_ggHH",  # contains alpha_s
            type=ParameterType.rate_gauss,
            process="ggHH_*",
            effect=1.023,
            group=["theory", "signal_norm_xs", "signal_norm_xsbr", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_Higgs_qqHH",  # contains alpha_s
            type=ParameterType.rate_gauss,
            process="qqHH_*",
            effect=1.027,
            group=["theory", "signal_norm_xs", "signal_norm_xsbr", "rate_nuisances"],
        )
        self.add_parameter(
            "QCDscale_ttbar",
            type=ParameterType.rate_gauss,
            process=inject_all_eras("ttbar"),
            effect=(0.965, 1.024),
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "QCDscale_qqHH",
            type=ParameterType.rate_gauss,
            process="qqHH_*",
            effect=(0.9997, 1.0005),
            group=["theory", "signal_norm_xs", "signal_norm_xsbr", "rate_nuisances"],
        )
        self.add_parameter(
            "bbH_norm_ggH",
            type=ParameterType.rate_gauss,
            process="ggH_*",
            effect=(0.5, 1.5),
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "bbH_norm_qqH",
            type=ParameterType.rate_gauss,
            process="qqH_*",
            effect=(0.5, 1.5),
            group=["theory", "rate_nuisances"],
        )
        # TODO: additional theory uncertainties, especially on background processes!

        # lumi
        for config_inst in self.config_insts:
            lumi = config_inst.x.luminosity
            for unc_name in lumi.uncertainties:
                self.add_parameter(
                    unc_name,
                    type=ParameterType.rate_gauss,
                    effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
                    process=process_matches(configs=config_inst, skip_qcd=True),
                    process_match_mode=all,
                    group=["experiment", "rate_nuisances"],
                )

        #
        # shape parameters from shifts acting on ProduceColumns or CreateHistograms (mostly weight variations)
        #

        # pileup
        for config_inst in self.config_insts:
            self.add_parameter(
                f"CMS_pileup_{self.campaign_keys[config_inst]}",
                type=ParameterType.shape,
                config_data={
                    config_inst.name: self.parameter_config_spec(shift_source="minbias_xs"),
                },
                process=process_matches(configs=config_inst, skip_qcd=True),
                process_match_mode=all,
                group=["experiment", "shape_nuisances"],
            )

        # top pt weight
        self.add_parameter(
            "CMS_top_pT_reweighting",
            type=ParameterType.shape,
            config_data={
                config_inst.name: self.parameter_config_spec(shift_source="top_pt")
                for config_inst in self.config_insts
            },
            process=inject_all_eras("ttbar"),
            group=["experiment", "shape_nuisances"],
        )

        # pdf shape (could be decorrelated across some process groups if needed)
        self.add_parameter(
            "pdf_shape",
            type=ParameterType.shape,
            config_data={
                config_inst.name: self.parameter_config_spec(shift_source="pdf")
                for config_inst in self.config_insts
            },
            process=self.processes_with_lhe_weights,
            group=["theory", "shape_nuisances"],
        )

        # mur/muf shape (could be decorrelated across some process groups if needed)
        self.add_parameter(
            "scale_shape",
            type=ParameterType.shape,
            config_data={
                config_inst.name: self.parameter_config_spec(shift_source="murmuf")
                for config_inst in self.config_insts
            },
            process=self.processes_with_lhe_weights,
            group=["theory", "shape_nuisances"],
        )

        # isr and fsr (could be decorrelated across some process groups if needed)
        for source in ["isr", "fsr"]:
            self.add_parameter(
                f"ps_{source}",
                type=ParameterType.shape,
                config_data={
                    config_inst.name: self.parameter_config_spec(shift_source=source)
                    for config_inst in self.config_insts
                },
                process=process_matches(skip_qcd=True),
                group=["theory", "shape_nuisances"],
            )

        # btag
        btag_map: collections.defaultdict[str, list[od.Config]] = collections.defaultdict(list)
        for config_inst in self.config_insts:
            for name in config_inst.x.btag_unc_names:
                btag_map[name].append(config_inst)
        for name, config_insts in btag_map.items():
            # decorrelate hf/lfstats across years, correlate others
            if re.match(r"^(l|h)fstats(1|2)$", name):
                for config_inst in config_insts:
                    self.add_parameter(
                        f"CMS_btag_{name}_{self.campaign_keys[config_inst]}",
                        type=ParameterType.shape,
                        config_data={
                            config_inst.name: self.parameter_config_spec(shift_source=f"btag_{name}"),
                        },
                        process=process_matches(configs=config_inst, skip_qcd=True),
                        process_match_mode=all,
                        group=["experiment", "shape_nuisances"],
                    )
            else:
                self.add_parameter(
                    f"CMS_btag_{name}",
                    type=ParameterType.shape,
                    config_data={
                        config_inst.name: self.parameter_config_spec(shift_source=f"btag_{name}")
                        for config_inst in config_insts
                    },
                    process=process_matches(configs=config_insts, skip_qcd=True),
                    process_match_mode=all,
                    group=["experiment", "shape_nuisances"],
                )

        # electron weights
        # TODO: possibly correlate?
        for e_source in ["e_id", "e_reco"]:
            for config_inst in self.config_insts:
                self.add_parameter(
                    f"CMS_eff_{e_source}_{self.campaign_keys[config_inst]}",
                    type=ParameterType.shape,
                    config_data={
                        config_inst.name: self.parameter_config_spec(shift_source=e_source),
                    },
                    category=["*_etau_*"],
                    process=process_matches(configs=config_inst, skip_qcd=True),
                    process_match_mode=all,
                    group=["experiment", "shape_nuisances"],
                )

        # muon weights
        # TODO: possibly correlate?
        for mu_source in ["mu_id", "mu_iso"]:
            for config_inst in self.config_insts:
                self.add_parameter(
                    f"CMS_eff_{mu_source}_{self.campaign_keys[config_inst]}",
                    type=ParameterType.shape,
                    config_data={
                        config_inst.name: self.parameter_config_spec(shift_source=mu_source),
                    },
                    category=["*_mutau_*"],
                    process=process_matches(configs=config_inst, skip_qcd=True),
                    process_match_mode=all,
                    group=["experiment", "shape_nuisances"],
                )

        # tau weights
        for config_inst in self.config_insts:
            for name in config_inst.x.tau_unc_names:
                # each uncertainty only applies to specific channels
                unc_channels = config_inst.get_shift(f"tau_{name}_up").x.applies_to_channels
                self.add_parameter(
                    f"CMS_eff_t_{name}_{self.campaign_keys[config_inst]}",
                    type=ParameterType.shape,
                    config_data={
                        config_inst.name: self.parameter_config_spec(shift_source=f"tau_{name}"),
                    },
                    category=[f"*_{ch}_*" for ch in unc_channels],
                    process=process_matches(configs=config_inst, skip_qcd=True),
                    process_match_mode=all,
                    group=["experiment", "shape_nuisances"],
                )

        # trigger weights
        for config_inst in self.config_insts:
            for name in config_inst.x.trigger_legs:
                # each uncertainty only applies to specific channels
                unc_channels = config_inst.get_shift(f"trigger_{name}_up").x.applies_to_channels
                self.add_parameter(
                    f"CMS_bbtt_eff_trig_{name}_{self.campaign_keys[config_inst]}",
                    type=ParameterType.shape,
                    config_data={
                        config_inst.name: self.parameter_config_spec(shift_source=f"trigger_{name}"),
                    },
                    category=[f"*_{ch}_*" for ch in unc_channels],
                    process=process_matches(configs=config_inst, skip_qcd=True),
                    process_match_mode=all,
                    group=["experiment", "shape_nuisances"],
                )

        #
        # shape parameters that alter the selection
        #

        # jec
        pass  # TODO

        # jer
        pass  # TODO

        # tec
        pass  # TODO

        # eec
        pass  # TODO

        # eer
        pass  # TODO

        # mec
        pass  # TODO

        # mer
        pass  # TODO

        #
        # shape parameters based on entire dataset variations
        #

        # hdamp
        # self.add_parameter(
        #     "hdamp",
        #     type=ParameterType.shape,
        #     config_data={
        #         config_inst.name: self.parameter_config_spec(shift_source="hdamp")
        #         for config_inst in self.config_insts
        #     },
        #     process=process_matches(processes=["ttbar", "singlet"], configs=self.config_insts, skip_qcd=True),
        #     process_match_mode=all,
        #     group=["experiment", "shape_nuisances"],
        # )

        # tune
        # self.add_parameter(
        #     "underlying_event",
        #     type=ParameterType.shape,
        #     config_data={
        #         config_inst.name: self.parameter_config_spec(shift_source="tune")
        #         for config_inst in self.config_insts
        #     },
        #     process=process_matches(processes=["ttbar", "singlet"], configs=self.config_insts, skip_qcd=True),
        #     process_match_mode=all,
        #     group=["experiment", "shape_nuisances"],
        # )

        # mtop
        # self.add_parameter(
        #     "mtop",
        #     type=ParameterType.shape,
        #     config_data={
        #         config_inst.name: self.parameter_config_spec(shift_source="mtop")
        #         for config_inst in self.config_insts
        #     },
        #     process=process_matches(processes=["ttbar", "singlet"], configs=self.config_insts, skip_qcd=True),
        #     process_match_mode=all,
        #     group=["experiment", "shape_nuisances"],
        # )


@default.inference_model
def default_no_shifts(self):
    super(default_no_shifts, self).init_func()

    # remove all parameters that require a shift source other than nominal
    for category_name, process_name, parameter in self.iter_parameters():
        remove = (
            (parameter.type.is_shape and not parameter.transformations.any_from_rate) or
            (parameter.type.is_rate and parameter.transformations.any_from_shape)
        )
        if remove:
            self.remove_parameter(parameter.name, process=process_name, category=category_name)

    # repeat the cleanup
    self.init_cleanup()


default_no_shifts_no_vbf = default_no_shifts.derive(
    "default_no_shifts_no_vbf",
    cls_dict={"category_names": ["res1b_novbf", "res2b_novbf", "boosted_novbf"]},
)

default_no_shifts_simple = default_no_shifts.derive(
    "default_no_shifts_simple",
    cls_dict={"variable_name": "run3_dnn_simple_hh_fine"},
)

# for variables from networks trained with different kl variations
for kl in ["kl1", "kl0", "allkl"]:
    default_no_shifts.derive(
        f"default_no_shifts_simple_{kl}",
        cls_dict={"variable_name": f"run3_dnn_simple_{kl}_hh_fine"},
    )

# even 5k binning
default_no_shifts_simple_5k = default_no_shifts.derive(
    "default_no_shifts_simple_5k",
    cls_dict={"variable_name": "run3_dnn_moe_hh_fine_5k"},
)


@default.inference_model(variable_name="run3_dnn_moe_hh_fine_5k", add_qcd=False)
def default_bin_opt(self):
    # set everything up as in the default model
    super(default_bin_opt, self).init_func()

    # only keep certain parameters
    keep_parameters = {
        "BR_*",
        "QCDscale_*",
        "bbH_norm_*",
        "lumi_*",
        "CMS_bbtt_eff_trig_*",
        "CMS_btag_*",
        "CMS_eff_e_*",
        "CMS_eff_mu_*",
        "CMS_eff_t_*",
        "CMS_top_pT_reweighting",
        "pdf_*",
        "ps_*",
        "scale_*",
    }
    for category_name, process_name, parameter in self.iter_parameters():
        if not law.util.multi_match(parameter.name, keep_parameters):
            self.remove_parameter(parameter.name, process=process_name, category=category_name)

    # repeat the cleanup
    self.init_cleanup()
