# coding: utf-8

"""
Default inference model.
"""

from __future__ import annotations

import re
import itertools
import collections

import law
import order as od

from columnflow.inference import ParameterType  # , ParameterTransformation
from columnflow.util import DotDict

from hbt.inference.base import HBTInferenceModel


logger = law.logger.get_logger(__name__)


class default(HBTInferenceModel):
    """
    Default statistical model for the HH -> bbtautau analysis.
    """

    # settings defined by HBTInferenceModel
    fake_data = True
    add_qcd = True

    # whether this model is used across run 3 campaigns or if it is meant for a single campaign only
    # (e.g. this influences the lumi uncertainty treatment)
    run3_multi_campaign = True

    # whether to include bbvv as an additional signal process
    add_bbvv = True

    # the default variable to use in all categories
    # (see get_category_variable for more details)
    variable = "run3_dnn_moe_hh_fine"

    # channels and phasespaces for category combinations
    channels = ["etau", "mutau", "tautau"]
    phasespaces = ["res1b", "res2b", "boosted"]

    def create_category_combinations(self) -> list[DotDict[str, str]]:
        return [
            DotDict(zip(["channel", "phasespace"], comb))
            for comb in itertools.product(self.channels, self.phasespaces)
        ]

    def create_category_info(self, *, channel: str, phasespace: str) -> HBTInferenceModel.CategoryInfo:
        return self.CategoryInfo(
            combine_category=f"cat_{self.campaign_key}_{channel}_{phasespace}",
            config_category=f"{channel}__{phasespace}__os__iso",
            config_variable=self.get_category_variable(channel=channel, phasespace=phasespace),
            config_data_datasets=["data_*"],
        )

    def get_category_variable(self, **kwargs) -> str:
        # return variable attribute if set
        if getattr(self, "variable", None):
            return self.variable

        raise NotImplementedError("get_category_variable not implemented in case of empty 'variable' attribute")

    def init_proc_map(self) -> None:
        name_map = self.create_proc_name_map()
        self.fill_proc_map(name_map)

    def create_proc_name_map(self) -> None:
        # mapping of process names in the datacard ("combine name") to configs and process names in a dict
        hh_nonbb_decays = [("tt", "tt")]  # maps postfix used in analysis to that used for combine name
        if self.add_bbvv:
            hh_nonbb_decays.append(("vv", "ww"))
        proc_name_map = {
            **{
                f"ggHH_kl_{kl}_kt_1_13p6TeV_hbbh{d_comb}": f"hh_ggf_hbb_h{d}_kl{kl}_kt1"
                for kl, in self.config_insts[0].x.hh_points.ggf
                for d, d_comb in hh_nonbb_decays
            },
            **{
                f"qqHH_CV_{kv}_C2V_{k2v}_kl_{kl}_13p6TeV_hbbh{d_comb}": f"hh_vbf_hbb_h{d}_kv{kv}_k2v{k2v}_kl{kl}"
                for kv, k2v, kl in self.config_insts[0].x.hh_points.vbf
                for d, d_comb in hh_nonbb_decays
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
            "WH_13p6TeV_hbb": "wh_hbb",
            "WH_13p6TeV_htt": "wh_htt",
            "ZH_13p6TeV_hbb": "zh_hbb",
            "ZH_13p6TeV_htt": "zh_htt",
            "ggH_13p6TeV_hbb": "h_ggf_hbb",
            "ggH_13p6TeV_htt": "h_ggf_htt",
            "qqH_13p6TeV_hbb": "h_vbf_hbb",
            "qqH_13p6TeV_htt": "h_vbf_htt",
            "ttH_13p6TeV_hbb": "tth_hbb",
            "ttH_13p6TeV_htt": "tth_hnonbb",
        }

        if self.add_qcd:
            proc_name_map["QCD_datadriven"] = "qcd"

        return proc_name_map

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

        #
        # simple rate parameters
        #

        # theory uncertainties
        self.add_parameter(
            "BR_hbb",
            type=ParameterType.rate_gauss,
            process=["*_hbb", "*_hbbhtt", "*_hbbhww"],
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
            "QCDscale_qqHH",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("qqHH_*"),
            effect=(0.9997, 1.0005),
            group=["theory", "signal_norm_xs", "signal_norm_xsbr", "rate_nuisances"],
        )
        self.add_parameter(
            "QCDscale_ttbar",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ttbar"),
            effect=(0.964, 1.024),
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "QCDscale_ttbar",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ttbarV"),
            effect=(0.981, 1.020),
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "QCDscale_ttbar",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("singlet"),
            effect=(0.981, 1.026),
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "QCDscale_V",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("W"),
            effect=(0.986, 1.013),
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "QCDscale_VH",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("WH_*"),
            effect=(0.993, 1.004),
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "QCDscale_VH",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ZH_*"),
            effect=(0.968, 1.038),
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "QCDscale_VV",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("VV"),
            effect=1.050,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "QCDscale_VVV",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("VVV"),
            effect=1.050,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "QCDscale_ggH",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ggH_*"),
            effect=1.039,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "QCDscale_qqH",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("qqH_*"),
            effect=(0.997, 1.005),
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "QCDscale_ttH",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ttH_*"),
            effect=(0.907, 1.06),
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_Higgs_ggHH",  # contains alpha_s
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ggHH_*_13p6TeV_hbbhtt", "ggHH_*_13p6TeV_hbbhww"),
            effect=1.023,
            group=["theory", "signal_norm_xs", "signal_norm_xsbr", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_Higgs_qqHH",  # contains alpha_s
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("qqHH_*_13p6TeV_hbbhtt", "qqHH_*_13p6TeV_hbbhww"),
            effect=1.027,
            group=["theory", "signal_norm_xs", "signal_norm_xsbr", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_qqbar",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ttbar"),
            effect=1.025,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_qqbar",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("W"),
            effect=1.008,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_qqbar",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("singlet"),
            effect=(0.978, 1.034),
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_qqbar",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("VV"),
            effect=1.050,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_qg",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ttbarV"),
            effect=1.024,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_gg",  # contains alpha_s
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ttbar"),
            effect=1.024,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_Higgs_gg",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ggH_*"),
            effect=1.019,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_Higgs_qqbar",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("qqH_*"),
            effect=1.021,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_Higgs_qqbar",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("WH_*"),
            effect=1.016,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_Higgs_qqbar",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ZH_*"),
            effect=1.013,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "pdf_Higgs_ttH",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ttH_*"),
            effect=1.030,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "alpha_s",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ggH_*"),
            effect=1.026,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "alpha_s",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("qqH_*"),
            effect=1.005,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "alpha_s",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("WH_*"),
            effect=1.009,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "alpha_s",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ZH_*"),
            effect=1.009,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "alpha_s",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ttH_*"),
            effect=1.020,
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "bbH_norm_ggH",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("ggH_*"),
            effect=(0.5, 1.5),
            group=["theory", "rate_nuisances"],
        )
        self.add_parameter(
            "bbH_norm_qqH",
            type=ParameterType.rate_gauss,
            process=self.inject_all_eras("qqH_*"),
            effect=(0.5, 1.5),
            group=["theory", "rate_nuisances"],
        )

        # lumi
        for config_inst in self.config_insts:
            lumi = config_inst.x.luminosity
            for unc_name in lumi.uncertainties:
                # depending on the run3_multi_campaign setting, either the single, year specific uncertainty is used,
                # or the correlated uncertainty scheme across all campaigns is used
                is_year_specific = str(config_inst.campaign.x.year) in unc_name
                if self.run3_multi_campaign == is_year_specific:
                    continue
                # add it
                self.add_parameter(
                    unc_name,
                    type=ParameterType.rate_gauss,
                    effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
                    process=self.process_matches(configs=config_inst, skip_datadriven=True),
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
                process=self.process_matches(configs=config_inst, skip_datadriven=True),
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
            process=self.inject_all_eras("ttbar"),
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
                process=self.process_matches(skip_datadriven=True),
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
                        process=self.process_matches(configs=config_inst, skip_datadriven=True),
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
                    process=self.process_matches(configs=config_insts, skip_datadriven=True),
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
                    process=self.process_matches(configs=config_inst, skip_datadriven=True),
                    process_match_mode=all,
                    group=["experiment", "shape_nuisances"],
                )

        # muon weights
        # TODO: possibly correlate?
        # TODO: 2024: MUO recommendations on de/correlating id/iso systematics across years should be implemented
        # see https://muon-wiki.docs.cern.ch/guidelines/corrections/#note-on-correlations
        for mu_source in ["mu_id", "mu_iso"]:
            for config_inst in self.config_insts:
                self.add_parameter(
                    f"CMS_eff_{mu_source}_{self.campaign_keys[config_inst]}",
                    type=ParameterType.shape,
                    config_data={
                        config_inst.name: self.parameter_config_spec(shift_source=mu_source),
                    },
                    category=["*_mutau_*"],
                    process=self.process_matches(configs=config_inst, skip_datadriven=True),
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
                    process=self.process_matches(configs=config_inst, skip_datadriven=True),
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
                    process=self.process_matches(configs=config_inst, skip_datadriven=True),
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

        # dy shifts
        for i, dy_name in enumerate(["syst", "stat"]):
            self.add_parameter(
                f"CMS_DY_{dy_name}_{config_inst.campaign.get_aux('year')}",
                type=ParameterType.shape,
                config_data={
                    config_inst.name: self.parameter_config_spec(shift_source=f"dy_{dy_name}")
                    for config_inst in self.config_insts
                },
                process=self.process_matches(processes=["DY"], configs=config_inst),
                process_match_mode=all,
                group=["experiment", "shape_nuisances"],
            )

        #
        # shape parameters based on entire dataset variations
        #

        # hdamp
        self.add_parameter(
            "hdamp",
            type=ParameterType.shape,
            config_data={
                config_inst.name: self.parameter_config_spec(shift_source="hdamp")
                for config_inst in self.config_insts
            },
            process=self.process_matches(processes=["ttbar", "singlet"], configs=self.config_insts),
            process_match_mode=all,
            group=["experiment", "shape_nuisances"],
        )

        # tune
        self.add_parameter(
            "underlying_event",
            type=ParameterType.shape,
            config_data={
                config_inst.name: self.parameter_config_spec(shift_source="tune")
                for config_inst in self.config_insts
            },
            process=self.process_matches(processes=["ttbar", "singlet"], configs=self.config_insts),
            process_match_mode=all,
            group=["experiment", "shape_nuisances"],
        )

        # mtop
        self.add_parameter(
            "mtop",
            type=ParameterType.shape,
            config_data={
                config_inst.name: self.parameter_config_spec(shift_source="mtop")
                for config_inst in self.config_insts
            },
            process=self.process_matches(processes=["ttbar", "singlet"], configs=self.config_insts),
            process_match_mode=all,
            group=["experiment", "shape_nuisances"],
        )


# helper to remove all parameters that require shifted inputs from a model instance
def remove_shift_parameters(model: default) -> None:
    # remove all parameters that require a shift source other than nominal
    for category_name, process_name, parameter in model.iter_parameters():
        remove = (
            (parameter.type.is_shape and not parameter.transformations.any_from_rate) or
            (parameter.type.is_rate and parameter.transformations.any_from_shape)
        )
        if remove:
            model.remove_parameter(parameter.name, process=process_name, category=category_name)


@default.inference_model
def default_no_shifts(self):
    super(default_no_shifts, self).init_func()
    remove_shift_parameters(self)
    self.init_cleanup()


default_no_shifts_jet1_pt = default_no_shifts.derive(
    "default_no_shifts_jet1_pt",
    cls_dict={"variable": "jet1_pt"},
)

default_no_shifts_no_vbf = default_no_shifts.derive(
    "default_no_shifts_no_vbf",
    cls_dict={"phasespaces": ["res1b_novbf", "res2b_novbf", "boosted_novbf"]},
)

default_no_shifts_simple = default_no_shifts.derive(
    "default_no_shifts_simple",
    cls_dict={"variable": "run3_dnn_simple_hh_fine"},
)

# for variables from networks trained with different kl variations
for kl in ["kl1", "kl0", "allkl"]:
    default_no_shifts.derive(
        f"default_no_shifts_simple_{kl}",
        cls_dict={"variable": f"run3_dnn_simple_{kl}_hh_fine"},
    )

# even 5k binning
default_no_shifts_simple_5k = default_no_shifts.derive(
    "default_no_shifts_simple_5k",
    cls_dict={"variable": "run3_dnn_moe_hh_fine_5k"},
)


@default.inference_model(variable="run3_dnn_moe_hh_fine_5k", empty_bin_value=0)
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


class default_cc(default):
    """
    Model that uses the full phase space (typically res1b, res2b, vbf, boosted) and also uses different variables.
    """

    phasespaces = ["res1b_cc", "res2b_cc", "boosted_cc", "vbf_cc"]
    use_logit = False

    def get_category_variable(self, *, channel: str, phasespace: str) -> str:
        # vbf dnn in vbf phasespace, otherwise default dnn
        if re.match(r"^vbf.*$", phasespace):
            return f"vbf_dnn_moe_hh_vbf_{'logit_' if self.use_logit else ''}fine"
        return f"run3_dnn_moe_hh_{'logit_' if self.use_logit else ''}fine"


@default_cc.inference_model
def default_cc_no_shifts(self):
    super(default_cc_no_shifts, self).init_func()
    remove_shift_parameters(self)
    self.init_cleanup()


default_cc_logit_no_shifts = default_cc_no_shifts.derive(
    "default_cc_logit_no_shifts",
    cls_dict={"use_logit": True},
)

default_cc_no_vbf_no_shifts = default_cc_no_shifts.derive(
    "default_cc_no_vbf_no_shifts",
    cls_dict={"phasespaces": ["res1b_inclvbf_cc", "res2b_inclvbf_cc", "boosted_cc"]},
)
