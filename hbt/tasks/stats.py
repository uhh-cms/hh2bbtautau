# coding: utf-8

"""
Tasks to print various statistics.
"""

from __future__ import annotations

import functools
import dataclasses
import collections
import shlex

import luigi
import law
import order as od

from columnflow.tasks.framework.base import ConfigTask
from columnflow.selection import Selector
from columnflow.inference import InferenceModel
from columnflow.util import expand_path

from hbt.tasks.base import HBTTask
from hbt.tasks.parameters import table_format_param


class ListDatasetStats(HBTTask, ConfigTask, law.tasks.RunOnceTask):

    single_config = True

    table_format = table_format_param

    # no version required
    version = None

    def run(self):
        import tabulate
        tabulate.PRESERVE_WHITESPACE = True

        # color helpers
        green = functools.partial(law.util.colored, color="green")
        green_bright = functools.partial(law.util.colored, color="green", style="bright")
        yellow = functools.partial(law.util.colored, color="yellow")
        yellow_bright = functools.partial(law.util.colored, color="yellow", style="bright")
        red = functools.partial(law.util.colored, color="red")
        cyan = functools.partial(law.util.colored, color="cyan")
        cyan_bright = functools.partial(law.util.colored, color="cyan", style="bright")
        bright = functools.partial(law.util.colored, style="bright")

        def get_color(dataset_inst):
            if dataset_inst.is_data:
                return red
            if dataset_inst.has_tag("nonresonant_signal"):
                return green if dataset_inst.has_tag("ggf") else green_bright
            if dataset_inst.has_tag("resonant_signal"):
                return cyan if dataset_inst.has_tag("ggf") else cyan_bright
            return yellow

        # headers
        headers = ["Dataset", "Files", "Events"]

        # content
        rows = []
        sum_files_s_nonres, sum_events_s_nonres = 0, 0
        sum_files_s_res, sum_events_s_res = 0, 0
        sum_files_b_nom, sum_events_b_nom = 0, 0
        sum_files_b_syst, sum_events_b_syst = 0, 0
        sum_files_data, sum_events_data = 0, 0
        for dataset_inst in self.config_inst.datasets:
            col = get_color(dataset_inst)
            # nominal info
            rows.append([col(dataset_inst.name), dataset_inst.n_files, dataset_inst.n_events])
            # increment sums
            if dataset_inst.is_data:
                sum_files_data += dataset_inst.n_files
                sum_events_data += dataset_inst.n_events
            elif dataset_inst.has_tag("nonresonant_signal"):
                sum_files_s_nonres += dataset_inst.n_files
                sum_events_s_nonres += dataset_inst.n_events
            elif dataset_inst.has_tag("resonant_signal"):
                sum_files_s_res += dataset_inst.n_files
                sum_events_s_res += dataset_inst.n_events
            else:
                sum_files_b_nom += dataset_inst.n_files
                sum_events_b_nom += dataset_inst.n_events
            # potential shifts
            for shift_name, info in dataset_inst.info.items():
                if shift_name == "nominal" or shift_name not in self.config_inst.shifts:
                    continue
                rows.append([yellow_bright(f"  → {shift_name}"), info.n_files, info.n_events])
                # increment sums
                sum_files_b_syst += info.n_files
                sum_events_b_syst += info.n_events
        # overall
        sum_files_all = (
            sum_files_s_nonres + sum_files_s_res + sum_files_b_nom + sum_files_b_syst +
            sum_files_data
        )
        sum_events_all = (
            sum_events_s_nonres + sum_events_s_res + sum_events_b_nom + sum_events_b_syst +
            sum_events_data
        )

        # sums
        rows.append([bright("total signal (non-res.)"), sum_files_s_nonres, sum_events_s_nonres])
        rows.append([bright("total signal (res.)"), sum_files_s_res, sum_events_s_res])
        rows.append([bright("total background (nominal)"), sum_files_b_nom, sum_events_b_nom])
        if sum_files_b_syst or sum_events_b_syst:
            rows.append([bright("total background (syst.)"), sum_files_b_syst, sum_events_b_syst])
        if sum_files_data or sum_events_data:
            rows.append([bright("total data"), sum_files_data, sum_events_data])
        if sum_files_all or sum_events_all:
            rows.append([bright("total"), sum_files_all, sum_events_all])

        # print the table
        table = tabulate.tabulate(rows, headers=headers, tablefmt=self.table_format, intfmt="_")
        self.publish_message(table)


class CalculateLumi(HBTTask, ConfigTask, law.tasks.RunOnceTask):

    # extra arguments
    normtag_file = luigi.Parameter(
        default=law.NO_STR,
        description="custom normatag file; when empty, uses the file configured in the config; default: empty",
    )
    lumi_file = luigi.Parameter(
        default=law.NO_STR,
        description="custom (golden) lumi file; when empty, uses the file configured in the config; default: empty",
    )
    unit = luigi.ChoiceParameter(
        default="pb",
        choices=("fb", "pb"),
        description="inverse unit for the luminosity output; default: pb",
    )
    brilcalc_args = luigi.Parameter(
        default=law.NO_STR,
        description="additional arguments to pass to brilcalc; default: empty",
    )

    single_config = True
    version = None
    sandbox = "bash::/cvmfs/cms-bril.cern.ch/cms-lumi-pog/brilws-docker/brilws-env"

    def run(self):
        twiki = "https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun3"
        self.publish_message(f"building brilcalc command according to {law.util.colored(twiki, style='bright')}")

        # build the brilcalc command
        cmd = [
            "singularity",
            "-s", "exec",
            "--env", "PYTHONPATH=/home/bril/.local/lib/python3.10/site-packages",
            "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cloud/brilws-docker:latest",
            "brilcalc", "lumi",
            "--normtag", (
                expand_path(self.normtag_file)
                if self.normtag_file not in {law.NO_STR, "", None}
                else self.config_inst.x.external_files.lumi.normtag.location
            ),
            "-i", (
                expand_path(self.lumi_file)
                if self.lumi_file not in {law.NO_STR, "", None}
                else self.config_inst.x.external_files.lumi.golden.location
            ),
            "-u", f"/{self.unit}",
            *(
                shlex.split(self.brilcalc_args)
                if self.brilcalc_args not in {law.NO_STR, "", None}
                else []
            ),
        ]
        self.publish_message(f"cmd: {law.util.colored(law.util.quote_cmd(cmd), style='bright')}")

        # execute the command
        code = law.util.interruptable_popen(cmd)[0]
        if code != 0:
            raise RuntimeError(f"brilcalc command failed with exit code {code}")


class ListShifts(HBTTask, ConfigTask, law.tasks.RunOnceTask):
    """
    Default command:
        law run hbt.ListShifts --configs "22{per,post}_v14"

    Add rate parameters from inference model and filter to used shifts:
        law run hbt.ListShifts --configs "22{per,post}_v14" --inference-model "default"

    Same as above, but do not filter (i.e, show all shifts regardless):
        law run hbt.ListShifts --configs "22{per,post}_v14" --inference-model "default" --filter-model-sources False
    """

    selector = luigi.Parameter(
        default=law.NO_STR,
        description="name of a selector to identify whether a shift influences the selection; uses the config's "
        "'default_selector' field when empty; default: empty",
    )
    show_ids = luigi.BoolParameter(
        default=False,
        description="whether to show the shift ids in the table; default: False",
    )
    show_remarks = luigi.BoolParameter(
        default=True,
        description="whether to show remarks in the table; default: True",
    )
    inference_model = luigi.Parameter(
        default=law.NO_STR,
        description="the name of the inference model whose parameters should be used to validate, compare and extend "
        "the list of shifts; not used when empty; default: empty",
    )
    filter_model_sources = luigi.BoolParameter(
        default=False,
        description="whether to filter shifts to only contain those needed by in the inference model; only used when "
        "inference_model is set; default: False",
    )
    add_model_parameters = luigi.BoolParameter(
        default=True,
        description="whether to add parameters from the inference model that need no shifts (usually rate parameters) "
        "to the list of shifts; only used when inference_model is set; default: True",
    )
    table_format = table_format_param

    single_config = False
    version = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.inference_model not in {law.NO_STR, "", None}:
            inference_model_cls = InferenceModel.get_cls(self.inference_model)
            self.inference_model_inst = inference_model_cls(self.config_insts)
        else:
            self.inference_model = law.NO_STR
            self.inference_model_inst = None

    def run(self):
        import tabulate

        # style helpers
        styles = {
            # a source does not exist in all configs
            "inhomogeneous": {"style": "bright"},
            # rate type source
            "rate": {"color": "green"},
            # shape type source
            "shape": {"color": "yellow"},
            # dataset variation
            "dataset_variation": {"color": "cyan"},
            # shift influences selection
            "selection": {"color": "red"},
            # added by inference model only
            "model": {"color": "magenta"},
        }

        def stylize(style, text):
            kwargs = law.util.merge_dicts({}, *(styles.get(s, {}) for s in law.util.make_list(style)))
            return law.util.colored(text, **kwargs)

        # inference model helpers
        def model_needs_shift(source, config_inst):
            assert self.inference_model_inst is not None
            # find categories that require any of the selected configs
            categories = [
                cat_obj.name for cat_obj in self.inference_model_inst.categories
                if config_inst.name in cat_obj.config_data
            ]
            # check if at least one parameter of one process needs the shift source
            for _, _, param_obj in self.inference_model_inst.iter_parameters(category=categories):
                sources = {
                    d.shift_source for config_name, d in param_obj.config_data.items()
                    if config_name == config_inst.name
                }
                if source in sources:
                    return True
            return False

        def iter_additional_parameters(config_inst):
            assert self.inference_model_inst is not None
            # find categories that require any of the selected configs
            categories = [
                cat_obj.name for cat_obj in self.inference_model_inst.categories
                if config_inst.name in cat_obj.config_data
            ]
            # loop through parameters
            params = {}
            for _, _, param_obj in self.inference_model_inst.iter_parameters(category=categories):
                if not param_obj.config_data:
                    if param_obj.name not in params:
                        params[param_obj.name] = param_obj.type
                    elif params[param_obj.name] != param_obj.type:
                        params[param_obj.name] = None
            return params

        # container for named access to shift info
        @dataclasses.dataclass
        class Info:
            config_inst: od.Config
            shift_type: str
            shift_up: od.Shift
            shift_down: od.Shift
            affects_selection: bool
            in_model: bool | None = None

        # collect shift information
        shift_info: dict[str, dict[od.Config, Info]] = collections.defaultdict(dict)
        for config_inst in self.config_insts:
            # build a selector inst for the check below whether a shift affects the selection
            selector_name = (
                config_inst.x("default_selector", None)
                if self.selector in {law.NO_STR, "", None} else
                self.selector
            )
            selector_cls = Selector.get_cls(selector_name)
            selector_inst = selector_cls(inst_dict={
                "analysis_inst": self.analysis_inst,
                "config_inst": config_inst,
                "dataset_inst": config_inst.get_dataset("hh_ggf_hbb_htt_kl1_kt1_powheg"),
            })

            # loop through shift sources
            for source in ({shift_inst.source for shift_inst in config_inst.shifts} - {"nominal"}):
                # get up and down shift instances
                shift_up = config_inst.get_shift(f"{source}_up")
                shift_down = config_inst.get_shift(f"{source}_down")
                # check if the model contains / needs the shift
                in_model = None
                if self.inference_model_inst:
                    in_model = model_needs_shift(source, config_inst)
                # validation
                if shift_up.id % 2 != 1:
                    raise ValueError(f"up-shifts require have odd ids, but found {shift_up.id} for '{shift_up.name}'")
                if shift_down.id % 2 != 0:
                    raise ValueError(f"down-shifts require even ids, but found {shift_down.id} for '{shift_down.name}'")
                if shift_up.id + 1 != shift_down.id:
                    raise ValueError(
                        f"up-shift and down-shift ids must be consecutive, but found {shift_up.id} and {shift_down.id} "
                        f"for shift source '{source}'",
                    )
                if shift_up.type != shift_down.type:
                    raise ValueError(
                        f"up-shift and down-shift must have the same type, but found types '{shift_up.type}' and "
                        f"'{shift_down.type}' for shift source {source}",
                    )
                if (shift_up.name in selector_inst.all_shifts) != (shift_down.name in selector_inst.all_shifts):
                    raise ValueError(
                        f"up-shift and down-shift should be either both in or both not in the selector shifts, but "
                        f"found inconsistencies between '{shift_up.name}' and '{shift_down.name}' in shifts of "
                        f"selector '{selector_name}':\n{selector_inst.all_shifts}",
                    )
                # check if the selection is affected
                affects_selection = shift_up.name in selector_inst.all_shifts
                # store info
                shift_info[source][config_inst] = Info(
                    config_inst=config_inst,
                    shift_type=shift_up.type,
                    shift_up=shift_up,
                    shift_down=shift_down,
                    affects_selection=affects_selection,
                    in_model=in_model,
                )

        # fill table
        headers = ["Source / Config"] + [config_inst.name for config_inst in self.config_insts]
        rows = {}
        remarks = collections.defaultdict(list)
        empty = "-"
        for source in sorted(shift_info):
            config_info = shift_info[source]

            # source column label
            label = source
            label_styles = set()

            # add a remark if the source is not needed by the inference model or skip entirely
            if self.inference_model_inst and not any([info.in_model for info in config_info.values()]):
                if self.filter_model_sources:
                    self.publish_message(f"source '{source}' not needed by inference model, dropped")
                    continue
                else:
                    remarks[source].append("not in model")

            # check if the source stems from a dataset variation or affects the selection otherwise
            disjoint_flags = [info.shift_up.has_tag("disjoint_from_nominal") for info in config_info.values()]
            selection_flags = [info.affects_selection for info in config_info.values()]
            flag_str = lambda flags: ", ".join(map("{0[0]}:{0[1]}".format, zip(self.configs, flags)))
            if any(disjoint_flags):
                label_styles.add("dataset_variation")
                if not all(disjoint_flags):
                    remarks[source].append("disjointness varies")
                    self.logger.warning(f"source '{source}': dataset usage varies: {flag_str(disjoint_flags)}")
            elif any(selection_flags):
                label_styles.add("selection")
                if not all(selection_flags):
                    remarks[source].append("selection effect varies")
                    self.logger.warning(f"source '{source}': selection effect varies: {flag_str(selection_flags)}")

            # add remark if type differs across configs
            if len({info.shift_type for info in config_info.values()}) != 1:
                remarks[source].append("types differ")

            # entries per config
            entries = []
            for config_inst in self.config_insts:
                if (info := config_info.get(config_inst)):
                    entry = stylize(info.shift_type, info.shift_type)
                    if self.show_ids:
                        entry = f"{entry} ({info.shift_up.id}/{info.shift_down.id})"
                else:
                    entry = empty
                entries.append(entry)

            # mark source as inhomogeneous if needed
            if empty in entries:
                label_styles.add("inhomogeneous")

            # add the row
            rows[source] = [stylize(label_styles, label), *entries]

        # add additional model parameters
        if self.inference_model_inst and self.add_model_parameters:
            additional_parameters = collections.defaultdict(dict)
            for config_inst in self.config_insts:
                for param, consistent_type in iter_additional_parameters(config_inst).items():
                    additional_parameters[param][config_inst.name] = consistent_type
            for param, config_info in additional_parameters.items():
                assert param not in rows
                label_styles = {"model"}
                entries = []
                for config_inst in self.config_insts:
                    if config_inst.name not in config_info:
                        entries.append(empty)
                    elif not (consistent_type := config_info[config_inst.name]):
                        entries.append("multiple types")
                    else:
                        entries.append(stylize("rate" if consistent_type.is_rate else "shape", str(consistent_type)))
                if empty in entries:
                    label_styles.add("inhomogeneous")
                rows[param] = [stylize(label_styles, param), *entries]

        # add remarks column if needed
        if self.show_remarks and any(remarks.values()):
            headers.append("Remarks")
            for source, row in rows.items():
                row.append("; ".join(map(str, remarks[source])))

        # nothing to print if the table is empty
        if not rows:
            msg = f"\nno shifts {'left' if shift_info else 'found'} to create table\n"
            self.publish_message(law.util.colored(msg, color="red"))
            return

        # color legend
        source_legend_parts = [
            stylize("dataset_variation", "dataset variation"),
            stylize("selection", "affects selection"),
            stylize("model", "from model") if self.inference_model_inst else None,
            (stylize("inhomogeneous", "inhomogeneous across configs") + " (stacks)") if len(self.configs) > 1 else None,
        ]
        type_legend_parts = [
            stylize("rate", "rate"),
            stylize("shape", "shape"),
        ]
        legend = (
            f"Source legend : {' | '.join(filter(None, source_legend_parts))}\n"
            f"Type   legend : {' | '.join(filter(None, type_legend_parts))}\n"
        )

        # print table and legend
        table = tabulate.tabulate(rows.values(), headers=headers, tablefmt=self.table_format)
        self.publish_message(f"\n{table}\n\n{legend}")
