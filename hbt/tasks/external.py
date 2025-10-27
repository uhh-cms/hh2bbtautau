# coding: utf-8

"""
Tasks to print various statistics.
"""

import functools
import shlex

import luigi
import law

from columnflow.tasks.framework.base import ConfigTask
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
                else self.config_inst.x.external_files.lumi.normtag[0]
            ),
            "-i", (
                expand_path(self.lumi_file)
                if self.lumi_file not in {law.NO_STR, "", None}
                else self.config_inst.x.external_files.lumi.golden[0]
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
