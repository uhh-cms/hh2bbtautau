# coding: utf-8

"""
Tasks to print various statistics.
"""

import functools

import law

from columnflow.tasks.framework.base import ConfigTask
from columnflow.util import dev_sandbox

from hbt.tasks.base import HBTTask
from hbt.tasks.parameters import table_format_param


class ListDatasetStats(HBTTask, ConfigTask, law.tasks.RunOnceTask):

    table_format = table_format_param

    # no version required
    version = None

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    def run(self):
        import tabulate
        tabulate.PRESERVE_WHITESPACE = True

        # color helpers
        green = functools.partial(law.util.colored, color="green")
        red = functools.partial(law.util.colored, color="red")
        red_bright = functools.partial(law.util.colored, color="red", style="bright")
        cyan = functools.partial(law.util.colored, color="cyan")
        bright = functools.partial(law.util.colored, style="bright")

        # headers
        headers = ["Dataset", "Files", "Events"]

        # content
        rows = []
        sum_files_nominal, sum_events_nominal = 0, 0
        sum_files_syst, sum_events_syst = 0, 0
        sum_files_data, sum_events_data = 0, 0
        for dataset_inst in self.config_inst.datasets:
            col = (
                cyan
                if dataset_inst.is_data
                else (green if dataset_inst.name.startswith("hh_") else red)
            )
            # nominal info
            rows.append([col(dataset_inst.name), dataset_inst.n_files, dataset_inst.n_events])
            # increment sums
            if dataset_inst.is_data:
                sum_files_data += dataset_inst.n_files
                sum_events_data += dataset_inst.n_events
            else:
                sum_files_nominal += dataset_inst.n_files
                sum_events_nominal += dataset_inst.n_events
            # potential shifts
            for shift_name, info in dataset_inst.info.items():
                if shift_name == "nominal" or shift_name not in self.config_inst.shifts:
                    continue
                rows.append([red_bright(f"  → {shift_name}"), info.n_files, info.n_events])
                # increment sums
                sum_files_syst += info.n_files
                sum_events_syst += info.n_events
        # sums
        rows.append([bright("total MC (nominal)"), sum_files_nominal, sum_events_nominal])
        if sum_files_syst or sum_events_syst:
            sum_files = sum_files_nominal + sum_files_syst
            sum_events = sum_events_nominal + sum_events_syst
            rows.append([bright("total MC (all)"), sum_files, sum_events])
        if sum_files_data or sum_events_data:
            rows.append([bright("total data"), sum_files_data, sum_events_data])

        # print the table
        table = tabulate.tabulate(rows, headers=headers, tablefmt=self.table_format, intfmt="_")
        self.publish_message(table)
