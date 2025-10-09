# coding: utf-8

"""
Tasks to print various statistics.
"""

import os
import glob
import functools

import law

from columnflow.tasks.framework.base import ConfigTask

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


class CheckCATUpdates(HBTTask, ConfigTask, law.tasks.RunOnceTask):

    version = None
    single_config = False

    def run(self):
        # helpers to convert date tuples to strings and back
        decode_date_str = lambda s: tuple(map(int, s.split("-")))
        encode_date_str = lambda tpl: "-".join(map(str, tpl))

        # loop through configs
        for config_inst in self.config_insts:
            with self.publish_step(
                f"checking CAT meta data updates for config '{law.util.colored(config_inst.name, style='bright')}' in "
                f"{config_inst.x.cat_root}",
            ):
                newest_dates = {}
                updated_any = False
                for pog, date_str in config_inst.x.cat_info.snapshot.items():
                    newest_dates[pog] = date_str
                    if not date_str:
                        continue
                    # get all versions in the cat directory, split by date numbers
                    pog_era_dir = os.path.join(config_inst.x.cat_root, pog.upper(), config_inst.x.cat_info.key)
                    dates = [
                        decode_date_str(os.path.basename(path))
                        for path in glob.glob(os.path.join(pog_era_dir, "*-*-*"))
                    ]
                    if not dates:
                        raise ValueError(f"no CAT snapshots found in '{pog_era_dir}'")
                    # compare with current date
                    latest_date = max(dates)
                    if date_str == "latest" or decode_date_str(date_str) < latest_date:
                        latest_date_str = encode_date_str(latest_date)
                        newest_dates[pog] = latest_date_str
                        updated_any = True
                        self.publish_message(f"found newer {pog.upper()} snapshot: {date_str} -> {latest_date_str}")

                # print a new CATSnapshot line that can be copy-pasted into the config
                if updated_any:
                    args_str = ", ".join(f"{pog}=\"{date_str}\"" for pog, date_str in newest_dates.items() if date_str)
                    self.publish_message(
                        f"{law.util.colored('new CATSnapshot line ->', style='bright')} CATSnapshot({args_str})\n",
                    )
                else:
                    self.publish_message("no updates found\n")
