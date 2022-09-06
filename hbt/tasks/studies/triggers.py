# coding: utf-8

"""
Trigger relatad studies.
"""

from collections import OrderedDict

import luigi
import law

from columnflow.tasks.framework.base import DatasetTask
from columnflow.tasks.framework.mixins import DatasetsProcessesMixin
from columnflow.tasks.external import GetDatasetLFNs
from columnflow.util import ensure_proxy, dev_sandbox

from hbt.tasks.base import HBTTask


class PrintTriggersInFile(HBTTask, DatasetTask, law.tasks.RunOnceTask):
    """
    Prints a list of all HLT paths contained in the first file of a dataset.

    Example:

        > law run hbt.PrintTriggersInFile --dataset hh_ggf_bbtautau_madgraph
    """

    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar.sh")

    version = None

    # default upstream dependency task classes
    dep_GetDatasetLFNs = GetDatasetLFNs

    def requires(self):
        return self.dep_GetDatasetLFNs.req(self)

    @law.decorator.log
    @ensure_proxy
    @law.tasks.RunOnceTask.complete_on_success
    def run(self):
        # prepare input
        input_file = list(self.requires().iter_nano_files(self, lfn_indices=[0]))[0][1]

        # open with uproot
        with self.publish_step("load and open ..."):
            nano_file = input_file.load(formatter="uproot")

        # read HLT paths
        hlt_paths = [
            key[4:] for key in nano_file["Events"].keys()
            if key.startswith("HLT_")
        ]

        # print them
        print("")
        print("\n".join(hlt_paths))
        print("")


class PrintExistingConfigTriggers(HBTTask, DatasetsProcessesMixin, law.tasks.RunOnceTask):
    """
    Prints a table showing datasets (one per column) and contained HLT paths (one per row).

    Example:

        > law run hbt.PrintExistingConfigTriggers --datasets "hh_ggf_bbtautau_madgraph,data_mu_{b,c,d,e,f}"
    """

    table_format = luigi.Parameter(
        default="fancy_grid",
        description="a tabulate table format; default: 'fancy_grid'",
    )
    escape_markdown = luigi.BoolParameter(
        default=False,
        description="escape some characters for markdown; default: False",
    )

    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar.sh")

    version = None
    processes = None
    allow_empty_processes = True

    # default upstream dependency task classes
    dep_GetDatasetLFNs = GetDatasetLFNs

    def requires(self):
        return OrderedDict([
            (dataset, self.dep_GetDatasetLFNs.req(self, dataset=dataset))
            for dataset in self.datasets
        ])

    @law.decorator.log
    @ensure_proxy
    @law.tasks.RunOnceTask.complete_on_success
    def run(self):
        from tabulate import tabulate

        fmt = law.util.escape_markdown if self.escape_markdown else (lambda s: s)

        # table data
        header = ["HLT path"]
        rows = [[fmt(trigger.hlt_field)] for trigger in self.config_inst.x.triggers]

        for dataset, lfn_task in self.requires().items():
            # prepare input
            input_file = list(lfn_task.iter_nano_files(self, lfn_indices=[0]))[0][1]

            # open with uproot
            with self.publish_step("load and open ..."):
                nano_file = input_file.load(formatter="uproot")

            # read HLT paths
            hlt_paths = [
                key for key in nano_file["Events"].keys()
                if key.startswith("HLT_")
            ]

            # extend header and rows
            header.append(fmt(dataset))
            for trigger, row in zip(self.config_inst.x.triggers, rows):
                row.append(int(trigger.name in hlt_paths))

        print("")
        print(tabulate(rows, headers=header, tablefmt=self.table_format))
        print("")
