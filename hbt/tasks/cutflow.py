# coding: utf-8

"""
Cutflow plotting task with hist-hook support (e.g. QCD estimation),
applied per selection step.
"""

from collections import OrderedDict

import law

from columnflow.tasks.cutflow import PlotCutflowVariables1D
from columnflow.tasks.framework.mixins import HistHookMixin
from columnflow.tasks.framework.decorators import view_output_plots
from columnflow.hist_util import select_category_bins

from hbt.tasks.base import HBTTask


class PlotCutflowVariables1DWithHooks(HBTTask, HistHookMixin, PlotCutflowVariables1D):
    """
    Drop-in replacement for cf.PlotCutflowVariables1D that additionally supports
    --hist-hooks (e.g. QCD estimation), matching the behavior of cf.PlotVariables1D.

    Unlike a naive port that hooks into run_postprocess(), this overrides run()
    itself so the hook fires *before* the "category" axis is reduced away -- our
    hooks (QCD estimation) need to see all ABCD regions (os/ss/iso/noniso) of a
    group simultaneously, and category_ids is populated identically across every
    cutflow "step", so the estimate is naturally computed per step for free.
    """

    def store_parts(self):
        parts = super().store_parts()
        hooks_repr = self.hist_hooks_repr
        if hooks_repr:
            parts["hook"] = f"hooks_{hooks_repr}"
        return parts

    @law.decorator.notify
    @law.decorator.log
    @view_output_plots
    def run(self):
        import hist
        import order as od

        # copy process instances once so that their auxiliary data fields can be used as a
        # storage for process-specific plot parameters later on, without affecting originals
        fake_root = od.Process(
            name=f"{hex(id(object()))[2:]}",
            id="+",
            processes=list(map(self.config_inst.get_process, self.processes)),
        ).copy()
        process_insts = list(fake_root.processes)
        fake_root.processes.clear()

        variable_tuple = self.variable_tuples[self.branch_data.variable]
        variable_insts = [
            self.config_inst.get_variable(var_name)
            for var_name in variable_tuple
        ]
        category_inst = self.config_inst.get_category(self.branch_data.category)
        sub_process_insts = {
            process_inst: [sub for sub, _, _ in process_inst.walk_processes(include_self=True)]
            for process_inst in process_insts
        }

        hists = {}

        with self.publish_step(f"plotting {self.branch_data.variable} in {category_inst.name}"):
            for dataset, inp in self.input().items():
                dataset_inst = self.config_inst.get_dataset(dataset)
                h_in = inp[self.branch_data.variable].load(formatter="pickle")

                # select shift
                if (n_shifts := len(h_in.axes["shift"])) != 1:
                    raise Exception(f"shift axis is supposed to only contain 1 bin, found {n_shifts}")
                h_in = h_in[{"shift": hist.loc(self.global_shift_inst.name)}]

                # loop and extract one histogram per process
                for process_inst in process_insts:
                    if not any(
                        dataset_inst.has_process(sub_process_inst.name)
                        for sub_process_inst in sub_process_insts[process_inst]
                    ):
                        continue

                    h = h_in.copy()
                    h = h[{
                        "process": [
                            hist.loc(p.name)
                            for p in sub_process_insts[process_inst]
                            if p.name in h.axes["process"]
                        ],
                    }]
                    h = h[{"process": sum}]

                    if process_inst in hists:
                        hists[process_inst] += h
                    else:
                        hists[process_inst] = h

            if not hists:
                raise Exception("no histograms found to plot")

            # keep process ordering, but do NOT reduce the category axis yet -- the hook
            # needs the full multi-region "category" axis (os/ss/iso/noniso) intact
            hists = OrderedDict(
                (process_inst, hists[process_inst])
                for process_inst in sorted(hists, key=process_insts.index)
            )

            # --- hook runs here: axes are still (category, step, variable) ---
            self.logger.info(f"hist axes going into hook: {[a.name for a in next(iter(hists.values())).axes]}")
            wrapped = self.invoke_hist_hooks(
                {self.config_inst: hists},
                hook_kwargs={
                    "category_name": category_inst.name,
                    "variable_name": self.branch_data.variable,
                },
            )
            hists = wrapped[self.config_inst]
 
            # now reduce down to the single requested region for plotting
            # (do NOT re-sort by process_insts.index here: hooks may add new process
            # instances -- e.g. "qcd" fetched fresh via config_inst.get_process() --
            # that are not identity-equal to the copies in process_insts, which would
            # break .index() lookups; iterate the hook's returned dict as-is instead,
            # matching how the stock PlotVariablesBase.run() handles this)
            hists = OrderedDict(
                (process_inst, select_category_bins(
                    h, category_inst, use_leaves=True, prefer_parents=True, reduce=True,
                ))
                for process_inst, h in hists.items()
            )
            # from IPython import embed; embed()
            # print(hists)
            self.run_postprocess(
                hists=hists,
                category_inst=category_inst,
                variable_insts=variable_insts,
            )