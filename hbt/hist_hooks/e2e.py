# coding: utf-8

"""
Histogram hooks for end-to-end DNN purposes.
"""

from __future__ import annotations

import functools

import law
import order as od

from columnflow.util import maybe_import
from columnflow.types import TYPE_CHECKING, Literal

np = maybe_import("numpy")
if TYPE_CHECKING:
    hist = maybe_import("hist")


logger = law.logger.get_logger(__name__)


def add_hooks(analysis_inst: od.Analysis) -> None:
    def e2e_sort(
        task: law.Task,
        hists: dict[od.Config, dict[od.Process, hist.Hist]],
        category_name: str,
        variable_name: str,
        *,
        sort_by: Literal["signal", "background", "signal_significance", "background_significance"],
        signal_process_name: str | None = "hh_ggf_hbb_htt_kl1_kt1",
    ) -> dict[od.Config, dict[od.Process, hist.Hist]]:
        import hist

        # get summed signal and background contents per bin, summing over configs and (leaf) categories
        s_data = None
        b_data = None
        for config_inst, proc_hists in hists.items():
            # get the leaf categories
            category_inst = config_inst.get_category(category_name)
            leaf_cats = [category_inst] if category_inst.is_leaf_category else category_inst.get_leaf_categories()

            # loop over all processes
            for proc_inst, h in proc_hists.items():
                # always skip data
                if proc_inst.is_data:
                    continue
                # when a signal process name pattern is given, use it to identify signals, otherwise use tag info
                is_signal = (
                    law.util.multi_match(proc_inst.name, signal_process_name)
                    if signal_process_name
                    else proc_inst.has_tag("signal")
                )
                # reduce and select axes
                h = h[{"category": [hist.loc(c.name) for c in leaf_cats if c.name in h.axes["category"]]}]
                h = h[{"category": sum, "shift": hist.loc("nominal")}]
                # add bin contents
                data = h.view().value
                if is_signal:
                    s_data = data if s_data is None else (s_data + data)
                else:
                    b_data = data if b_data is None else (b_data + data)

        # helper to check for data
        def check(name: str, data: np.ndarray | None) -> None:
            if data is None:
                raise ValueError(f"no {name} histograms found for e2e sorting")

        # determine sorting indices
        label_postfix = ""
        if sort_by == "signal":
            check("signal", s_data)
            sort_indices = np.argsort(s_data)
            label_postfix = " (sig. sorted)"
        elif sort_by == "background":
            check("background", b_data)
            sort_indices = np.argsort(b_data)
            label_postfix = " (bkg. sorted)"
        elif sort_by == "signal_significance":
            check("signal", s_data)
            check("background", b_data)
            sort_indices = np.argsort(s_data / np.sqrt(s_data + b_data))
            label_postfix = " (s/sqrt(s + b) sorted)"
        elif sort_by == "background_significance":
            check("signal", s_data)
            check("background", b_data)
            sort_indices = np.argsort(b_data / np.sqrt(s_data + b_data))
            label_postfix = " (b/sqrt(s + b) sorted)"
        else:
            raise ValueError(f"invalid sort_by value: {sort_by}")

        # apply sorting to all hists
        for proc_hists in hists.values():
            for h in proc_hists.values():
                # in place re-ordering of values and variances
                view = h.view()
                view.value[...] = view.value[..., sort_indices]
                view.variance[...] = view.variance[..., sort_indices]
                # ammend the label
                if label_postfix:
                    h.axes[variable_name].label += label_postfix

        return hists

    # add hooks
    analysis_inst.x.hist_hooks.e2e_sort_s = functools.partial(e2e_sort, sort_by="signal")
    analysis_inst.x.hist_hooks.e2e_sort_s_sig = functools.partial(e2e_sort, sort_by="signal_significance")
    analysis_inst.x.hist_hooks.e2e_sort_b = functools.partial(e2e_sort, sort_by="background")
    analysis_inst.x.hist_hooks.e2e_sort_b_sig = functools.partial(e2e_sort, sort_by="background_significance")
