# coding: utf-8

"""
Histogram hooks for blinding data points.
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


def add_hooks(analysis_inst: od.Analysis) -> None:
    """
    Add histogram hooks to an analysis.
    """
    def remove_data_hists(
        task: law.Task,
        hists: dict[od.Config, dict[od.Process, hist.Hist]],
        **kwargs,
    ) -> dict[od.Config, dict[od.Process, hist.Hist]]:
        """
        Remove data histograms from the input histograms.
        """
        return {
            config_inst: {proc: h for proc, h in proc_hists.items() if not proc.is_data}
            for config_inst, proc_hists in hists.items()
        }

    def _blind_dynamic(
        task: law.Task,
        hists: dict[od.Config, dict[od.Process, hist.Hist]],
        *,
        op: Literal["gt", "lt"],
        threshold: int | float,
        null_value: int | float = -100,
        **kwargs,
    ) -> dict[od.Config, dict[od.Process, hist.Hist]]:
        """
        Remove data histograms from the input histograms.
        """
        assert op in ["gt", "lt"]

        def blind(h: hist.Hist) -> hist.Hist:
            # get the variable axis
            ax = h.axes[-1]
            # find the first bin who's left (right) edge is strictly below (above) the treshold
            if op == "gt" and threshold < ax.edges[-1]:
                first_idx = np.where(ax.edges[1:] > threshold)[0][0]
                # offset by one to account for underflow bin when setting flow=True
                h.view(flow=True).value[..., first_idx + 1:] = null_value
                h.view(flow=True).variance[..., first_idx + 1:] = null_value
            elif op == "lt" and threshold > ax.edges[0]:
                last_idx = np.where(ax.edges[:-1] < threshold)[0][-1]
                # offset by one to account for overflow bin when setting flow=True
                h.view(flow=True).value[..., :last_idx - 1] = null_value
                h.view(flow=True).variance[..., :last_idx - 1] = null_value
            return h

        return {
            config_inst: {
                proc: (blind(h) if proc.is_data else h)
                for proc, h in proc_hists.items()
            }
            for config_inst, proc_hists in hists.items()
        }

    # add hooks
    analysis_inst.x.hist_hooks.blind = remove_data_hists
    analysis_inst.x.hist_hooks.blind_gt0p5 = functools.partial(_blind_dynamic, op="gt", threshold=0.5)
    analysis_inst.x.hist_hooks.blind_gt0p75 = functools.partial(_blind_dynamic, op="gt", threshold=0.75)
    analysis_inst.x.hist_hooks.blind_lt0p5 = functools.partial(_blind_dynamic, op="lt", threshold=0.5)
    analysis_inst.x.hist_hooks.blind_lt0p75 = functools.partial(_blind_dynamic, op="lt", threshold=0.75)
