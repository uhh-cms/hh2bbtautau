# coding: utf-8

"""
Histogram hooks for blinding data points.
"""

from __future__ import annotations

import law
import order as od

from columnflow.util import maybe_import

hist = maybe_import("hist")


def add_hooks(analysis_inst: od.Analysis) -> None:
    """
    Add histogram hooks to an analysis.
    """
    def remove_data_hists(
        task: law.Task,
        hists: dict[od.Config, dict[od.Process, hist.Hist]],
    ) -> dict[od.Config, dict[od.Process, hist.Hist]]:
        """
        Remove data histograms from the input histograms.
        """
        return {
            config_inst: {proc: hist for proc, hist in proc_hists.items() if not proc.is_data}
            for config_inst, proc_hists in hists.items()
        }

    # add hooks
    analysis_inst.x.hist_hooks.blind = remove_data_hists
