# coding: utf-8

from __future__ import annotations

import law
import order as od

from columnflow.plotting.plot_all import plot_all
from columnflow.plotting.plot_util import prepare_style_config, remove_residual_axis, use_flow_bins
from columnflow.util import maybe_import
from columnflow.types import TYPE_CHECKING, Any

if TYPE_CHECKING:
    hist = maybe_import("hist")
    plt = maybe_import("matplotlib.pyplot")


def multi_hist_plot(
    hists: dict[od.Process, hist.Hist] | dict[str, dict[od.Process, hist.Hist]],
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable] | dict[str, list[od.Variable]],
    shift_insts: list[od.Shift],
    style_config: dict | None = None,
    variable_settings: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    # emulate multi-variable input
    if isinstance(list(hists)[0], od.Process):
        assert isinstance(variable_insts, list)
        assert len(variable_insts) == 1
        hists = {variable_insts[0].name: hists}
        variable_insts = {variable_insts[0].name: variable_insts}

    # make sure all variable lists are 1D
    for var_name, var_insts in variable_insts.items():
        if len(var_insts) != 1:
            raise NotImplementedError(f"multi_plot only supports 1D histograms, but got variables {var_insts}")
        variable_insts[var_name] = var_insts[0]

    # remove shift axis
    hists = {var_name: remove_residual_axis(proc_hists, "shift") for var_name, proc_hists in hists.items()}

    # unique process instances
    unique_proc_insts = set(sum((list(v.keys()) for v in hists.values()), []))

    # helper to create legend labels
    def create_legend_label(proc_inst: od.Process, var_inst: od.Variable) -> str:
        return var_inst.x_title_short if len(unique_proc_insts) == 1 else f"{proc_inst.label}: {var_inst.x_title_short}"

    # helper to move flow bins into first/last ones
    def move_flow(h: hist.Hist, variable_inst: od.Variable) -> hist.Hist:
        overflow = variable_inst.x("overflow", False)
        underflow = variable_inst.x("underflow", False)
        if overflow or underflow:
            h = use_flow_bins(h, variable_inst.name, underflow=underflow, overflow=overflow)
        return h

    # create plot instructions for plot_all
    plot_config = {
        **{
            f"calib__{proc_inst.name}__{var_name}": {
                "method": "draw_hist",
                "hist": move_flow(h, variable_insts[var_name]),
                "kwargs": dict(
                    label=create_legend_label(proc_inst, variable_insts[var_name]),
                    color=config_inst.x.get_color_from_sequence(i),
                    linestyle=config_inst.x.get_line_style_from_sequence(j),
                    error_type="variance",
                    linewidth=1.5,
                ),
            }
            for i, (var_name, proc_hists) in enumerate(hists.items())
            for j, (proc_inst, h) in enumerate(proc_hists.items())
        },
    }

    # combine the style config
    style_config = law.util.merge_dicts(
        # inferred from config, category, variable
        prepare_style_config(
            config_inst=config_inst,
            category_inst=category_inst,
            variable_inst=list(variable_insts.values())[0],
        ),
        # passed styles
        style_config,
        (variable_settings or {}),
        # overwrite with hardcoded settings
        {
            "annotate_cfg": {"xy": (0.035, 0.95)},
            "cms_label_cfg": {"loc": 0},
            "legend_cfg": {"loc": "upper right", "fontsize": 18},
        },
        deep=True,
    )

    # update kwargs for plot_all
    kwargs = kwargs | {
        "skip_ratio": True,
        "skip_legend": False,
    }

    return plot_all(plot_config, style_config, **kwargs)
