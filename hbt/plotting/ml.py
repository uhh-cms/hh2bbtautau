# coding: utf-8

from __future__ import annotations

import law
import order as od
import scinum

from columnflow.plotting.plot_all import plot_all
from columnflow.plotting.plot_util import prepare_style_config, remove_residual_axis
from columnflow.hist_util import create_hist_from_variables, sum_hists
from columnflow.util import maybe_import
from columnflow.types import TYPE_CHECKING

if TYPE_CHECKING:
    hist = maybe_import("hist")
    plt = maybe_import("matplotlib.pyplot")


def calibration_curve_plot(
    hists: dict[od.Process, hist.Hist] | dict[str, dict[od.Process, hist.Hist]],
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable] | dict[str, list[od.Variable]],
    shift_insts: list[od.Shift],
    style_config: dict | None = None,
    classifier_setup: str = "hh_tt_dy__equal",
    classifier_label: str = r"HH vs. $t\bar{t}$ vs. DY (all 1/3)",
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    # emulate multi-variable input
    if isinstance(list(hists)[0], od.Process):
        assert isinstance(variable_insts, list)
        assert len(variable_insts) == 1
        hists = {variable_insts[0].name: hists}
        variable_insts = {variable_insts[0].name: variable_insts}

    # helper to convert histogram to scinum array for stat error propagation
    hist_to_num = lambda h: scinum.Number(h.view().value, {h.proc_name: h.view().variance**0.5})

    # loop over variables and create calibration curve histograms
    calib_hists = {}
    for var_name, var_hists in hists.items():
        assert len(variable_insts[var_name]) == 1
        variable_inst = variable_insts[var_name][0]

        # remove shift axis
        var_hists = remove_residual_axis(var_hists, ax_name="shift", select_value="nominal")

        # separate into signal and background histograms
        signal_hists = []
        background_hists = []
        for process_inst, h in var_hists.items():
            h.proc_name = process_inst.name
            if process_inst.has_tag("signal"):
                signal_hists.append(h)
            else:
                background_hists.append(h)
        assert signal_hists
        assert background_hists

        # create calibration curve histogram
        calib_hist = create_hist_from_variables(variable_inst, weight=True)
        values = calib_hist.view().value
        variances = calib_hist.view().variance

        # fill it (this very much depends on the training setup)
        if classifier_setup == "hh_tt_dy__equal":
            # consider plain sum of signal histograms
            signal_hist = sum_hists(signal_hists)
            s = hist_to_num(signal_hist) / signal_hist.sum().value
            # sum background histograms, separated into dy and ttbar
            tt = sum_hists([h for h in background_hists if h.proc_name.startswith("tt")])
            dy = sum_hists([h for h in background_hists if h.proc_name.startswith("dy")])
            b_tt = hist_to_num(tt) / tt.sum().value
            b_dy = hist_to_num(dy) / dy.sum().value
            # r = s / (s + b_tt + b_dy)
            r = s / (s + b_tt + b_dy)
            r = r.combine_uncertainties()
            values[...] = r.n
            variances[...] = r.u()[0]**2  # symmetric uncertainties, pick up variation
        else:
            raise NotImplementedError(f"calibration curve for classifier setup '{classifier_setup}' not implemented")

        calib_hists[var_name] = calib_hist

    # create plot instructions for plot_all
    plot_config = {
        **{
            f"calib_{var_name}": {
                "method": "draw_errorbars",
                "hist": calib_hist,
                "kwargs": dict(
                    label=var_name,  # config_inst.get_variable(var_name).x_title, not descriptive enough yet
                    color=config_inst.x.get_color_from_sequence(i),
                    error_type="variance",
                    linestyle="-",
                    linewidth=1.0,
                ),
            }
            for i, (var_name, calib_hist) in enumerate(calib_hists.items())
        },
        "diagonal": {
            "method": (lambda ax, *args, **kwargs: ax.plot(
                [0, 1], [0, 1],
                ls="--",
                color="gray",
            )),
        },
        "classifier_label": {
            "method": (lambda ax, *args, **kwargs: ax.annotate(
                text=classifier_label,
                fontsize=style_config.get("annotate_cfg", {}).get("fontsize", 22),
                xy=(0.035, 0.905),
                xycoords="axes fraction",
                horizontalalignment="left",
                verticalalignment="top",
            )),
        },
    }

    # combine the style config
    style_config = law.util.merge_dicts(
        # inferred from config, category, variable
        prepare_style_config(
            config_inst=config_inst,
            category_inst=category_inst,
            variable_inst=variable_inst,
        ),
        # passed styles
        style_config,
        # overwrite with hardcoded settings
        {
            "annotate_cfg": {"xy": (0.035, 0.95)},
            "ax_cfg": {"ylim": (0.0, 1.0), "ylabel": "S/(S+B)"},
            "cms_label_cfg": {"loc": 0},
            "legend_cfg": {
                "loc": "upper left",
                "bbox_to_anchor": (-0.19, 0.84),
                "borderaxespad": 0.0,
            },
        },
        deep=True,
    )

    # update kwargs for plot_all
    kwargs = kwargs | {
        "skip_ratio": True,
        "skip_legend": len(hists) <= 1,
    }

    return plot_all(plot_config, style_config, **kwargs)
