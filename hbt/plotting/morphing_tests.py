from __future__ import annotations

from collections import OrderedDict
from functools import partial

import law

from columnflow.util import maybe_import
from columnflow.plotting.plot_all import plot_all
from columnflow.plotting.plot_util import (
    # prepare_plot_config,
    prepare_style_config,
    remove_residual_axis,
    apply_variable_settings,
    apply_process_settings,
    apply_density_to_hists,
    # get_position,
    # get_profile_variations,
    blind_sensitive_bins,
)

from columnflow.types import Callable

hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
od = maybe_import("order")


def prepare_plot_config_custom_ratio(
    hists: OrderedDict,
    shape_norm: bool | None = False,
    hide_errors: bool | None = None,
) -> OrderedDict:
    """
    Prepares a plot config with one entry to create plots containing a stack of
    backgrounds with uncertainty bands, unstacked processes as lines and
    data entrys with errorbars.
    """

    # separate histograms into stack, lines and data hists
    mc_hists, mc_colors, mc_edgecolors, mc_labels = [], [], [], []
    line_hists, line_colors, line_labels, line_hide_errors = [], [], [], []
    data_hists, data_hide_errors = [], []

    # not necessary since we are not using any data, but kept for now
    for process_inst, h in hists.items():
        # if given, per-process setting overrides task parameter
        proc_hide_errors = hide_errors
        if getattr(process_inst, "hide_errors", None) is not None:
            proc_hide_errors = process_inst.hide_errors
        if process_inst.is_data:
            data_hists.append(h)
            data_hide_errors.append(proc_hide_errors)
        elif process_inst.is_mc:
            if getattr(process_inst, "unstack", False):
                line_hists.append(h)
                line_colors.append(process_inst.color1)
                line_labels.append(process_inst.label)
                line_hide_errors.append(proc_hide_errors)
            else:
                mc_hists.append(h)
                mc_colors.append(process_inst.color1)
                mc_edgecolors.append(process_inst.color2)
                mc_labels.append(process_inst.label)

    h_data, h_mc, h_mc_stack = None, None, None
    if data_hists:
        h_data = sum(data_hists[1:], data_hists[0].copy())
    if mc_hists:
        h_mc = sum(mc_hists[1:], mc_hists[0].copy())
        # reverse hists when building MC stack so that the
        # first process is on top
        h_mc_stack = hist.Stack(*mc_hists[::-1])

    # setup plotting configs
    plot_config = OrderedDict()

    # draw stack
    if h_mc_stack is not None:
        mc_norm = sum(h_mc.values()) if shape_norm else 1
        plot_config["mc_stack"] = {
            "method": "draw_stack",
            "hist": h_mc_stack,
            "kwargs": {
                "norm": mc_norm,
                "label": mc_labels[::-1],
                "color": mc_colors[::-1],
                "edgecolor": mc_edgecolors[::-1],
                "linewidth": [(0 if c is None else 1) for c in mc_colors[::-1]],
            },
        }

    # draw lines
    for i, h in enumerate(line_hists):
        line_norm = sum(h.values()) if shape_norm else 1
        plot_config[f"line_{i}"] = plot_cfg = {
            "method": "draw_hist",
            "hist": h,
            "kwargs": {
                "norm": line_norm,
                "label": line_labels[i],
                "color": line_colors[i],
            },
            # "ratio_kwargs": {
            #     "norm": h.values(),
            #     "color": line_colors[i],
            # },
        }

        # suppress error bars by overriding `yerr`
        if line_hide_errors[i]:
            for key in ("kwargs", "ratio_kwargs"):
                if key in plot_cfg:
                    plot_cfg[key]["yerr"] = False

    # build ratio_kwargs: standard divided by morphed, error bars are the standard error
    if len(line_hists) == 2:
        if "morphed" in plot_config["line_0"]["kwargs"]["label"]:
            i = 0
            j = 1
        else:
            i = 1
            j = 0
        plot_config[f"line_{j}"]["ratio_kwargs"] = {
            "norm": line_hists[i].values(),
            "color": line_colors[j],
        }

    # draw stack error
    if h_mc_stack is not None and not hide_errors:
        mc_norm = sum(h_mc.values()) if shape_norm else 1
        plot_config["mc_uncert"] = {
            "method": "draw_error_bands",
            "hist": h_mc,
            "kwargs": {"norm": mc_norm, "label": "MC stat. unc."},
            "ratio_kwargs": {"norm": h_mc.values()},
        }

    # draw data
    if data_hists:
        data_norm = sum(h_data.values()) if shape_norm else 1
        plot_config["data"] = plot_cfg = {
            "method": "draw_errorbars",
            "hist": h_data,
            "kwargs": {
                "norm": data_norm,
                "label": "Data",
            },
        }

        if h_mc is not None:
            plot_config["data"]["ratio_kwargs"] = {
                "norm": h_mc.values() * data_norm / mc_norm,
            }

        # suppress error bars by overriding `yerr`
        if any(data_hide_errors):
            for key in ("kwargs", "ratio_kwargs"):
                if key in plot_cfg:
                    plot_cfg[key]["yerr"] = False

    return plot_config


def plot_morphing_comparison(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool | None = False,
    yscale: str | None = "",
    hide_errors: bool | None = None,
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:

    # separate only morphed and corresponding unmorphed histograms
    morphed_hists = OrderedDict()
    for key, hist in hists.items():
        if "morphed" in key.name:
            morphed_hists[key] = hist

    if not morphed_hists:
        raise ValueError("No morphed histograms found in input hists.")

    if len(morphed_hists.keys()) > 1:
        raise ValueError("More than one morphed histogram found in input hists. Not implemented")

    non_morphed_hist = OrderedDict()
    for key, hist in hists.items():
        if list(morphed_hists.keys())[0].name.replace("_morphed", "") == key.name:
            non_morphed_hist[key] = hist

    if len(morphed_hists.keys()) != len(non_morphed_hist.keys()):
        raise ValueError("Number of morphed and non-morphed histograms do not match.")

    # modify standard styles
    for key, hist in morphed_hists.items():
        key.color1 = config_inst.x.colors.red
        setattr(key, "unstack", True)

    # merge morphed and non-morphed histograms into hists
    hists_to_plot = morphed_hists
    hists_to_plot.update(non_morphed_hist)

    hists = hists_to_plot

    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    blinding_threshold = kwargs.get("blinding_threshold", None)

    if blinding_threshold:
        hists = blind_sensitive_bins(hists, config_inst, blinding_threshold)
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_process_settings(hists, process_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = prepare_plot_config_custom_ratio(
        hists,
        shape_norm=shape_norm,
        hide_errors=hide_errors,
    )

    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)
    if shape_norm:
        style_config["ax_cfg"]["ylabel"] = r"$\Delta N/N$"

    return plot_all(plot_config, style_config, **kwargs)


def plot_bin_morphing(
    function_bin_search: Callable,
    bin_type: str,
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool | None = False,
    yscale: str | None = "",
    hide_errors: bool | None = None,
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:

    # separate morphed, true and guidance points histograms
    morphed_hists = OrderedDict()
    for key, hist in hists.items():
        if "morphed" in key.name:
            morphed_hists[key] = hist

    if not morphed_hists:
        raise ValueError("No morphed histograms found in input hists.")

    if len(morphed_hists.keys()) > 1:
        raise ValueError("More than one morphed histogram found in input hists. Not implemented")

    # non morphed
    non_morphed_hist = OrderedDict()
    for key, hist in hists.items():
        if list(morphed_hists.keys())[0].name.replace("_morphed", "") == key.name:
            non_morphed_hist[key] = hist

    if len(morphed_hists.keys()) != len(non_morphed_hist.keys()):
        raise ValueError("Number of morphed and non-morphed histograms do not match.")

    # guidance point histograms
    guidance_hists = OrderedDict()
    for key, hist in hists.items():
        not_in_non_morphed_hists = (key.name != list(non_morphed_hist.keys())[0].name)
        not_in_morphed_hists = (key.name != list(morphed_hists.keys())[0].name)
        if ("hh_ggf_hbb_htt_kl" in key.name) and not_in_non_morphed_hists and not_in_morphed_hists:
            guidance_hists[key] = hist

    if len(guidance_hists.keys()) != 3:
        raise ValueError("Number of guidance point histograms is not 3.")

    # get guidance points
    guidance_points_str = [key.name.replace("hh_ggf_hbb_htt_kl", "") for key in guidance_hists.keys()]
    guidance_points_str = [point.replace("_kt1", "") for point in guidance_points_str]
    guidance_points = [float(point.replace("p", ".")) for point in guidance_points_str]

    # choose bin
    chosen_bin = function_bin_search(list(morphed_hists.values())[0].values())

    # get value of chosen bin for each histogram
    morphed_chosen_bin = np.array([
        list(morphed_hists.values())[0].values()[..., chosen_bin][0],
        list(morphed_hists.values())[0].variances()[..., chosen_bin][0],
    ])

    non_morphed_chosen_bin = np.array([
        list(non_morphed_hist.values())[0].values()[..., chosen_bin][0],
        list(non_morphed_hist.values())[0].variances()[..., chosen_bin][0],
    ])

    guidance_chosen_bins_values = np.array([
        list(guidance_hists.values())[i].values()[..., chosen_bin][0] for i in range(3)
    ])

    guidance_chosen_bins_variances = np.array([
        list(guidance_hists.values())[i].variances()[..., chosen_bin][0] for i in range(3)
    ])

    # fit a parabola to the guidance points
    from scipy.optimize import curve_fit

    def parabola(x, a, b, c):
        return a * x**2 + b * x + c

    popt, pcov = curve_fit(parabola, guidance_points, guidance_chosen_bins_values, sigma=guidance_chosen_bins_variances)

    # plot the parabola
    fig, ax = plt.subplots()
    axs = (ax,)
    mplhep.style.use("CMS")

    x = np.linspace(-1, 6, 150)
    y = parabola(x, *popt)

    ax.plot(x, y, label="Parabola fit", color="black")

    # plot the guidance points
    ax.errorbar(
        guidance_points,
        guidance_chosen_bins_values,
        yerr=np.sqrt(guidance_chosen_bins_variances),
        fmt="bo",
        label="Guidance points",
    )

    # plot the chosen bin values for morphed and non-morphed histograms
    morphed_hist_name = list(morphed_hists.keys())[0].name
    morphed_point = float(morphed_hist_name.replace("hh_ggf_hbb_htt_kl", "").replace("_kt1_morphed", "").replace("p", "."))  # noqa
    ax.errorbar(morphed_point, morphed_chosen_bin[0], yerr=np.sqrt(morphed_chosen_bin[1]), fmt="r+", label="Morphed")
    ax.errorbar(morphed_point, non_morphed_chosen_bin[0], yerr=np.sqrt(non_morphed_chosen_bin[1]), fmt="go", label="True value")  # noqa

    ax.set_xlabel("kl")
    ax.set_ylabel(bin_type + " value")
    ax.legend()
    return fig, axs


plot_max_bin_morphing = partial(plot_bin_morphing, function_bin_search=np.argmax, bin_type="Max bin")
plot_min_bin_morphing = partial(plot_bin_morphing, function_bin_search=np.argmin, bin_type="Min bin")
