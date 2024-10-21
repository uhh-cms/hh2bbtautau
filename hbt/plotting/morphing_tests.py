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
    get_position,
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

    # build ratio_kwargs: morphed divided by standard, error bars are the propagated ratio errors
    if len(line_hists) == 2:
        if "morphed" in plot_config["line_0"]["kwargs"]["label"]:
            i = 1
            j = 0
        else:
            i = 0
            j = 1
        plot_config[f"line_{j}"]["ratio_kwargs"] = {
            "norm": line_hists[i].values(),
            # yerr obtained by error propagation of the ratio
            "yerr": np.sqrt(
                (line_hists[j].values()**2) * line_hists[i].variances() / (line_hists[i].values()**4) +
                line_hists[j].variances() / (line_hists[i].values()**2)),
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
        if list(morphed_hists.keys())[0].name.replace("_morphed", "").replace("_exact", "").replace("_average", "").replace("_fit", "") == key.name:  # noqa
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

    # add custom ylim params to style_config due to partially huge error bars
    whitespace_fraction = 0.3
    magnitudes = 4

    log_y = style_config.get("ax_cfg", {}).get("yscale", "linear") == "log"

    hists_and_norms = []
    for params in plot_config.values():
        if "hist" in params:
            h = params["hist"]
        if "kwargs" in params and "norm" in params["kwargs"]:
            norm = params["kwargs"]["norm"]
        else:
            norm = 1
        hists_and_norms.append((h, norm))

    max_important_value = np.max([np.max(h.values() / norm) for h, norm in hists_and_norms])

    ax_ymin = max_important_value / 10**magnitudes if log_y else 0.0000001
    # ax_ymin = 0.0000001
    ax_ymax = get_position(ax_ymin, max_important_value, factor=1 / (1 - whitespace_fraction), logscale=log_y)
    # ax_ymax = 0.0014  # 0.085
    style_config["ax_cfg"]["ylim"] = (ax_ymin, ax_ymax)
    style_config["rax_cfg"]["ylim"] = (0.41, 1.59)
    style_config["rax_cfg"]["ylabel"] = "Ratio"

    # style_config["ax_cfg"][""] = "upper right"
    # ax.tick_params(axis="both", which="major", labelsize=16)

    style_config["legend_cfg"]["facecolor"] = "white"
    style_config["legend_cfg"]["edgecolor"] = "black"
    style_config["legend_cfg"]["framealpha"] = 0.8
    style_config["legend_cfg"]["frameon"] = True
    style_config["legend_cfg"]["fontsize"] = 22
    style_config["annotate_cfg"] = {}
    style_config["cms_label_cfg"]["fontsize"] = 28

    # from IPython import embed; embed(header="making plot from hists")

    return plot_all(plot_config, style_config, **kwargs)


def plot_bin_morphing(
    function_bin_search: Callable,
    bin_type: str,
    production_channel: str,
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

    variable_inst = variable_insts[0]
    averaged = False
    fitted = False

    # separate morphed, true and guidance points histograms
    morphed_hists = OrderedDict()
    for key, hist in hists.items():
        if "morphed" in key.name:
            morphed_hists[key] = hist
            if "average" in key.name:
                averaged = True
            if "fit" in key.name:
                fitted = True

    if not morphed_hists:
        raise ValueError("No morphed histograms found in input hists.")

    # if len(morphed_hists.keys()) > 1:
    #     raise ValueError("More than one morphed histogram found in input hists. Not implemented")

    if not averaged and not fitted:
        # non morphed
        non_morphed_hist = OrderedDict()
        for key, hist in hists.items():
            if list(morphed_hists.keys())[0].name.replace("_morphed", "").replace("_exact", "") == key.name:
                non_morphed_hist[key] = hist

        # if len(morphed_hists.keys()) != len(non_morphed_hist.keys()):
        #     raise ValueError("Number of morphed and non-morphed histograms do not match.")

        # guidance point histograms
        guidance_hists = OrderedDict()
        if production_channel == "ggf":
            for key, hist in hists.items():
                not_in_non_morphed_hists = (key.name != list(non_morphed_hist.keys())[0].name)
                not_in_morphed_hists = (key.name not in [list(morphed_hists.keys())[i].name for i in range(len(morphed_hists.keys()))])  # noqa
                if "hh_ggf_hbb_htt_kl" in key.name and not_in_non_morphed_hists and not_in_morphed_hists:
                    guidance_hists[key] = hist
            if len(guidance_hists.keys()) != 3:
                raise ValueError("Number of guidance point histograms is not 3.")
        elif production_channel == "vbf":
            for key, hist in hists.items():
                not_in_non_morphed_hists = (key.name != list(non_morphed_hist.keys())[0].name)
                not_in_morphed_hists = (key.name not in [list(morphed_hists.keys())[i].name for i in range(len(morphed_hists.keys()))])  # noqa

                if "hh_vbf_hbb_htt_kv" in key.name and not_in_non_morphed_hists and not_in_morphed_hists:
                    guidance_hists[key] = hist
            if len(guidance_hists.keys()) != 6:
                raise ValueError("Number of guidance point histograms is not 6.")
    else:
        # guidance point histograms
        guidance_hists = OrderedDict()
        if production_channel == "ggf":
            for key, hist in hists.items():
                if "hh_ggf_hbb_htt_kl" in key.name and "morphed" not in key.name:
                    guidance_hists[key] = hist
            if len(guidance_hists.keys()) < 4:
                raise ValueError("Number of guidance point histograms is under 4.")
        elif production_channel == "vbf":
            for key, hist in hists.items():
                if "hh_vbf_hbb_htt_kv" in key.name and "morphed" not in key.name:
                    guidance_hists[key] = hist
                if len(guidance_hists.keys()) < 7:
                    raise ValueError("Number of guidance point histograms is under 7.")

    # get guidance points
    if production_channel == "ggf":
        guidance_points_str = [key.name.replace("hh_ggf_hbb_htt_kl", "") for key in guidance_hists.keys()]
        guidance_points_str = [point.replace("_kt1", "") for point in guidance_points_str]
        guidance_points = [float(point.replace("p", ".").replace("m", "-")) for point in guidance_points_str]

    elif production_channel == "vbf":
        raise NotImplementedError("VBF not implemented yet")

    # choose bin
    chosen_bin = function_bin_search(list(morphed_hists.values())[0].values())

    # get value of chosen bin for each histogram
    morphed_chosen_bins = []
    for key, hist in morphed_hists.items():
        morphed_chosen_bins.append(np.array([
            hist.values()[..., chosen_bin][0],
            hist.variances()[..., chosen_bin][0],
        ]))
    morphed_chosen_bins = np.array(morphed_chosen_bins)
    # morphed_chosen_bin = np.array([
    #     list(morphed_hists.values())[0].values()[..., chosen_bin][0],
    #     list(morphed_hists.values())[0].variances()[..., chosen_bin][0],
    # ])

    if not averaged and not fitted:
        non_morphed_chosen_bin = np.array([
            list(non_morphed_hist.values())[0].values()[..., chosen_bin][0],
            list(non_morphed_hist.values())[0].variances()[..., chosen_bin][0],
        ])

    guidance_chosen_bins_values = np.array([
        list(guidance_hists.values())[i].values()[..., chosen_bin][0] for i in range(len(guidance_points))
    ])

    guidance_chosen_bins_variances = np.array([
        list(guidance_hists.values())[i].variances()[..., chosen_bin][0] for i in range(len(guidance_points))
    ])

    if production_channel == "ggf":
        # fit a parabola to the guidance points
        from scipy.optimize import curve_fit

        def parabola(x, a, b, c):
            return a * x**2 + b * x + c

    if production_channel == "vbf":
        raise NotImplementedError("VBF not implemented yet")

    popt, pcov = curve_fit(
        parabola,
        guidance_points,
        guidance_chosen_bins_values,
        sigma=np.sqrt(guidance_chosen_bins_variances),
        p0=[0, 0, 0],
    )

    # plot the parabola
    fig, ax = plt.subplots()
    axs = (ax,)
    plt.style.use(mplhep.style.CMS)
    # mplhep.style.use("CMS")

    x = np.linspace(-1, 6, 150)
    y = parabola(x, *popt)

    if not averaged and not fitted:
        ax.plot(x, y, label="Parabola fit", color="black")
    else:
        ax.plot(x, y, label="Parabola fit with uncertainty", color="black")

    # calculate the chi2 of the fit
    chi2 = np.sum((parabola(np.array(guidance_points), *popt) -
        np.array(guidance_chosen_bins_values))**2 / np.array(guidance_chosen_bins_variances))
    ndf = len(guidance_points) - 3
    chi2_ndf = chi2 / ndf

    # calculate the probability of the chi2
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, ndf)

    # commented out
    # # put the chi2 and p-value in the plot
    # ax.text(
    #     0.01,
    #     1.01,
    #     f"chi^2 / ndf = {chi2_ndf:.2f} \n p-value = {p_value:.2f}",
    #     horizontalalignment="left",
    #     verticalalignment="bottom",
    #     fontsize=12,
    #     transform=ax.transAxes,
    # )

    if averaged or fitted:
        # fit parabola unweighted
        popt_unweighted, pcov_unweighted = curve_fit(
            parabola,
            guidance_points,
            guidance_chosen_bins_values,
            p0=[0, 0, 0],
        )

        y_unweighted = parabola(x, *popt_unweighted)
        ax.plot(x, y_unweighted, label="Parabola fit w/o uncertainty", color="orange")

        # calculate the chi2 of the fit
        chi2_unweighted = np.sum((parabola(np.array(guidance_points), *popt_unweighted) -
            np.array(guidance_chosen_bins_values))**2 / np.array(guidance_chosen_bins_variances))
        chi2_ndf_unweighted = chi2_unweighted / ndf

        # calculate the probability of the chi2
        p_value_unweighted = 1 - chi2_dist.cdf(chi2_unweighted, ndf)

        # commented out
        # # put the chi2 and p-value in the plot
        # ax.text(
        #     0.01,
        #     0.99,
        #     f"Without uncertainties: chi^2 / ndf = {chi2_ndf_unweighted:.2f} \n p-value = {p_value_unweighted:.2f}",
        #     horizontalalignment="left",
        #     verticalalignment="top",
        #     fontsize=12,
        #     transform=ax.transAxes,
        # )

    # if not averaged and not fitted:

    #     # fit a parabola to the four true points
    #     # get fourth point from non-morphed histogram
    #     non_morphed_hist_name = list(non_morphed_hist.keys())[0].name
    #     if production_channel == "ggf":
    #         non_morphed_point = float(non_morphed_hist_name.replace("hh_ggf_hbb_htt_kl", "").replace("_kt1", "").replace("p", "."))  # noqa
    #     elif production_channel == "vbf":
    #         raise NotImplementedError("VBF not implemented yet")
    #     popt_2, pcov_2 = curve_fit(
    #         parabola,
    #         [non_morphed_point, *guidance_points],
    #         [non_morphed_chosen_bin[0], *guidance_chosen_bins_values],
    #         sigma=np.sqrt(np.array([non_morphed_chosen_bin[1], *guidance_chosen_bins_variances])),
    #         p0=[0, 0, 0],
    #     )

    #     y_2 = parabola(x, *popt_2)
    #     ax.plot(x, y_2, label="Parabola fit with 4 values", color="orange")

    #     # calculate the chi2 of the fit
    #     chi2_2 = np.sum((parabola(np.array([non_morphed_point, *guidance_points]), *popt_2) -
    #         np.array([non_morphed_chosen_bin[0], *guidance_chosen_bins_values]))**2 /
    #         np.array([non_morphed_chosen_bin[1], *guidance_chosen_bins_variances]))
    #     ndf_2 = len(guidance_points) + 1 - 3
    #     chi2_ndf_2 = chi2_2 / ndf_2

    #     # calculate the probability of the chi2
    #     p_value_2 = 1 - chi2_dist.cdf(chi2_2, ndf_2)

    #     # commented out
    #     # # put the chi2 and p-value in the plot
    #     # ax.text(
    #     #     0.01,
    #     #     0.99,
    #     #     f"4 points: chi^2 / ndf = {chi2_ndf_2:.2f} \n p-value = {p_value_2:.2f}",
    #     #     horizontalalignment="left",
    #     #     verticalalignment="top",
    #     #     fontsize=12,
    #     #     transform=ax.transAxes,
    #     # )

    #     # same result as above

    #     # # fit a parabola with 4 points but through the polyfit function
    #     # coefs = np.polyfit(
    #     #     [non_morphed_point, *guidance_points],
    #     #     [non_morphed_chosen_bin[0], *guidance_chosen_bins_values],
    #     #     2,
    #     #     w=1 / np.sqrt(np.array([non_morphed_chosen_bin[1], *guidance_chosen_bins_variances])),
    #     # )

    #     # y_3 = np.polyval(coefs, x)
    #     # ax.plot(x, y_3, label="Polyfit with 4 values", color="green")

    # plot the guidance points
    ax.errorbar(
        guidance_points,
        guidance_chosen_bins_values,
        yerr=np.sqrt(guidance_chosen_bins_variances),
        fmt="bo",
        label="Guidance points",
    )

    # plot the chosen bin values for morphed and non-morphed histograms
    if averaged:
        label_ = "4-point morphed"
    else:
        label_ = "Morphed"
    if fitted:
        label_ = "Fit morphed"
    morphed_points = []
    # label_list = []
    i = 0
    for key, hist in morphed_hists.items():
        morphed_hist_name = list(morphed_hists.keys())[i].name
        if "exact" in morphed_hist_name:
            morphed_hist_name = morphed_hist_name.replace("_exact", "")
        if "average" in morphed_hist_name:
            morphed_hist_name = morphed_hist_name.replace("_average", "")
        if "fit" in morphed_hist_name:
            morphed_hist_name = morphed_hist_name.replace("_fit", "")
        if production_channel == "ggf":
            morphed_point = float(morphed_hist_name.replace("hh_ggf_hbb_htt_kl", "").replace("_kt1_morphed", "").replace("p", "."))  # noqa
        elif production_channel == "vbf":
            raise NotImplementedError("VBF not implemented yet")
        morphed_points.append(morphed_point)
        i += 1
        # label_list.append(label_ + " kl " + str(morphed_point))
    ax.errorbar(
        morphed_points,
        morphed_chosen_bins[:, 0],
        yerr=np.sqrt(morphed_chosen_bins[:, 1]),
        fmt="rs",
        markerfacecolor="none",
        ms=10,
        markeredgecolor="red",
        label=label_,
    )
    if not averaged and not fitted:
        non_morphed_hist_name = list(non_morphed_hist.keys())[0].name
        if production_channel == "ggf":
            non_morphed_point = float(non_morphed_hist_name.replace("hh_ggf_hbb_htt_kl", "").replace("_kt1", "").replace("p", "."))  # noqa
        else:
            raise NotImplementedError("VBF not implemented yet")
        ax.errorbar(
            non_morphed_point,
            non_morphed_chosen_bin[0],
            yerr=np.sqrt(non_morphed_chosen_bin[1]),
            fmt="go",
            label="True value",
        )

    # commented out
    # # add text with the chosen bin value and the variable name on the top right corner
    # ax.text(
    #     0.99,
    #     1.01,
    #     f"Bin number: {chosen_bin} \n for variable {variable_inst.name}",
    #     horizontalalignment="right",
    #     verticalalignment="bottom",
    #     fontsize=12,
    #     transform=ax.transAxes,
    # )

    # put the cms logo and the lumi text on the top left corner
    mplhep.cms.text(text="Private Work", fontsize=16, ax=ax)
    # ax.text(
    #     0.01,
    #     1.01,
    #     "CMS",
    #     horizontalalignment="left",
    #     verticalalignment="bottom",
    #     fontsize=18,
    #     weight="bold",
    #     transform=ax.transAxes,
    # )
    # ax.text(
    #     0.13,
    #     1.01,
    #     "Private work",
    #     horizontalalignment="left",
    #     verticalalignment="bottom",
    #     fontsize=16,
    #     transform=ax.transAxes,
    # )

    ax.set_xlabel(r"$\kappa_\lambda$", fontsize=16)
    # ax.set_ylabel(bin_type + " value", fontsize=16)
    ax.set_ylabel("Bin yield", fontsize=16)
    ax.legend(fontsize=16, loc="upper center")
    ax.tick_params(axis="both", which="major", labelsize=16)
    fig.tight_layout()
    return fig, axs


def plot_ratios(
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

    variable_inst = variable_insts[0]

    # separate morphed and non morphed histograms, but in same order
    morphed_hists = OrderedDict()
    for key, hist in hists.items():
        if "morphed" in key.name:
            morphed_hists[key] = hist

    if not morphed_hists:
        raise ValueError("No morphed histograms found in input hists.")

    non_morphed_hists = OrderedDict()
    for morphed_key in morphed_hists.keys():
        for key, hist in hists.items():
            if morphed_key.name.replace("_morphed", "").replace("_exact", "").replace("_average", "").replace("_fit", "") == key.name:  # noqa
                non_morphed_hists[key] = hist

    from IPython import embed; embed(header="making plot from hists")

    if len(morphed_hists.keys()) != len(non_morphed_hists.keys()):
        raise ValueError("Number of morphed and non-morphed histograms do not match.")

    # get values from histograms and calculate ratios with uncertainties
    ratios = OrderedDict()
    uncertainties = OrderedDict()
    for imorphed_key, morphed_key in enumerate(morphed_hists.keys()):
        morphed_hist = morphed_hists[morphed_key]
        non_morphed_hist = list(non_morphed_hists.values())[imorphed_key]
        ratio = morphed_hist / non_morphed_hist.values()
        ratios[morphed_key] = ratio
        uncertainties[morphed_key] = np.sqrt(
            (morphed_hist.variances() / non_morphed_hist.values()**2) +
            (non_morphed_hist.variances() * morphed_hist.values()**2 / non_morphed_hist.values()**4))

    # plot the ratios
    fig, ax = plt.subplots()
    axs = (ax,)
    mplhep.style.use("CMS")

    for key, ratio in ratios.items():
        ax.errorbar(
            ratio.axes[1].centers,
            ratio.values()[0],
            yerr=uncertainties[key][0],
            fmt="-o",
            label=key.name + " / non-morphed",
        )

    # ax.set_ylim(0.41, 1.59)
    ax.set_xlabel(variable_inst.name)
    ax.set_ylabel("Ratio")
    ax.legend(fontsize=12)

    # style_config["legend_cfg"]["facecolor"] = "white"
    # style_config["legend_cfg"]["edgecolor"] = "black"
    # style_config["legend_cfg"]["framealpha"] = 0.8
    # style_config["legend_cfg"]["frameon"] = True
    return fig, axs

def plot_3d_morphing(
    function_bin_search: Callable,
    points: dict,
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
    distance_measure: str = "chi2",
    **kwargs,
) -> plt.Figure:



    fig, ax = plt.subplots()
    axs = (ax,)
    plt.style.use(mplhep.style.CMS)
    # mplhep.style.use("CMS")

    # put the cms logo and the lumi text on the top left corner
    mplhep.cms.text(text="Private Work", fontsize=16, ax=ax)

    ax.set_xlabel(r"$\kappa_\lambda$", fontsize=16)
    # ax.set_ylabel(bin_type + " value", fontsize=16)
    ax.set_ylabel(r"$\kappa_{2V}$", fontsize=16)
    ax.legend(fontsize=16, loc="upper center")
    ax.tick_params(axis="both", which="major", labelsize=16)
    fig.tight_layout()
    return fig, axs


plot_max_bin_morphing = partial(
    plot_bin_morphing,
    function_bin_search=np.argmax,
    bin_type="Max bin",
    production_channel="ggf",
)
plot_min_bin_morphing = partial(
    plot_bin_morphing,
    function_bin_search=np.argmin,
    bin_type="Min bin",
    production_channel="ggf",
)
plot_bin_5_morphing = partial(
    plot_bin_morphing,
    function_bin_search=lambda x: 5,
    bin_type="Bin 5",
    production_channel="ggf",
)

plot_3d_morphing_2022_pre_chi2_sm_morphed = partial(
    plot_3d_morphing,
    function_bin_search=lambda x: 5,
    points={
        {"kv": 1., "k2v": 1., "kl": 1., "name"= "kv1_k2v1_kl1", "type"="morphed"},
        {"kv": 1., "k2v": 0., "kl": 1., "name"= "kv1_k2v0_kl1", "type"=""},
        {"kv": 1., "k2v": 1., "kl": 2., "name": "kv1_k2v1_kl2", "type"=""},
        {"kv": 1., "k2v": 2., "kl": 1., "name": "kv1_k2v2_kl1", "type"=""},
        {"kv": 1.74, "k2v": 1.37, "kl": 14.4, "name": "kv1p74_k2v1p37_kl14p4", "type"=""},
        {"kv": -0.758, "k2v": 1.44, "kl": -19.3, "name": "kvm0p758_k2v1p44_klm19p3", "type"=""},
        {"kv": -0.012, "k2v": 0.03, "kl": 10.2, "name": "kvm0p012_k2v0p03_kl10p2", "type"=""},
        {"kv": -0.962, "k2v": 0.959, "kl": -1.43, "name": "kvm0p962_k2v0p959_klm1p43", "type"=""},
        {"kv": -1.21, "k2v": 1.94, "kl": -0.94, "name": "kvm1p21_k2v1p94_klm0p94", "type"=""},
        {"kv": -1.6, "k2v": 2.72, "kl": -1.36, "name": "kvm1p6_k2v2p72_klm1p36", "type"=""},
        {"kv": -1.83, "k2v": 3.57, "kl": -3.39, "name": "kvm1p83_k2v3p57_klm3p39", "type"=""},
        {"kv": -2.12, "k2v": 3.87, "kl": -5.96, "name": "kvm2p12_k2v3p87_klm5p96", "type"=""},
    },
    distance_measure="chi2",
)

plot_3d_morphing_2022_pre_ratio_sm_morphed = partial(
    plot_3d_morphing,
    function_bin_search=lambda x: 5,
    points={
        {"kv": 1., "k2v": 1., "kl": 1., "name"= "kv1_k2v1_kl1", "type"="morphed"},
        {"kv": 1., "k2v": 0., "kl": 1., "name"= "kv1_k2v0_kl1", "type"=""},
        {"kv": 1., "k2v": 1., "kl": 2., "name": "kv1_k2v1_kl2", "type"=""},
        {"kv": 1., "k2v": 2., "kl": 1., "name": "kv1_k2v2_kl1", "type"=""},
        {"kv": 1.74, "k2v": 1.37, "kl": 14.4, "name": "kv1p74_k2v1p37_kl14p4", "type"=""},
        {"kv": -0.758, "k2v": 1.44, "kl": -19.3, "name": "kvm0p758_k2v1p44_klm19p3", "type"=""},
        {"kv": -0.012, "k2v": 0.03, "kl": 10.2, "name": "kvm0p012_k2v0p03_kl10p2", "type"=""},
        {"kv": -0.962, "k2v": 0.959, "kl": -1.43, "name": "kvm0p962_k2v0p959_klm1p43", "type"=""},
        {"kv": -1.21, "k2v": 1.94, "kl": -0.94, "name": "kvm1p21_k2v1p94_klm0p94", "type"=""},
        {"kv": -1.6, "k2v": 2.72, "kl": -1.36, "name": "kvm1p6_k2v2p72_klm1p36", "type"=""},
        {"kv": -1.83, "k2v": 3.57, "kl": -3.39, "name": "kvm1p83_k2v3p57_klm3p39", "type"=""},
        {"kv": -2.12, "k2v": 3.87, "kl": -5.96, "name": "kvm2p12_k2v3p87_klm5p96", "type"=""},
    },
    distance_measure="ratio",
)

plot_3d_morphing_2022_pre_chi2_all_morphed = partial(
    plot_3d_morphing,
    function_bin_search=lambda x: 5,
    points={
        {"kv": 1., "k2v": 1., "kl": 1., "name"= "kv1_k2v1_kl1", "type"="morphed"},
        {"kv": 1., "k2v": 0., "kl": 1., "name"= "kv1_k2v0_kl1", "type"="morphed"},
        {"kv": 1., "k2v": 1., "kl": 2., "name": "kv1_k2v1_kl2", "type"="morphed"},
        {"kv": 1., "k2v": 2., "kl": 1., "name": "kv1_k2v2_kl1", "type"="morphed"},
        {"kv": 1.74, "k2v": 1.37, "kl": 14.4, "name": "kv1p74_k2v1p37_kl14p4", "type"="morphed"},
        {"kv": -0.758, "k2v": 1.44, "kl": -19.3, "name": "kvm0p758_k2v1p44_klm19p3", "type"="morphed"},
        {"kv": -0.012, "k2v": 0.03, "kl": 10.2, "name": "kvm0p012_k2v0p03_kl10p2", "type"="morphed"},
        {"kv": -0.962, "k2v": 0.959, "kl": -1.43, "name": "kvm0p962_k2v0p959_klm1p43", "type"="morphed"},
        {"kv": -1.21, "k2v": 1.94, "kl": -0.94, "name": "kvm1p21_k2v1p94_klm0p94", "type"="morphed"},
        {"kv": -1.6, "k2v": 2.72, "kl": -1.36, "name": "kvm1p6_k2v2p72_klm1p36", "type"="morphed"},
        {"kv": -1.83, "k2v": 3.57, "kl": -3.39, "name": "kvm1p83_k2v3p57_klm3p39", "type"="morphed"},
        {"kv": -2.12, "k2v": 3.87, "kl": -5.96, "name": "kvm2p12_k2v3p87_klm5p96", "type"="morphed"},
    },
    distance_measure="chi2",
)
