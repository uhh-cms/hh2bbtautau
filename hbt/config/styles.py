# coding: utf-8

"""
Style definitions.
"""

from __future__ import annotations

import re
from collections import defaultdict

import order as od

from columnflow.util import DotDict, try_int
from columnflow.types import Callable


def setup_plot_styles(config: od.Config) -> None:
    """
    Setup plot styles.
    """
    # general settings
    config.x.default_general_settings = {
        "cms_label": "wip", "whitespace_fraction": 0.31,
    }

    # default component configs
    gridspec = {
        "height_ratios": [3, 0.9],
    }
    legend = {
        "borderpad": 0, "borderaxespad": 1.2, "columnspacing": 1.8, "labelspacing": 0.28, "fontsize": 16,
        "cf_line_breaks": True, "cf_short_labels": False,
    }
    ratio = {
        "yloc": "center",
    }
    annotate = {
        "fontsize": 18, "style": "italic", "xycoords": "axes fraction", "xy": (0.035, 0.955),
    }

    # wide legend
    # - 3 columns, backgrounds in first 2 columns
    # - shortened process labels
    # - changed annotation (channel) position to fit right under legend
    wide_legend = legend | {
        "ncols": 3, "loc": "upper left", "cf_entries_per_column": legend_entries_per_column, "cf_short_labels": True,
    }
    annotate_wide = annotate | {
        "xy": (0.035, 0.765),
    }

    # wide extended legend, same as wide legend except
    # - process labels are not shortened
    # - annotation (channel) moved slightly down to fut under (now taller) legend
    wide_ext_legend = wide_legend | {
        "cf_short_labels": False,
    }
    annotate_wide_ext = annotate_wide | {
        "xy": (0.035, 0.750),
    }

    # construct named style configs
    config.x.custom_style_config_groups = {
        "default": (default_cfg := {
            "gridspec_cfg": gridspec,
            "rax_cfg": ratio,
            "legend_cfg": legend,
            "annotate_cfg": annotate,
        }),
        "wide_legend": (wide_legend_cfg := {
            **default_cfg,
            "legend_cfg": wide_legend,
            "annotate_cfg": annotate_wide,
        }),
        "wide_ext_legend": {
            **wide_legend_cfg,
            "legend_cfg": wide_ext_legend,
            "annotate_cfg": annotate_wide_ext,
        },
    }

    config.x.default_custom_style_config = "wide_legend"
    config.x.default_blinding_threshold = 0


def stylize_processes(config: od.Config) -> None:
    """
    Adds process colors and adjust labels.
    """
    cfg = config

    # recommended cms colors
    # see https://cms-analysis.docs.cern.ch/guidelines/plotting/colors
    cfg.x.colors = DotDict(
        bright_blue="#3f90da",
        dark_blue="#011c87",
        purple="#832db6",
        aubergine="#964a8b",
        yellow="#f7c331",
        bright_orange="#ffa90e",
        dark_orange="#e76300",
        red="#bd1f01",
        teal="#92dadd",
        grey="#94a4a2",
        brown="#a96b59",
        green="#30c300",
        dark_green="#269c00",
    )

    ggf_colors = {
        "0": cfg.x.colors.bright_orange,
        "1": cfg.x.colors.dark_blue,
        "2p45": cfg.x.colors.red,
        "5": cfg.x.colors.green,
    }

    for kl in ["0", "1", "2p45", "5"]:
        if (p := config.get_process(f"hh_ggf_hbb_htt_kl{kl}_kt1", default=None)):
            # p.color1 = cfg.x.colors.dark_blue
            p.color1 = ggf_colors.get(kl, cfg.x.colors.dark_blue)
            kappa_label = create_kappa_label(**{r"\lambda": kl, "t": "1"}, group=False)
            p.label = rf"$HH_{{ggf}} \rightarrow bb\tau\tau$ __SCALE____SHORT____BREAK__({kappa_label})"

    for kv, k2v, kl in [
        ("1", "1", "1"),
        ("1", "0", "1"),
        ("1", "2", "1"),
        ("1", "1", "2"),
        ("1p74", "1p37", "14p4"),
        ("m0p012", "0p03", "10p2"),
        ("m0p758", "1p44", "m19p3"),
        ("m0p962", "0p959", "m1p43"),
        ("m1p21", "1p94", "m0p94"),
        ("m1p6", "2p72", "m1p36"),
        ("m1p83", "3p57", "m3p39"),
        ("m2p12", "3p87", "m5p96"),
    ]:
        if (p := config.get_process(f"hh_vbf_hbb_htt_kv{kv}_k2v{k2v}_kl{kl}", default=None)):
            p.color1 = cfg.x.colors.brown
            kappa_label = create_kappa_label(**{"2V": k2v, r"\lambda": kl, "V": kv})
            p.label = rf"$HH_{{vbf}} \rightarrow bb\tau\tau$ __SCALE____SHORT____BREAK__({kappa_label})"

    if (p := config.get_process("h", default=None)):
        p.color1 = cfg.x.colors.teal

    if (p := config.get_process("tt", default=None)):
        p.color1 = cfg.x.colors.bright_orange
        p.label = r"$t\bar{t}$"

    if (p := config.get_process("st", default=None)):
        p.color1 = cfg.x.colors.purple

    if (p := config.get_process("dy", default=None)):
        p.color1 = cfg.x.colors.bright_blue

    if (p := config.get_process("vv", default=None)):
        p.color1 = cfg.x.colors.yellow

    if (p := config.get_process("vvv", default=None)):
        p.color1 = cfg.x.colors.yellow

    if (p := config.get_process("multiboson", default=None)):
        p.color1 = cfg.x.colors.yellow

    if (p := config.get_process("w", default=None)):
        p.color1 = cfg.x.colors.aubergine
        p.label = "W"

    if (p := config.get_process("z", default=None)):
        p.color1 = cfg.x.colors.aubergine
        p.label = "Z"

    if (p := config.get_process("v", default=None)):
        p.color1 = cfg.x.colors.aubergine

    if (p := config.get_process("all_v", default=None)):
        p.color1 = cfg.x.colors.aubergine

    if (p := config.get_process("ewk", default=None)):
        p.color1 = cfg.x.colors.dark_orange

    if (p := config.get_process("ttv", default=None)):
        p.color1 = cfg.x.colors.grey
        p.label = r"$t\bar{t} + V$"

    if (p := config.get_process("ttvv", default=None)):
        p.color1 = cfg.x.colors.grey
        p.label = r"$t\bar{t} + VV$"

    if (p := config.get_process("tt_multiboson", default=None)):
        p.color1 = cfg.x.colors.grey

    if (p := config.get_process("qcd", default=None)):
        p.color1 = cfg.x.colors.red


def legend_entries_per_column(ax, handles: list, labels: list, n_cols: int) -> list[int]:
    """
    Controls number of entries such that backgrounds are in the first n - 1 columns, and everything
    else in the last one.
    """
    # get number of background and remaining entries
    n_backgrounds = sum(1 for handle in handles if handle.__class__.__name__ == "StepPatch")
    n_other = len(handles) - n_backgrounds

    # fill number of entries per column
    entries_per_col = n_cols * [0]
    n_bkg_cols = n_cols
    # set last column if non-backgrounds are present
    if n_other:
        entries_per_col[-1] = n_other
        n_bkg_cols -= 1
    # fill background columns
    for i in range(n_bkg_cols):
        entries_per_col[i] = n_backgrounds // n_bkg_cols + (n_backgrounds % n_bkg_cols > i)

    return entries_per_col


def update_handles_labels_factory(remove_mc_stat_label: bool = False) -> Callable:
    """
    Factory to generate a function that updates legend handles and labels given some conditions passed as arguments.
    """
    def remove_mc_stat_label(handles: list, labels: list) -> None:
        for i, label in enumerate(labels):
            if re.match(r"^MC stat\.? unc.*$", label):
                labels.pop(i)
                handles.pop(i)
                break

    # actual update function
    def update_handles_labels(ax, handles: list, labels: list, ncols: int) -> None:
        if remove_mc_stat_label:
            remove_mc_stat_label(handles, labels)

    return update_handles_labels


def kappa_str_to_num(value: str) -> int | float:
    """
    Converts a string-encoded kappa value to an actual number. An integer is returned if possible,
    and a float otherwise. Examples:

    .. code-block:: python

        kappa_str_to_num("1")     # 1
        kappa_str_to_num("2.45")  # 2.45
        kappa_str_to_num("m1p7")  # -1.7
    """
    value = value.replace("p", ".").replace("m", "-")
    return int(value) if try_int(value) else float(value)


def group_kappas(**kappas: dict[str, str]) -> dict[int | float, list[str]]:
    """
    Groups kappa values by their coupling strength. Examples:

    .. code-block:: python

        group_kappas(kl="1", kt="1")           # {1: ["kl", "kt"]}
        group_kappas(kl="2p45", kt="1")        # {2.45: ["kl"], 1: ["kt"]}
        group_kappas(k2v="0", kv="1", kl="1")  # {0: ["k2v"], 1: ["kv", "kl"]}
    """
    str_groups = defaultdict(list)
    for k, v in kappas.items():
        str_groups[v].append(k)

    # convert keys to numbers
    return {kappa_str_to_num(k): v for k, v in str_groups.items()}


def create_kappa_label(*, sep: str = ",", group: bool = True, **kappas: dict[str, str]) -> str:
    # either group or just list kappas
    if group:
        gen = group_kappas(**kappas).items()
    else:
        gen = ((kappa_str_to_num(v), [k]) for k, v in kappas.items())
    # loop over kappas and join them
    parts = []
    for v, _kappas in gen:
        k_str = "=".join(rf"\kappa_{{{k}}}"for k in _kappas)
        parts.append(f"{k_str}={v}")
    return "$" + sep.join(parts) + "$"
