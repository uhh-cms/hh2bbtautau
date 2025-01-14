# coding: utf-8

"""
Style definitions.
"""

import order as od

from columnflow.util import DotDict


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
    )

    for kl in ["0", "1", "2p45", "5"]:
        if (p := config.get_process(f"hh_ggf_hbb_htt_kl{kl}_kt1", default=None)):
            p.color1 = cfg.x.colors.dark_blue
            p.label = (
                r"$HH_{ggf} \rightarrow bb\tau\tau$ __SCALE__"
                "\n"
                rf"($\kappa_{{\lambda}}$={kl.replace('p', '.')},$\kappa_{{t}}$=1)"
            )

    if (p := config.get_process("hh_vbf_hbb_htt_kv1_k2v1_kl1", default=None)):
        p.color1 = cfg.x.colors.brown
        p.label = (
            r"$HH_{vbf} \rightarrow bb\tau\tau$ __SCALE__"
            "\n"
            r"($\kappa_{\lambda}$=1,$\kappa_{V}$=1,$\kappa_{2V}$=1)"
        )

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
        p.color1 = cfg.x.colors.aubergine

    if (p := config.get_process("vvv", default=None)):
        p.color1 = cfg.x.colors.aubergine

    if (p := config.get_process("multiboson", default=None)):
        p.color1 = cfg.x.colors.aubergine

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


def update_legend_labels(ax, handles, labels, n_cols):
    """
    Fill empty handles so that all backgrounds are in the first column.
    """
    # only implemented for 2 columns so far
    if n_cols != 2:
        return
    # get number of entries per column
    n_backgrounds = sum(1 for handle in handles if handle.__class__.__name__ == "StepPatch")
    n_other = len(handles) - n_backgrounds
    if n_backgrounds == n_other:
        return

    # fill left or right column
    n_add = abs(n_backgrounds - n_other)
    pos = len(handles) if n_backgrounds > n_other else n_backgrounds
    empty_handle = ax.plot([], label="", linestyle="None")[0]
    for _ in range(n_add):
        handles.insert(pos, empty_handle)
        labels.insert(pos, "")
