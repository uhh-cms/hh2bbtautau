# coding: utf-8

"""
Style definitions.
"""

import order as od


def stylize_processes(config: od.Config) -> None:
    """
    Adds process colors and adjust labels.
    """
    if config.has_process("hh_ggf_hbb_htt_kl1_kt1"):
        config.processes.n.hh_ggf_hbb_htt_kl1_kt1.color1 = "#3f90da"  # old: (67, 118, 201)
        config.processes.n.hh_ggf_hbb_htt_kl1_kt1.label = r"$HH_{ggf} \rightarrow bb\tau\tau$" + " \n " + r"($\kappa_{\lambda}=1$, $\kappa_{t}=1$)"  # noqa

    if config.has_process("hh_vbf_hbb_htt_kv1_k2v1_kl1"):
        config.processes.n.hh_vbf_hbb_htt_kv1_k2v1_kl1.color1 = "#011c87"  # old: (86, 211, 71)

    if config.has_process("h"):
        config.processes.n.h.color1 = "#92dadd"  # old: (65, 180, 219)

    if config.has_process("tt"):
        config.processes.n.tt.color1 = "#ffa90e"  # old: (244, 182, 66)

    if config.has_process("st"):
        config.processes.n.st.color1 = "#e76300"  # old: (244, 93, 66)

    if config.has_process("dy"):
        config.processes.n.dy.color1 = "#bd1f01"  # old: (68, 186, 104)

    if config.has_process("vv"):
        config.processes.n.vv.color1 = "#832db6"  # old: (2, 24, 140)

    if config.has_process("w"):
        config.processes.n.w.color1 = "#717581"

    if config.has_process("ewk"):
        config.processes.n.ewk.color1 = "#b9ac70"

    # if config.has_process("ttv"):
    #     config.processes.n.ttv.color1 = "#f7c331"

    if config.has_process("ttvv"):
        config.processes.n.ttvv.color1 = "#a96b59"

    if config.has_process("vvv"):
        config.processes.n.vvv.color1 = "#832db6"

    if config.has_process("multi_boson"):
        config.processes.n.multi_boson.color1 = "#832db6"

    if config.has_process("qcd"):
        config.processes.n.qcd.color1 = "#94a4a2"  # old: (242, 149, 99)
