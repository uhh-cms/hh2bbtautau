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
        config.processes.n.hh_ggf_hbb_htt_kl1_kt1.color1 = (67, 118, 201)

    if config.has_process("hh_vbf_hbb_htt_kv1_k2v1_kl1"):
        config.processes.n.hh_vbf_hbb_htt_kv1_k2v1_kl1.color1 = (86, 211, 71)

    if config.has_process("h"):
        config.processes.n.h.color1 = (65, 180, 219)

    if config.has_process("tt"):
        config.processes.n.tt.color1 = (244, 182, 66)

    if config.has_process("st"):
        config.processes.n.st.color1 = (244, 93, 66)

    if config.has_process("dy"):
        config.processes.n.dy.color1 = (68, 186, 104)

    if config.has_process("vv"):
        config.processes.n.vv.color1 = (2, 24, 140)

    if config.has_process("qcd"):
        config.processes.n.qcd.color1 = (242, 149, 99)
