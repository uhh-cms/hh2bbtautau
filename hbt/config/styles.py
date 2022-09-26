# coding: utf-8

"""
Style definitions.
"""

import order as od


def stylize_processes(config: od.Config) -> None:
    """
    Adds process colors and adjust labels.
    """
    if config.has_process("hh_ggf_bbtautau"):
        config.processes.n.hh_ggf_bbtautau.color = (67, 118, 201)

    if config.has_process("h"):
        config.processes.n.h.color = (65, 180, 219)

    if config.has_process("tt"):
        config.processes.n.tt.color = (244, 182, 66)

    if config.has_process("st"):
        config.processes.n.st.color = (244, 93, 66)

    if config.has_process("dy"):
        config.processes.n.dy.color = (68, 186, 104)

    if config.has_process("qcd"):
        config.processes.n.qcd.color = (242, 149, 99)
