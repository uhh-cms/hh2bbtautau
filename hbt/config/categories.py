# coding: utf-8

"""
Definition of categories.
"""

import functools

import order as od

from columnflow.config_util import add_category, create_category_combinations
from columnflow.types import Any


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    # root category (-1 has special meaning in cutflow)
    root_cat = add_category(config, name="all", id=-1, selection="cat_all", label="")
    _add_category = functools.partial(add_category, parent=root_cat)

    # lepton channels
    _add_category(config, name="etau", id=1, selection="cat_etau", label=config.channels.n.etau.label)
    _add_category(config, name="mutau", id=2, selection="cat_mutau", label=config.channels.n.mutau.label)
    _add_category(config, name="tautau", id=3, selection="cat_tautau", label=config.channels.n.tautau.label)
    _add_category(config, name="ee", id=4, selection="cat_ee", label=config.channels.n.ee.label)
    _add_category(config, name="mumu", id=5, selection="cat_mumu", label=config.channels.n.mumu.label)
    _add_category(config, name="emu", id=6, selection="cat_emu", label=config.channels.n.emu.label)

    # qcd regions
    _add_category(config, name="os", id=10, selection="cat_os", label="OS", tags={"os"})
    _add_category(config, name="ss", id=11, selection="cat_ss", label="SS", tags={"ss"})
    _add_category(config, name="iso", id=12, selection="cat_iso", label=r"iso", tags={"iso"})
    _add_category(config, name="noniso", id=13, selection="cat_noniso", label=r"non-iso", tags={"noniso"})  # noqa: E501

    # costum categories for ABCD tests
    _add_category(config, name="all_incl", id=101, selection="cat_incl", label="inclusive")

    # positive and negative same sign regions
    _add_category(config, name="ss_pos", id=102, selection="cat_ss_pos", label="ss_pos", tags={"ss_pos"})
    _add_category(config, name="ss_neg", id=103, selection="cat_ss_neg", label="ss_neg", tags={"ss_neg"})

    # alternative tau2 isolation categories
    _add_category(config, name="iso_l", id=104, selection="cat_iso_l", label="passes_loose", tags={"passes_loose"})
    _add_category(config, name="iso_vl", id=105, selection="cat_iso_vl", label="passes_vloose", tags={"passes_vloose"})
    _add_category(config, name="iso_vvl", id=106, selection="cat_iso_vvl", label="passes_vvloose", tags={"passes_vvloose"})  # noqa: E501
    _add_category(config, name="l_m", id=107, selection="cat_l_m", label="loose_medium", tags={"loose_medium"})
    _add_category(config, name="vl_l", id=108, selection="cat_vl_l", label="vloose_loose", tags={"vloose_loose"})  # noqa: E501
    _add_category(config, name="vvl_vl", id=109, selection="cat_vvl_vl", label="vvloose_vloose", tags={"vvloose_vloose"})  # noqa: E501
    _add_category(config, name="vvvl_vvl", id=111, selection="cat_vvvl_vvl", label="vvvloose_vvloose", tags={"vvvloose_vvloose"})  # noqa: E501

    # kinematic categories
    _add_category(config, name="incl", id=100, selection="cat_incl", label="inclusive")
    _add_category(config, name="2j", id=110, selection="cat_2j", label="2 jets")
    _add_category(config, name="dy", id=210, selection="cat_dy", label="DY enriched")
    _add_category(config, name="tt", id=220, selection="cat_tt", label=r"$t\bar{t}$ enriched")

    _add_category(config, name="res1b", id=300, selection="cat_res1b", label="res1b")
    _add_category(config, name="res2b", id=301, selection="cat_res2b", label="res2b")
    _add_category(config, name="boosted", id=310, selection="cat_boosted", label="boosted")

    #
    # build groups
    #

    def name_fn(categories: dict[str, od.Category]) -> str:
        return "__".join(cat.name for cat in categories.values() if cat)

    def kwargs_fn(categories: dict[str, od.Category], add_qcd_group: bool = True) -> dict[str, Any]:
        # build auxiliary information
        aux = {}
        if add_qcd_group:
            aux["qcd_group"] = name_fn({
                name: cat for name, cat in categories.items()
                if name not in {"sign", "tau2"}
            })
        # return the desired kwargs
        return {
            # just increment the category id
            # NOTE: for this to be deterministic, the order of the categories must no change!
            "id": "+",
            # join all tags
            "tags": set.union(*[cat.tags for cat in categories.values() if cat]),
            # auxiliary information
            "aux": aux,
            # label
            "label": ", ".join([
                cat.label or cat.name
                for cat in categories.values()
                if cat.name != "os"  # os is the default
            ]) or None,
        }

    # main analysis categories
    main_categories = {
        # channels first
        "channel": [
            config.get_category("etau"), config.get_category("mutau"), config.get_category("tautau"),
        ],
        # kinematic regions in the middle (to be extended)
        "kin": [
            config.get_category("incl"),
            config.get_category("2j"),
            config.get_category("res1b"),
            config.get_category("res2b"),
            config.get_category("boosted"),
        ],
        # qcd regions last
        # "sign": [config.get_category("os"), config.get_category("ss_pos"), config.get_category("ss_neg")],
        "sign": [config.get_category("os"), config.get_category("ss")],
        "tau2": [config.get_category("iso"), config.get_category("noniso")],
    }

    create_category_combinations(
        config=config,
        categories=main_categories,
        name_fn=name_fn,
        kwargs_fn=functools.partial(kwargs_fn, add_qcd_group=True),
    )

    # control categories
    control_categories = {
        # channels first
        "channel": [
            config.get_category("ee"), config.get_category("mumu"), config.get_category("emu"),
        ],
        # kinematic regions in the middle (to be extended)
        "kin": [config.get_category("incl"), config.get_category("dy"), config.get_category("tt")],
        # relative sign last
        "sign": [config.get_category("os")],
    }

    def skip_fn_ctrl(categories: dict[str, od.Category]) -> bool:
        if "channel" not in categories or "kin" not in categories:
            return False
        ch_cat = categories["channel"]
        kin_cat = categories["kin"]
        # skip dy in emu
        if kin_cat.name == "dy" and ch_cat.name == "emu":
            return True
        # skip tt in ee/mumu
        if kin_cat.name == "tt" and ch_cat.name in ("ee", "mumu"):
            return True
        return False

    create_category_combinations(
        config=config,
        categories=control_categories,
        name_fn=name_fn,
        kwargs_fn=functools.partial(kwargs_fn, add_qcd_group=False),
        skip_fn=skip_fn_ctrl,
    )
