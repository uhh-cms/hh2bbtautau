# coding: utf-8

"""
Definition of categories.
"""

import functools

import order as od

from columnflow.config_util import add_category, create_category_combinations, CategoryGroup
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

    # kinematic categories
    _add_category(config, name="incl", id=100, selection="cat_incl", label="inclusive")
    _add_category(config, name="2j", id=110, selection="cat_2j", label="2 jets")
    _add_category(config, name="res1b", id=300, selection="cat_res1b", label="res1b")
    _add_category(config, name="res2b", id=301, selection="cat_res2b", label="res2b")
    _add_category(config, name="boosted", id=310, selection="cat_boosted", label="boosted")
    _add_category(config, name="tt", id=220, selection="cat_tt", label=r"$t\bar{t}$ enriched")

    # DY enriched with CF definition
    _add_category(config, name="dy", id=210, selection="cat_dy", label="DY enriched", tags={"dy"})
    _add_category(config, name="dy_res1b", id=211, selection=["cat_dy", "cat_res1b"], label="DY enriched res1b", tags={"dy_res1b"})  # noqa: E501
    _add_category(config, name="dy_res2b", id=212, selection=["cat_dy", "cat_res2b"], label="DY enriched res2b", tags={"dy_res2b"})  # noqa: E501
    _add_category(config, name="dy_boosted", id=216, selection=["cat_dy", "cat_boosted"], label="DY enriched boosted", tags={"dy_boosted"})  # noqa: E501

    _add_category(config, name="dy_eq0j", id=218, selection=["cat_dy", "cat_eq0j"], label="DY enriched 0j", tags={"dy_eq0j"})  # noqa: E501
    _add_category(config, name="dy_eq1j", id=219, selection=["cat_dy", "cat_eq1j"], label="DY enriched 1j", tags={"dy_eq1j"})  # noqa: E501
    _add_category(config, name="dy_eq2j", id=221, selection=["cat_dy", "cat_eq2j"], label="DY enriched 2j", tags={"dy_eq2j"})  # noqa: E501
    _add_category(config, name="dy_eq3j", id=222, selection=["cat_dy", "cat_eq3j"], label="DY enriched 3j", tags={"dy_eq3j"})  # noqa: E501
    _add_category(config, name="dy_eq4j", id=226, selection=["cat_dy", "cat_eq4j"], label="DY enriched 4j", tags={"dy_eq4j"})  # noqa: E501
    _add_category(config, name="dy_eq5j", id=227, selection=["cat_dy", "cat_eq5j"], label="DY enriched 5j", tags={"dy_eq5j"})  # noqa: E501
    _add_category(config, name="dy_eq6j", id=228, selection=["cat_dy", "cat_eq6j"], label="DY enriched 6j", tags={"dy_eq6j"})  # noqa: E501
    _add_category(config, name="dy_eq7j", id=229, selection=["cat_dy", "cat_eq7j"], label="DY enriched 7j", tags={"dy_eq7j"})  # noqa: E501

    _add_category(config, name="dy_ge4j", id=223, selection=["cat_dy", "cat_ge4j"], label="DY enriched >4j", tags={"dy_ge4j"})  # noqa: E501
    _add_category(config, name="dy_ge6j", id=230, selection=["cat_dy", "cat_ge6j"], label="DY enriched >6j", tags={"dy_ge6j"})  # noqa: E501
    _add_category(config, name="dy_ge7j", id=231, selection=["cat_dy", "cat_ge7j"], label="DY enriched >7j", tags={"dy_ge7j"})  # noqa: E501

    _add_category(config, name="dy_eq2j_eq0bj", id=241, selection=["cat_dy", "cat_eq2j", "cat_eq0bj"], label="DY enriched eq2j 0bj", tags={"dy_eq2j_eq0bj"})  # noqa: E501
    _add_category(config, name="dy_eq2j_eq1bj", id=242, selection=["cat_dy", "cat_eq2j", "cat_eq1bj"], label="DY enriched eq2j 1bj", tags={"dy_eq2j_eq1bj"})  # noqa: E501
    _add_category(config, name="dy_eq2j_eq2bj", id=243, selection=["cat_dy", "cat_eq2j", "cat_eq2bj"], label="DY enriched eq2j 2bj", tags={"dy_eq2j_eq2bj"})  # noqa: E501

    # DY enriched with CCLUB definition
    _add_category(config, name="dyc", id=213, selection="cat_dyc", label="DY enriched (CCLUB)", tags={"dyc"})
    _add_category(config, name="dyc_res1b", id=214, selection=["cat_dyc", "cat_res1b"], label="DY enriched res1b (CCLUB)", tags={"dyc_res1b"})  # noqa: E501
    _add_category(config, name="dyc_res2b", id=215, selection=["cat_dyc", "cat_res2b"], label="DY enriched res2b (CCLUB)", tags={"dyc_res2b"})  # noqa: E501
    _add_category(config, name="dyc_boosted", id=217, selection=["cat_dyc", "cat_boosted"], label="DY enriched boosted (CCLUB)", tags={"dyc_boosted"})  # noqa: E501

    _add_category(config, name="dyc_eq0j", id=224, selection=["cat_dyc", "cat_eq0j"], label="DY enriched 0j (CCLUB)", tags={"dyc_eq0j"})  # noqa: E501
    _add_category(config, name="dyc_eq1j", id=225, selection=["cat_dyc", "cat_eq1j"], label="DY enriched 1j (CCLUB)", tags={"dyc_eq1j"})  # noqa: E501
    _add_category(config, name="dyc_eq2j", id=232, selection=["cat_dyc", "cat_eq2j"], label="DY enriched 2j (CCLUB)", tags={"dyc_eq2j"})  # noqa: E501
    _add_category(config, name="dyc_eq3j", id=233, selection=["cat_dyc", "cat_eq3j"], label="DY enriched 3j (CCLUB)", tags={"dyc_eq3j"})  # noqa: E501
    _add_category(config, name="dyc_eq4j", id=234, selection=["cat_dyc", "cat_eq4j"], label="DY enriched 4j (CCLUB)", tags={"dyc_eq4j"})  # noqa: E501
    _add_category(config, name="dyc_eq5j", id=235, selection=["cat_dyc", "cat_eq5j"], label="DY enriched 5j (CCLUB)", tags={"dyc_eq5j"})  # noqa: E501
    _add_category(config, name="dyc_eq6j", id=236, selection=["cat_dyc", "cat_eq6j"], label="DY enriched 6j (CCLUB)", tags={"dyc_eq6j"})  # noqa: E501
    _add_category(config, name="dyc_eq7j", id=237, selection=["cat_dyc", "cat_eq7j"], label="DY enriched 7j (CCLUB)", tags={"dyc_eq7j"})  # noqa: E501

    _add_category(config, name="dyc_ge4j", id=238, selection=["cat_dyc", "cat_ge4j"], label="DY enriched >4j (CCLUB)", tags={"dyc_ge4j"})  # noqa: E501
    _add_category(config, name="dyc_ge6j", id=239, selection=["cat_dyc", "cat_ge6j"], label="DY enriched >6j (CCLUB)", tags={"dyc_ge6j"})  # noqa: E501
    _add_category(config, name="dyc_ge7j", id=240, selection=["cat_dyc", "cat_ge7j"], label="DY enriched >7j (CCLUB)", tags={"dyc_ge7j"})  # noqa: E501

    _add_category(config, name="dyc_eq2j_eq0bj", id=244, selection=["cat_dyc", "cat_eq2j", "cat_eq0bj"], label="DY enriched eq2j 0bj (CCLUB)", tags={"dyc_eq2j_eq0bj"})  # noqa: E501
    _add_category(config, name="dyc_eq2j_eq1bj", id=245, selection=["cat_dyc", "cat_eq2j", "cat_eq1bj"], label="DY enriched eq2j 1bj (CCLUB)", tags={"dyc_eq2j_eq1bj"})  # noqa: E501
    _add_category(config, name="dyc_eq2j_eq2bj", id=246, selection=["cat_dyc", "cat_eq2j", "cat_eq2bj"], label="DY enriched eq2j 2bj (CCLUB)", tags={"dyc_eq2j_eq2bj"})  # noqa: E501

    #
    # build groups
    #

    def name_fn(categories: dict[str, od.Category]) -> str:
        return "__".join(cat.name for cat in categories.values() if cat)

    def kwargs_fn(categories: dict[str, od.Category], add_qcd_group: bool = True, add_dy_group: bool = True) -> dict[str, Any]:
        # build auxiliary information
        aux = {}
        if add_qcd_group:
            aux["qcd_group"] = name_fn({
                name: cat for name, cat in categories.items()
                if name not in {"sign", "tau2"}
            })
        if add_dy_group:
            aux["dy_group"] = name_fn({
                name: cat for name, cat in categories.items()
                if name not in {"kin"}
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
                if cat.name != "os" and cat.name != "iso"  # os and iso are the defaults
            ]) or None,
        }

    # main analysis categories
    main_categories = {
        # channels first
        "channel": CategoryGroup(["etau", "mutau", "tautau"], is_complete=False, has_overlap=False),
        # kinematic regions in the middle (to be extended)
        "kin": CategoryGroup(["incl", "2j", "res1b", "res2b", "boosted"], is_complete=True, has_overlap=True),
        # qcd regions last
        "sign": CategoryGroup(["os", "ss"], is_complete=True, has_overlap=False),
        "tau2": CategoryGroup(["iso", "noniso"], is_complete=True, has_overlap=False),
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
        "channel": CategoryGroup(["ee", "mumu", "emu"], is_complete=False, has_overlap=False),
        # kinematic regions in the middle (to be extended)
        "kin": CategoryGroup([
            "incl",
            "dy", "dy_res1b", "dy_res2b", "dy_boosted",
            "dy_eq0j", "dy_eq1j", "dy_eq2j", "dy_eq3j", "dy_eq4j",
            "dy_eq5j", "dy_eq6j", "dy_eq7j", "dy_ge4j", "dy_ge6j", "dy_ge7j",
            "dy_eq2j_eq0bj", "dy_eq2j_eq1bj", "dy_eq2j_eq2bj",
            "dyc", "dyc_res1b", "dyc_res2b", "dyc_boosted",
            "dyc_eq0j", "dyc_eq1j", "dyc_eq2j", "dyc_eq3j", "dyc_eq4j",
            "dyc_eq5j", "dyc_eq6j", "dyc_eq7j", "dyc_ge4j", "dyc_ge6j", "dyc_ge7j",
            "dyc_eq2j_eq0bj", "dyc_eq2j_eq1bj", "dyc_eq2j_eq2bj",
            "tt",
        ], is_complete=True, has_overlap=True),
        # relative sign last
        "sign": CategoryGroup(["os"], is_complete=False, has_overlap=False),
    }

    def skip_fn_ctrl(categories: dict[str, od.Category]) -> bool:
        if "channel" not in categories or "kin" not in categories:
            return False
        ch_cat = categories["channel"]
        kin_cat = categories["kin"]
        # skip dy in emu
        if ("dy" in kin_cat.name) and ch_cat.name == "emu":
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
