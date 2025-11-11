# coding: utf-8

"""
Definition of categories.
"""

from __future__ import annotations

import functools

import law
import order as od

from columnflow.config_util import add_category, create_category_combinations, CategoryGroup, track_category_changes
from columnflow.types import Any


logger = law.logger.get_logger(__name__)


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    # root category (-1 has special meaning in cutflow)
    root_cat = add_category(config, name="all", id=-1, selection="cat_all", label="")
    _add_category = functools.partial(add_category, parent=root_cat)

    # reset the maximum category id counter
    od.Category._max_id = 0

    # lepton channels
    _add_category(config, name="etau", id=1, selection="cat_etau", label=config.channels.n.etau.label)
    _add_category(config, name="mutau", id=2, selection="cat_mutau", label=config.channels.n.mutau.label)
    _add_category(config, name="tautau", id=3, selection="cat_tautau", label=config.channels.n.tautau.label)
    _add_category(config, name="ee", id=4, selection="cat_ee", label=config.channels.n.ee.label)
    _add_category(config, name="mumu", id=5, selection="cat_mumu", label=config.channels.n.mumu.label)
    _add_category(config, name="emu", id=6, selection="cat_emu", label=config.channels.n.emu.label)

    # qcd regions
    _add_category(config, name="os", id="+", selection="cat_os", label="OS", tags={"os"})
    _add_category(config, name="ss", id="+", selection="cat_ss", label="SS", tags={"ss"})
    _add_category(config, name="iso", id="+", selection="cat_iso", label=r"iso", tags={"iso"})
    _add_category(config, name="noniso", id="+", selection="cat_noniso", label=r"non-iso", tags={"noniso"})  # noqa: E501

    # kinematic categories
    _add_category(config, name="incl", id="+", selection="cat_incl", label="inclusive")
    _add_category(config, name="eq0j", id="+", selection="cat_eq0j", label="0 jets")
    _add_category(config, name="eq1j", id="+", selection="cat_eq1j", label="1 jet")
    _add_category(config, name="eq2j", id="+", selection="cat_eq2j", label="2 jets")
    _add_category(config, name="ge2j", id="+", selection="cat_ge2j", label=r"$\geq$2 jets")
    _add_category(config, name="eq3j", id="+", selection="cat_eq3j", label="3 jets")
    _add_category(config, name="eq4j", id="+", selection="cat_eq4j", label="4 jets")
    _add_category(config, name="ge4j", id="+", selection="cat_ge4j", label=r"$\geq$4 jets")
    _add_category(config, name="eq5j", id="+", selection="cat_eq5j", label="5 jets")
    _add_category(config, name="ge5j", id="+", selection="cat_ge5j", label=r"$\geq$5 jets")
    _add_category(config, name="ge6j", id="+", selection="cat_ge6j", label=r"$\geq$6 jets")
    _add_category(config, name="eq0b", id="+", selection="cat_eq0b", label="0 b-tags")
    _add_category(config, name="ge0b", id="+", selection="cat_ge0b", label="")
    _add_category(config, name="eq1b", id="+", selection="cat_eq1b", label="1 b-tag")
    _add_category(config, name="eq2b", id="+", selection="cat_eq2b", label="2 b-tags")
    _add_category(config, name="ge2b", id="+", selection="cat_ge2b", label=r"$\geq$2 b-tags")
    _add_category(config, name="dy", id="+", selection="cat_dy", label="DY enriched")
    _add_category(config, name="dyc", id="+", selection="cat_dyc", label="DY region")
    _add_category(config, name="dy_st", id="+", selection=["cat_dy", "cat_single_triggered"], label="DY enriched, ST")
    _add_category(config, name="tt", id="+", selection="cat_tt", label=r"$t\bar{t}$ enriched")
    _add_category(config, name="mll40", id="+", selection="cat_mll40", label=r"$m_{ll} > 40$")

    _add_category(config, name="res1b", id="+", selection="cat_res1b", label="res1b")
    _add_category(config, name="res2b", id="+", selection="cat_res2b", label="res2b")
    _add_category(config, name="boosted", id="+", selection="cat_boosted", label="boosted")
    _add_category(config, name="res1b_novbf", id="+", selection="cat_res1b_novbf", label="res1b (vbf < 0.5)", tags={"skip_cutflow"})  # noqa: E501
    _add_category(config, name="res2b_novbf", id="+", selection="cat_res2b_novbf", label="res2b (vbf < 0.5)", tags={"skip_cutflow"})  # noqa: E501
    _add_category(config, name="boosted_novbf", id="+", selection="cat_boosted_novbf", label="boosted (vbf < 0.5)", tags={"skip_cutflow"})  # noqa: E501

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
                cat.label
                for cat in categories.values()
                if cat.label and cat.name != "os" and cat.name != "iso"  # os and iso are the defaults
            ]) or None,
        }

    # main analysis categories
    main_categories = {
        # channels first
        "channel": CategoryGroup(["etau", "mutau", "tautau"], is_complete=False, has_overlap=False),
        # kinematic regions in the middle (to be extended)
        "kin": CategoryGroup(["incl", "res1b", "res2b", "boosted", "res1b_novbf", "res2b_novbf", "boosted_novbf"], is_complete=True, has_overlap=True),  # noqa: E501
        # qcd regions last
        "sign": CategoryGroup(["os", "ss"], is_complete=True, has_overlap=False),
        "tau2": CategoryGroup(["iso", "noniso"], is_complete=True, has_overlap=False),
    }

    create_category_combinations(
        config=config,
        categories=main_categories,
        name_fn=name_fn,
        parent_mode="safe",
        kwargs_fn=functools.partial(kwargs_fn, add_qcd_group=True),
        skip_existing=False,
    )

    # control categories
    control_categories = {
        # channels
        "channel": CategoryGroup(["ee", "mumu", "emu"], is_complete=False, has_overlap=False),
        # kinematic regions
        "kin": CategoryGroup(["incl", "dy", "dyc", "tt", "dy_st", "mll40"], is_complete=True, has_overlap=True),
        "jets": CategoryGroup(["eq2j", "eq3j", "eq4j", "ge2j", "ge4j", "eq5j", "ge6j"], is_complete=True, has_overlap=True),  # noqa: E501
        "tags": CategoryGroup(["eq0b", "eq1b", "eq2b", "ge0b", "ge2b"], is_complete=True, has_overlap=True),
        # relative sign
        "sign": CategoryGroup(["os"], is_complete=False, has_overlap=False),
    }

    def skip_fn_ctrl(categories: dict[str, od.Category]) -> bool:
        if "channel" not in categories or "kin" not in categories:
            return False
        ch_cat = categories["channel"]
        kin_cat = categories["kin"]
        # skip dy in emu
        if kin_cat.name.startswith("dy") and ch_cat.name == "emu":
            return True
        # skip tt in ee/mumu
        if kin_cat.name == "tt" and ch_cat.name in {"ee", "mumu"}:
            return True
        return False

    create_category_combinations(
        config=config,
        categories=control_categories,
        name_fn=name_fn,
        parent_mode="none",
        kwargs_fn=functools.partial(kwargs_fn, add_qcd_group=False),
        skip_existing=False,
        skip_fn=skip_fn_ctrl,
    )

    track_category_changes(config)
