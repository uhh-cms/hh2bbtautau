# coding: utf-8

"""
Definition of categories.
"""

import order as od

from columnflow.config_util import add_category, create_category_combinations


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    # lepton channels
    add_category(config, name="etau", id=1, selection="cat_etau", label=r"$e\tau_{h}$")
    add_category(config, name="mutau", id=2, selection="cat_mutau", label=r"$\mu\tau_{h}$")
    add_category(config, name="tautau", id=3, selection="cat_tautau", label=r"$\tau_{h}\tau_{h}$")

    # qcd regions
    add_category(config, name="os", id=10, selection="cat_os", label="Opposite sign", tags={"os"})
    add_category(config, name="ss", id=11, selection="cat_ss", label="Same sign", tags={"ss"})
    add_category(config, name="iso", id=12, selection="cat_iso", label=r"$\tau_{h,2} isolated$", tags={"iso"})
    add_category(config, name="noniso", id=13, selection="cat_noniso", label=r"$\tau_{h,2} non-isolated$", tags={"noniso"})  # noqa: E501

    # kinematic categories
    add_category(config, name="incl", id=100, selection="cat_incl", label="inclusive")
    add_category(config, name="2j", id=110, selection="cat_2j", label="2 jets")

    #
    # build groups
    #

    categories = {
        # channels first
        "channel": [config.get_category("etau"), config.get_category("mutau"), config.get_category("tautau")],
        # kinematic regions in the middle (to be extended)
        "kin": [config.get_category("incl"), config.get_category("2j")],
        # qcd regions last
        "sign": [config.get_category("os"), config.get_category("ss")],
        "tau2": [config.get_category("iso"), config.get_category("noniso")],
    }

    def name_fn(categories):
        return "__".join(cat.name for cat in categories.values() if cat)

    def kwargs_fn(categories):
        return {
            # just increment the category id
            # NOTE: for this to be deterministic, the order of the categories must no change!
            "id": "+",
            # join all tags
            "tags": set.union(*[cat.tags for cat in categories.values() if cat]),
            # auxiliary information
            "aux": {
                # the qcd group name
                "qcd_group": name_fn({name: cat for name, cat in categories.items() if name not in {"sign", "tau2"}}),
            },
        }

    create_category_combinations(config, categories, name_fn, kwargs_fn)
