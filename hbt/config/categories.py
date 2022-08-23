# coding: utf-8

"""
Definition of categories.
"""

import order as od


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    config.add_category(
        name="incl",
        id=1,
        selection="sel_incl",
        label="inclusive",
    )
    config.add_category(
        name="2j",
        id=100,
        selection="sel_2j",
        label="2 jets",
    )
