# coding: utf-8

"""
Definition of categories.
"""

import order as od

from columnflow.config_util import add_category

from collections import OrderedDict


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    add_category(
        config,
        name="incl",
        id=1,
        selection="sel_incl",
        label="inclusive",
    )
    add_category(
        config,
        name="2j",
        id=100,
        selection="sel_2j",
        label="2 jets",
    )


def add_categories_ml(config, ml_model_inst):

    # add ml categories directly to the config
    ml_categories = []
    for i, proc in enumerate(ml_model_inst.processes):
        ml_categories.append(config.add_category(
            # NOTE: name and ID is unique as long as we don't use
            #       multiple ml_models simutaneously
            name=f"ml_{proc}",
            id=(i + 1) * 10000,
            selection=f"catid_ml_{proc}",
            label=f"ml_{proc}",
        ))

    category_blocks = OrderedDict({
        "lep": [config.get_category("1e"), config.get_category("1mu")],
        "jet": [config.get_category("resolved"), config.get_category("boosted")],
        "b": [config.get_category("1b"), config.get_category("2b")],
        "dnn": ml_categories,
    })

    # create combination of categories
    n_cats = create_category_combinations(
        config,
        category_blocks,
        name_fn=name_fn,
        kwargs_fn=kwargs_fn,
        skip_existing=True,
    )
    logger.info(f"Number of produced ml category insts: {n_cats}")
