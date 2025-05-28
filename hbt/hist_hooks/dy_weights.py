# coding: utf-8

"""
Histogram hooks for DY weights.
"""

from __future__ import annotations

from collections import defaultdict

import law
import order as od
import scinum as sn

from columnflow.util import maybe_import, DotDict
from columnflow.types import Any

np = maybe_import("numpy")
hist = maybe_import("hist")


logger = law.logger.get_logger(__name__)


# helper to convert a histogram to a number object containing bin values and uncertainties
# from variances stored in an array of values
def hist_to_num(h: hist.Histogram, unc_name=str(sn.DEFAULT)) -> sn.Number:
    return sn.Number(h.values(), {unc_name: h.variances()**0.5})


# helper to integrate values stored in an array based number object
def integrate_num(num: sn.Number, axis=None) -> sn.Number:
    return sn.Number(
        nominal=num.nominal.sum(axis=axis),
        uncertainties={
            unc_name: (
                (unc_values_up**2).sum(axis=axis)**0.5,
                (unc_values_down**2).sum(axis=axis)**0.5,
            )
            for unc_name, (unc_values_up, unc_values_down) in num.uncertainties.items()
        },
    )


# helper to ensure that a specific category exists on the "category" axis of a histogram
def ensure_category(h: hist.Histogram, category_name: str) -> hist.Histogram:
    cat_axis = h.axes["category"]
    if category_name in cat_axis:
        return h
    dummy_fill = {ax.name: ax[0] for ax in h.axes if ax.name != "category"}
    h.fill(**dummy_fill, category=category_name, weight=0.0)
    return h


def add_hooks(analysis_inst: od.Analysis) -> None:
    """
    Add histogram hooks to a analysis.
    """
    def dy_ratio_per_config(
        task: law.Task,
        config_inst: od.Config,
        hists: dict[od.Process, Any],
    ) -> dict[od.Process, Any]:

        # get dummy processes
        factor_bin = config_inst.get_process("qcd", default=None)
        if not factor_bin:
            return hists

        # extract all unique category ids and verify that the axis order is exactly
        # "category -> shift -> variable" which is needed to insert values at the end
        CAT_AXIS, SHIFT_AXIS, VAR_AXIS = range(3)
        category_names = set()
        for proc, h in hists.items():
            # validate axes
            assert len(h.axes) == 3
            assert h.axes[CAT_AXIS].name == "category"
            assert h.axes[SHIFT_AXIS].name == "shift"
            # get the category axis
            cat_ax = h.axes["category"]
            category_names.update(list(cat_ax))

        # define DY enriched regions
        kin_regions = [
            "dy", "dy_res1b", "dy_res2b", "dy_boosted",
            "dy_eq0j", "dy_eq1j", "dy_eq2j", "dy_eq3j", "dy_ge4j",
            "dyc", "dyc_res1b", "dyc_res2b", "dyc_boosted",
            "dyc_eq0j", "dyc_eq1j", "dyc_eq2j", "dyc_eq3j", "dyc_ge4j",
        ]

        dy_groups: dict[str, dict[str, od.Category]] = defaultdict(DotDict)

        for cat_name in category_names:
            cat_inst = config_inst.get_category(cat_name)
            # get dy ratio in specific control channel in inclusive region
            for kin_region in kin_regions:
                if f"mumu__{kin_region}__os" in cat_inst.name:
                    if cat_inst.has_tag({kin_region}, mode=all):
                        dy_groups[cat_inst.x.dy_group][kin_region] = cat_inst

        # get complete qcd groups
        complete_groups = [name for name, cats in dy_groups.items()]

        # nothing to do if there are no complete groups
        if not complete_groups:
            return hists

        # sum up mc and data histograms, stop early when empty
        dy_hists = [h for p, h in hists.items() if p.is_mc and p.has_tag("dy")]
        mc_hists = [h for p, h in hists.items() if p.is_mc and not p.has_tag("signal") and not p.has_tag("dy")]
        data_hists = [h for p, h in hists.items() if p.is_data]
        if not dy_hists or not mc_hists or not data_hists:
            return hists

        dy_hist = sum(dy_hists[1:], dy_hists[0].copy())
        mc_hist = sum(mc_hists[1:], mc_hists[0].copy())
        data_hist = sum(data_hists[1:], data_hists[0].copy())

        # start by copying the mc hist and reset it, then fill it at specific category slices
        hists = {}
        hists[factor_bin] = factor_hist = mc_hist.copy().reset()

        # initializing dictionary for later use
        dict_hists = {}

        for group_name in complete_groups:
            group = dy_groups[group_name]

            # get the corresponding histograms and convert them to number objects, each one storing an array of values
            # with uncertainties
            # shapes: (SHIFT, VAR)
            def get_hist(h: hist.Histogram, region_name: str) -> hist.Histogram:
                h = ensure_category(h, cat_name)
                return h[{"category": hist.loc(cat_name)}]

            for region in kin_regions:
                cat_name = f"mumu__{region}__os"
                hist_dy = hist_to_num(get_hist(dy_hist, cat_name), region + "_dy")
                hist_mc = hist_to_num(get_hist(mc_hist, cat_name), region + "_mc")
                hist_data = hist_to_num(get_hist(data_hist, cat_name), region + "_data")
                hist_diff = hist_data - hist_mc

                # save hists
                dict_hists[region + "_dy"] = (hist_dy)
                dict_hists[region + "_mc"] = (hist_mc)
                dict_hists[region + "_data"] = (hist_data)
                dict_hists[region + "_diff"] = (hist_diff)

                print(dict_hists[region + "_mc"])
                print(dict_hists[region + "_data"])
                print("Hist data - non-DY MC:")
                print(dict_hists[region + "_diff"])
                print("DY Hist:")
                print(dict_hists[region + "_dy"])

                # calculate the ratio factor per bin
                num_region = dict_hists[region + "_diff"]
                den_region = dict_hists[region + "_dy"]

                # calculate the ratio factor per bin
                factor = (num_region / den_region)[:, None]
                factor_values = np.squeeze(np.nan_to_num(factor()), axis=0)
                factor_variances = factor(sn.UP, sn.ALL, unc=True)**2

                print("-----------")
                print(factor)
                print("-----------")

                # insert per bin ratio of (data-MC)/DY into plotting histogram
                cat_axis = factor_hist.axes["category"]
                for cat_index in range(cat_axis.size):
                    if hasattr(group, region):
                        if cat_axis.value(cat_index) == group.get(region).name:
                            factor_hist.view().value[cat_index, ...] = factor_values
                            factor_hist.view().variance[cat_index, ...] = factor_variances
                            break
                    else:
                        continue

        return hists

    def dy_ratio(
        task: law.Task,
        hists: dict[od.Config, dict[od.Process, Any]],
    ) -> dict[od.Config, dict[od.Process, Any]]:
        return {
            config_inst: dy_ratio_per_config(task, config_inst, hists[config_inst])
            for config_inst in hists.keys()
        }

    # add the hook
    analysis_inst.x.hist_hooks.dy = dy_ratio
