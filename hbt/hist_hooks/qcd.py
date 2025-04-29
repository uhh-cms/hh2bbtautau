# coding: utf-8

"""
Histogram hooks for QCD estimation.
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
    def qcd(
        task: law.Task,
        config_inst: od.Config,
        hists: dict[od.Process, Any],
    ) -> dict[od.Process, Any]:
        """
        This hook calculates the qcd estimation via ABCD method.
        """

        # SETUP
        # --------------------------------------------------------------------------------------
        os_tag, ss_tag, iso_tag, noniso_tag = "os", "ss", "iso", "noniso"
        analysis_channels = ["etau", "mutau", "tautau"]
        control_channels = ["ee", "emu", "mumu"]
        control_regions = ["ss_iso", "os_noniso", "ss_noniso"]
        # choose MC minimum uncertainty threshold. Uncertainties bellow min_mc_unc are disregraded
        min_mc_unc = 0.15
        # --------------------------------------------------------------------------------------

        # get the qcd process
        qcd_proc = config_inst.get_process("qcd", default=None)
        if (not qcd_proc or not hists):
            logger.warning("No hists or no qcd process found. Exiting hook...")
            return hists

        # extract all unique category names and verify that the axis order is exactly
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

        # create dictionary to store the BCD categories
        qcd_groups: dict[str, dict[str, od.Category]] = defaultdict(DotDict)
        for cat_name in category_names:
            cat_inst = config_inst.get_category(cat_name)
            if cat_inst.has_tag({"os", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_iso = cat_inst
            elif cat_inst.has_tag({"os", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_noniso = cat_inst
            elif cat_inst.has_tag({ss_tag, iso_tag}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_iso = cat_inst
            elif cat_inst.has_tag({ss_tag, noniso_tag}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_noniso = cat_inst

        # get category names without qcd regions, e.g. "etau__incl"
        group_names = [name for name, cats in qcd_groups.items()]

        # nothing to do if there are no complete groups
        if not group_names:
            return hists

        # sum up mc and data histograms, stop early when empty
        mc_hists = [h for p, h in hists.items() if p.is_mc and not p.has_tag("signal")]
        data_hists = [h for p, h in hists.items() if p.is_data]
        if not mc_hists or not data_hists:
            return hists
        mc_hist = sum(mc_hists[1:], mc_hists[0].copy())
        data_hist = sum(data_hists[1:], data_hists[0].copy())

        # start by copying the mc hist and reset it, then fill it at specific category slices
        hists[qcd_proc] = qcd_hist = mc_hist.copy().reset()

        # initialize variables
        dict_hists = {}
        channel = ""
        kin_region = ""
        all_channels = False

        # get requested kinematic region
        cat_name = task.categories[0]
        cat_info = cat_name.split("__")
        if any(ch_name in cat_info for ch_name in control_channels):
            logger.warning(f"Skipping QCD estimation for control region {task.categories[0]}.")
            return hists
        elif any(ch_name in cat_info for ch_name in analysis_channels):
            channel = cat_info[0]
            kin_region = cat_info[1]
        else:
            all_channels = True
            kin_region = cat_info[0]

        # decide which decay channels to use
        selected_groups = []
        for group in group_names:
            if (all_channels is False and group in cat_name):
                # get the requested single decay channel and kinematic region
                selected_groups.append(group)
            elif (all_channels is True and kin_region in group):
                # get the etau, mutau and tautau groups in the requested kinematic region
                selected_groups.append(group)

        # save the category ids of decay channels, per qcd region
        dic_ids = {}
        for region_name in control_regions:
            dic_ids[region_name] = []
            for group_name in selected_groups:
                group = qcd_groups[group_name]
                dic_ids[region_name].append(hist.loc(group[region_name].id))

        # helper function to sum the hists from the selected decay channels, per qcd region
        get_hist = lambda h, region_name: h[{"category": dic_ids[region_name]}][{"category": sum}]

            # get the corresponding histograms and convert them to number objects, each one storing an array of values
            # with uncertainties
            # shapes: (SHIFT, VAR)
            def get_hist(h: hist.Histogram, region_name: str) -> hist.Histogram:
                h = ensure_category(h, group[region_name].name)
                return h[{"category": hist.loc(group[region_name].name)}]
            os_noniso_mc = hist_to_num(get_hist(mc_hist, "os_noniso"), "os_noniso_mc")
            ss_noniso_mc = hist_to_num(get_hist(mc_hist, "ss_noniso"), "ss_noniso_mc")
            ss_iso_mc = hist_to_num(get_hist(mc_hist, "ss_iso"), "ss_iso_mc")
            os_noniso_data = hist_to_num(get_hist(data_hist, "os_noniso"), "os_noniso_data")
            ss_noniso_data = hist_to_num(get_hist(data_hist, "ss_noniso"), "ss_noniso_data")
            ss_iso_data = hist_to_num(get_hist(data_hist, "ss_iso"), "ss_iso_data")

        # data will always have a single shift whereas mc might have multiple,
        # broadcast numbers in-place manually if necessary
        if (n_shifts := mc_hist.axes["shift"].size) > 1:
            def broadcast_data_num(num: sn.Number) -> None:
                num._nominal = np.repeat(num.nominal, n_shifts, axis=0)
                for name, (unc_up, unc_down) in num._uncertainties.items():
                    num._uncertainties[name] = (
                        np.repeat(unc_up, n_shifts, axis=0),
                        np.repeat(unc_down, n_shifts, axis=0),
                    )

            for region in control_regions:
                broadcast_data_num(dict_hists[region + "_data"])

        num_region = dict_hists[numerator + "_qcd"]
        den_region = dict_hists[denominator + "_qcd"]
        shape_estimation = dict_hists[shape_region + "_qcd"]

        # sum over bins
        # shapes: (SHIFT,)
        int_num = integrate_num(num_region, axis=1)
        int_den = integrate_num(den_region, axis=1)

        # complain about negative integrals
        int_num_neg = int_num <= 0
        int_den_neg = int_den <= 0
        if int_num_neg.any():
            shift_ids = list(map(mc_hist.axes["shift"].value, np.where(int_num_neg)[0]))
            shifts = list(map(config_inst.get_shift, shift_ids))
            logger.warning(
                f"negative QCD integral found in {numerator} region and shifts: "
                f"{', '.join(map(str, shifts))}",
            )
        if int_den_neg.any():
            shift_ids = list(map(mc_hist.axes["shift"].value, np.where(int_den_neg)[0]))
            shifts = list(map(config_inst.get_shift, shift_ids))
            logger.warning(
                f"negative QCD integral found in {denominator} region and shifts: "
                f"{', '.join(map(str, shifts))}",
            )

        # shape: (SHIFT, VAR)
        factor_int = (int_num / int_den)[0, None]

        # calculate qcd estimation
        qcd_estimation = shape_estimation * factor_int
        qcd_estimation_values = qcd_estimation()
        # combine uncertainties and store values in bare arrays
        qcd_estimation_variances = qcd_estimation(sn.UP, sn.ALL, unc=True)**2

        # define uncertainties from control regions
        data_unc_list = [f"{region}_data" for region in control_regions]
        mc_unc_list = [f"{region}_mc" for region in control_regions]
        unc_data = qcd_estimation(sn.UP, data_unc_list, unc=True)
        unc_mc = qcd_estimation(sn.UP, mc_unc_list, unc=True)
        # calculate relative uncertainties
        unc_data_rel = abs(unc_data / qcd_estimation_values)
        unc_mc_rel = abs(unc_mc / qcd_estimation_values)

        # only keep the MC uncertainty if it is larger than the data uncertainty and larger than min_mc_unc
        keep_variance_mask = (
            np.isfinite(unc_mc_rel) &
            (unc_mc_rel > unc_data_rel) &
            (unc_mc_rel > min_mc_unc)
        )
        qcd_estimation_variances[keep_variance_mask] = unc_mc[keep_variance_mask]**2
        qcd_estimation_variances[~keep_variance_mask] = 0

        # retro-actively set values to zero for shifts that had negative integrals
        neg_int_mask = int_num_neg | int_den_neg
        qcd_estimation_values[neg_int_mask] = 1e-5
        qcd_estimation_variances[neg_int_mask] = 0

        # residual zero filling
        zero_mask = qcd_estimation_values <= 0
        qcd_estimation_values[zero_mask] = 1e-5
        qcd_estimation_variances[zero_mask] = 0

            # insert values into the qcd histogram
            cat_axis = qcd_hist.axes["category"]
            for cat_index in range(cat_axis.size):
                if cat_axis.value(cat_index) == group.os_iso.name:
                    qcd_hist.view().value[cat_index, ...] = os_iso_qcd_values
                    qcd_hist.view().variance[cat_index, ...] = os_iso_qcd_variances
                    break
            else:
                raise RuntimeError(
                    f"could not find index of bin on 'category' axis of qcd histogram {qcd_hist} for category "
                    f"{group.os_iso}",
                )

        return hists

    # add all hooks
    analysis_inst.x.hist_hooks.qcd = qcd
