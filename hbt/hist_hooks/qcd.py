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


def add_hooks(config: od.Config) -> None:
    """
    Add histogram hooks related to QCD estimation to a configuration.
    """

    def abcd_stats(task, hists):
        """
        hist hook to plot the statistics in each of the ABCD qcd regions.
        When calling the abcd_stats hook make sure to also call --categories all_incl
        To plot the ABCD regions for a specific channel, modify l771 accordingly.

        Note:
        The plotting style of the x-axis must be set in l190 of columnflow/columnflow/tasks/plotting.py
        over/underflow of histograms must be commented out in l287-290 of columnflow/columnflow/plotting/plot_util.py
        """

        cats = [
            task.config_inst.get_category(c)
            for c in [f"incl__{a}__{b}" for a in ["os", "ss"] for b in ["iso", "noniso"]]
        ]

        results = {}
        for process, h in hists.items():
            h_new = hist.Hist(
                hist.axes.StrCategory([c.name for c in cats], name=h.axes[-1].name),
                hist.axes.IntCategory([0], name="shift"),
                hist.axes.IntCategory([101], name="category"),  # 101: all_incl
                storage=hist.storage.Weight(),
            )
            for ind, big_cat in enumerate(cats):
                h_sum = h[
                    {
                        # use [0]: etau; [1]: mutau; [2]: tautau; remove [] for all_incl
                        "category": [hist.loc(cat.id) for cat in big_cat.get_leaf_categories()],
                        "shift": sum,
                    }
                ].sum()
                h_new.values()[ind][0][0] = h_sum.value
                h_new.variances()[ind][0][0] = h_sum.variance
            results[process] = h_new
        return results

    def qcd_estimation(task, hists):
        """
        This hook calculates qcd estimation (shape x factor) for a choosen category.
        In the command line call --categories {etau,mutau,tautau}_{incl,2j,...}__os__iso
        """

        # SETUP
        # --------------------------------------------------------------------------------------
        control_regions = ["ss_iso", "os_noniso", "ss_noniso"]
        sig_region = "os_iso"
        # choose region to extract qcd shape estimation
        shape_region = control_regions[1]
        # choose numerator and denominator for ratio factor calculation
        numerator = control_regions[0]
        denominator = control_regions[2]
        # choose MC minimum uncertainty theshold. Uncertainties bellow min_mc_unc are disregraded
        min_mc_unc = 0.15
        # --------------------------------------------------------------------------------------

        if len(control_regions) != 3:
            raise ValueError("Please define exactly 3 control regions!")
        if sig_region in control_regions:
            raise ValueError("Signal region must not be in control regions set!")
        if shape_region == numerator or shape_region == denominator:
            raise ValueError("shape_region must not be the same as numerator or denominator!")
        if numerator == denominator:
            raise ValueError("Numerator and denominator for ratio factor cannot be the same region!")
        if not hists:
            return hists

        # get the qcd process
        qcd_proc = config.get_process("qcd", default=None)
        if not qcd_proc:
            return hists

        # extract all unique category ids and verify that the axis order is exactly
        # "category -> shift -> variable" which is needed to insert values at the end
        CAT_AXIS, SHIFT_AXIS, VAR_AXIS = range(3)
        category_ids = set()
        for proc, h in hists.items():
            # validate axes
            assert len(h.axes) == 3
            assert h.axes[CAT_AXIS].name == "category"
            assert h.axes[SHIFT_AXIS].name == "shift"
            # get the category axis
            cat_ax = h.axes["category"]
            for cat_index in range(cat_ax.size):
                category_ids.add(cat_ax.value(cat_index))

        # create qcd groups
        qcd_groups: dict[str, dict[str, od.Category]] = defaultdict(DotDict)
        for cat_id in category_ids:
            cat_inst = config.get_category(cat_id)
            if cat_inst.has_tag({"os", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_iso = cat_inst
            elif cat_inst.has_tag({"os", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_noniso = cat_inst
            elif cat_inst.has_tag({"ss", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_iso = cat_inst
            elif cat_inst.has_tag({"ss_pos", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_pos_iso = cat_inst
            elif cat_inst.has_tag({"ss_neg", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_neg_iso = cat_inst
            elif cat_inst.has_tag({"ss", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_noniso = cat_inst
            elif cat_inst.has_tag({"ss_pos", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_pos_noniso = cat_inst
            elif cat_inst.has_tag({"ss_neg", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_neg_noniso = cat_inst

        # get complete qcd groups
        complete_groups = [name for name, cats in qcd_groups.items() if len(cats) == 4]

        # nothing to do if there are no complete groups
        if not complete_groups:
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

        # initializing dictionary for later use
        dict_hists = {}

        for group_name in complete_groups:
            group = qcd_groups[group_name]

            # helper
            get_hist = lambda h, region_name: h[{"category": hist.loc(group[region_name].id)}]

            for region in control_regions:
                # get the corresponding histograms and convert them to number objects,
                # each one storing an array of values with uncertainties
                # shapes: (SHIFT, VAR)
                hist_mc = hist_to_num(get_hist(mc_hist, region), region + "_mc")
                hist_data = hist_to_num(get_hist(data_hist, region), region + "_data")
                hist_qcd = hist_data - hist_mc

                # save hists
                dict_hists[region + "_mc"] = (hist_mc)
                dict_hists[region + "_data"] = (hist_data)
                dict_hists[region + "_qcd"] = (hist_qcd)

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
            signal_region = getattr(group, sig_region)

            # sum over bins
            # shapes: (SHIFT,)
            int_num = integrate_num(num_region, axis=1)
            int_den = integrate_num(den_region, axis=1)

            # complain about negative integrals
            int_num_neg = int_num <= 0
            int_den_neg = int_den <= 0
            if int_num_neg.any():
                shift_ids = list(map(mc_hist.axes["shift"].value, np.where(int_num_neg)[0]))
                shifts = list(map(config.get_shift, shift_ids))
                logger.warning(
                    f"negative QCD integral in ss_iso region for group {group_name} and shifts: "
                    f"{', '.join(map(str, shifts))}",
                )
            if int_den_neg.any():
                shift_ids = list(map(mc_hist.axes["shift"].value, np.where(int_den_neg)[0]))
                shifts = list(map(config.get_shift, shift_ids))
                logger.warning(
                    f"negative QCD integral in ss_noniso region for group {group_name} and shifts: "
                    f"{', '.join(map(str, shifts))}",
                )

            # ABCD method
            # shape: (SHIFT, VAR)
            qcd_estimation = shape_estimation * ((int_num / int_den)[:, None])

            # combine uncertainties and store values in bare arrays
            qcd_estimation_values = qcd_estimation()
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

            # insert qcd estimation into signal region histograms
            cat_axis = qcd_hist.axes["category"]
            for cat_index in range(cat_axis.size):
                if cat_axis.value(cat_index) == signal_region.id:
                    qcd_hist.view().value[cat_index, ...] = qcd_estimation_values
                    qcd_hist.view().variance[cat_index, ...] = qcd_estimation_variances
                    break
            else:
                raise RuntimeError(
                    f"could not find index of bin on 'category' axis of qcd histogram {qcd_hist} "
                    f"for category {signal_region}",
                )

        return hists

    def factor(task, hists):
        """
        This hook calculates the per bin ratio factor for a choosen category.
        In the command line call --categories {etau,mutau,tautau}_{incl,2j,...}__os__iso
        """

        # SETUP
        # --------------------------------------------------------------------------------------
        control_regions = ["ss_iso", "os_noniso", "ss_noniso"]
        sig_region = "os_iso"
        # choose numerator and denominator for ratio factor calculation
        numerator = control_regions[1]
        denominator = control_regions[2]
        # --------------------------------------------------------------------------------------

        if len(control_regions) != 3:
            raise ValueError("Please define exactly 3 control regions!")
        if sig_region in control_regions:
            raise ValueError("Signal region must not be in control regions set!")
        if numerator == denominator:
            raise ValueError("Numerator and denominator for ratio factor cannot be the same region!")
        if not hists:
            return hists

        # get dummy processes
        factor_bin = config.get_process("qcd", default=None)
        if not factor_bin:
            return hists
        factor_int = config.get_process("dy", default=None)
        if not factor_int:
            return hists

        # extract all unique category ids and verify that the axis order is exactly
        # "category -> shift -> variable" which is needed to insert values at the end
        CAT_AXIS, SHIFT_AXIS, VAR_AXIS = range(3)
        category_ids = set()
        for proc, h in hists.items():
            # validate axes
            assert len(h.axes) == 3
            assert h.axes[CAT_AXIS].name == "category"
            assert h.axes[SHIFT_AXIS].name == "shift"
            # get the category axis
            cat_ax = h.axes["category"]
            for cat_index in range(cat_ax.size):
                category_ids.add(cat_ax.value(cat_index))

        # create qcd groups
        qcd_groups: dict[str, dict[str, od.Category]] = defaultdict(DotDict)
        for cat_id in category_ids:
            cat_inst = config.get_category(cat_id)
            if cat_inst.has_tag({"os", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_iso = cat_inst
            elif cat_inst.has_tag({"os", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_noniso = cat_inst
            elif cat_inst.has_tag({"ss", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_iso = cat_inst
            elif cat_inst.has_tag({"ss", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_noniso = cat_inst

        # get complete qcd groups
        complete_groups = [name for name, cats in qcd_groups.items() if len(cats) == 4]
        # nothing to do if there are no complete groups
        if not complete_groups:
            return hists

        # sum up mc and data histograms, stop early when empty
        mc_hists = [h for p, h in hists.items() if p.is_mc and not p.has_tag("signal")]
        data_hists = [h for p, h in hists.items() if p.is_data]
        if not mc_hists or not data_hists:
            return hists
        mc_hist = sum(mc_hists[1:], mc_hists[0].copy())
        data_hist = sum(data_hists[1:], data_hists[0].copy())

        # start by copying the mc hist and reset it, then fill it at specific category slices
        hists = {}
        hists[factor_bin] = factor_hist = mc_hist.copy().reset()
        hists[factor_int] = factor_hist_int = mc_hist.copy().reset()

        # initializing dictionary for later use
        dict_hists = {}

        for group_name in complete_groups:
            group = qcd_groups[group_name]
            # helper
            get_hist = lambda h, region_name: h[{"category": hist.loc(group[region_name].id)}]

            for region in control_regions:
                # get the corresponding histograms and convert them to number objects,
                # each one storing an array of values with uncertainties
                # shapes: (SHIFT, VAR)
                hist_mc = hist_to_num(get_hist(mc_hist, region), region + "_mc")
                hist_data = hist_to_num(get_hist(data_hist, region), region + "_data")
                hist_qcd = hist_data - hist_mc

                # save hists
                dict_hists[region + "_mc"] = (hist_mc)
                dict_hists[region + "_data"] = (hist_data)
                dict_hists[region + "_qcd"] = (hist_qcd)

            num_region = dict_hists[numerator + "_qcd"]
            den_region = dict_hists[denominator + "_qcd"]
            signal_region = getattr(group, sig_region)

            # calculate the ratio factor per bin
            factor = (num_region / den_region)[:, None]
            factor_values = np.squeeze(np.nan_to_num(factor()), axis=0)
            factor_variances = factor(sn.UP, sn.ALL, unc=True)**2

            # calculate the average ratio factor summed over bins
            int_num = integrate_num(num_region, axis=1)
            int_den = integrate_num(den_region, axis=1)
            factor_int = (int_num / int_den)[0, None]
            # change shape of factor_int for plotting
            factor_int_values = factor_values.copy()
            factor_int_values.fill(factor_int()[0])

            # insert qcd estimation into the qcd histogram
            cat_axis = factor_hist.axes["category"]
            for cat_index in range(cat_axis.size):
                if cat_axis.value(cat_index) == signal_region.id:
                    factor_hist.view().value[cat_index, ...] = factor_values
                    factor_hist.view().variance[cat_index, ...] = factor_variances
                    factor_hist_int.view().value[cat_index, ...] = factor_int_values
                    break
            else:
                raise RuntimeError(
                    f"could not find index of bin on 'category' axis of qcd histogram {factor_hist} "
                    f"for category {signal_region}",
                )
        return hists

    def factor_incl(task, hists):
        """
        This hook calculates the summed over ratio factor for a choosen kinematic category considering
        all decay channels simultaneously (e.g. etau + mutau + tautau)
        In the command line call --categories {incl,2j,...}__os__iso
        """

        # SETUP
        # --------------------------------------------------------------------------------------
        control_regions = ["ss_iso", "os_noniso", "ss_noniso"]
        decay_channels = ["etau", "mutau", "tautau"]
        kinematic_region = ["incl"]
        sig_region = "os_iso"
        # choose numerator and denominator for ratio factor calculation
        numerator = control_regions[1]
        denominator = control_regions[2]
        # --------------------------------------------------------------------------------------

        if len(control_regions) != 3:
            raise ValueError("Please define exactly 3 control regions!")
        if sig_region in control_regions:
            raise ValueError("Signal region must not be in control regions set!")
        if len(kinematic_region) != 1:
            raise ValueError("Only one kinematic region allowed!")
        if numerator == denominator:
            raise ValueError("Numerator and denominator for ratio factor cannot be the same region!")
        if not hists:
            return hists

        # get dummy processes
        factor_bin = config.get_process("qcd", default=None)
        if not factor_bin:
            return hists
        factor_int = config.get_process("dy", default=None)
        if not factor_int:
            return hists

        # extract all unique category ids and verify that the axis order is exactly
        # "category -> shift -> variable" which is needed to insert values at the end
        CAT_AXIS, SHIFT_AXIS, VAR_AXIS = range(3)
        category_ids = set()
        for proc, h in hists.items():
            # validate axes
            assert len(h.axes) == 3
            assert h.axes[CAT_AXIS].name == "category"
            assert h.axes[SHIFT_AXIS].name == "shift"
            # get the category axis
            cat_ax = h.axes["category"]
            for cat_index in range(cat_ax.size):
                category_ids.add(cat_ax.value(cat_index))

        # create qcd groups
        qcd_groups: dict[str, dict[str, od.Category]] = defaultdict(DotDict)
        for cat_id in category_ids:
            cat_inst = config.get_category(cat_id)
            if cat_inst.has_tag({"os", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_iso = cat_inst
            elif cat_inst.has_tag({"os", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_noniso = cat_inst
            elif cat_inst.has_tag({"ss", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_iso = cat_inst
            elif cat_inst.has_tag({"ss", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_noniso = cat_inst

        # get complete qcd groups
        complete_groups = [name for name, cats in qcd_groups.items() if len(cats) == 4]

        # nothing to do if there are no complete groups
        if not complete_groups:
            return hists

        # sum up mc and data histograms, stop early when empty
        mc_hists = [h for p, h in hists.items() if p.is_mc and not p.has_tag("signal")]
        data_hists = [h for p, h in hists.items() if p.is_data]
        if not mc_hists or not data_hists:
            return hists
        mc_hist = sum(mc_hists[1:], mc_hists[0].copy())
        data_hist = sum(data_hists[1:], data_hists[0].copy())

        # start by copying the mc hist and reset it, then fill it at specific category slices
        hists = {}
        hists[factor_bin] = factor_hist = mc_hist.copy().reset()
        hists[factor_int] = factor_hist_int = mc_hist.copy().reset()

        # initializing dictionaries for later use
        channels = {}
        dict_hists = {}

        for group_name in complete_groups:
            for k in kinematic_region:
                # loop should only run once, in the chosen kinematic region
                if k in group_name:
                    group = qcd_groups[group_name]
                    channels[group_name] = {}

                    # helper
                    get_hist = lambda h, region_name: h[{"category": hist.loc(group[region_name].id)}]

                    for region in control_regions:
                        # get hists per region and convert them to number objects
                        # each number object stores an array of values with uncertainties
                        # shapes: (SHIFT, VAR)
                        hist_mc = hist_to_num(get_hist(mc_hist, region), region + "_mc")
                        hist_data = hist_to_num(get_hist(data_hist, region), region + "_data")

                        channels[group_name][region + "_mc"] = hist_mc
                        channels[group_name][region + "_data"] = hist_data

        for group_name in complete_groups:
            for k in kinematic_region:
                # loop should only run once. use dummy group_name
                if group_name == f"{decay_channels[0]}__{k}":
                    group = qcd_groups[group_name]

                    # sum all channels into single hists
                    for region in control_regions:
                        hist_mc = sn.Number()
                        hist_data = sn.Number()
                        for c in decay_channels:
                            hist_mc += channels[f"{c}__{k}"][f"{region}_mc"]
                            hist_data += channels[f"{c}__{k}"][f"{region}_data"]

                        hist_qcd = hist_data - hist_mc

                        # save hists
                        dict_hists[region + "_mc"] = hist_mc
                        dict_hists[region + "_data"] = hist_data
                        dict_hists[region + "_qcd"] = hist_qcd

                    # define numerator/denominator for factor calculation
                    num_region = dict_hists[numerator + "_qcd"]
                    den_region = dict_hists[denominator + "_qcd"]
                    # define signal region
                    signal_region = getattr(group, sig_region)

                    # calculate the pt-independent fake factor
                    int_num = integrate_num(num_region, axis=1)
                    int_dem = integrate_num(den_region, axis=1)
                    factor_int = (int_num / int_dem)[0, None]

                    # calculate the pt-dependent fake factor
                    factor = (num_region / den_region)[:, None]
                    factor_values = np.squeeze(np.nan_to_num(factor()), axis=0)
                    factor_variances = factor(sn.UP, sn.ALL, unc=True)**2

                    # change shape of factor_int for plotting
                    factor_int_values = factor_values.copy()
                    factor_int_values.fill(factor_int()[0])

                    # insert values into the qcd histogram
                    cat_axis = factor_hist.axes["category"]
                    for cat_index in range(cat_axis.size):
                        if cat_axis.value(cat_index) == signal_region.id:
                            factor_hist.view().value[cat_index, ...] = factor_values
                            factor_hist.view().variance[cat_index, ...] = factor_variances
                            factor_hist_int.view().value[cat_index, ...] = factor_int_values
                            break
                    else:
                        raise RuntimeError(
                            f"could not find index of bin on 'category' axis of qcd histogram {factor_hist} "
                            f"for category {signal_region}",
                        )
        return hists

    # -----------------------------------------------------------------------------------------------------------

    # test ABCD method with pseudodata from scaled up ttbar MC
    def closure_test(task, hists):
        if not hists:
            return hists

        # get the qcd process
        qcd_proc = config.get_process("qcd", default=None)
        if not qcd_proc:
            return hists

        # get the data process
        data_proc = config.get_process("data", default=None)
        if not data_proc:
            return hists

        # extract all unique category ids and verify that the axis order is exactly
        # "category -> shift -> variable" which is needed to insert values at the end
        CAT_AXIS, SHIFT_AXIS, VAR_AXIS = range(3)
        category_ids = set()
        for proc, h in hists.items():
            # validate axes
            assert len(h.axes) == 3
            assert h.axes[CAT_AXIS].name == "category"
            assert h.axes[SHIFT_AXIS].name == "shift"
            # get the category axis
            cat_ax = h.axes["category"]
            for cat_index in range(cat_ax.size):
                category_ids.add(cat_ax.value(cat_index))

        # create qcd groups
        qcd_groups: dict[str, dict[str, od.Category]] = defaultdict(DotDict)
        for cat_id in category_ids:
            cat_inst = config.get_category(cat_id)
            if cat_inst.has_tag({"os", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_iso = cat_inst
            elif cat_inst.has_tag({"os", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_noniso = cat_inst
            elif cat_inst.has_tag({"ss", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_iso = cat_inst
            elif cat_inst.has_tag({"ss", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_noniso = cat_inst

        # get complete qcd groups
        complete_groups = [name for name, cats in qcd_groups.items() if len(cats) == 4]

        # nothing to do if there are no complete groups
        if not complete_groups:
            return hists

        # get mc, data and ttbar histograms
        mc_hists = [h for p, h in hists.items() if p.is_mc and not p.has_tag("signal")]
        tt_hists = [h for p, h in hists.items() if p.has_tag("ttbar")]
        data_hists = [h for p, h in hists.items() if p.is_data]

        # sum up hists, stop early when empty
        if not mc_hists or not tt_hists or not data_hists:
            return hists
        mc_hist = sum(mc_hists[1:], mc_hists[0].copy())
        tt_hist = sum(tt_hists[1:], tt_hists[0].copy())
        data_hist = sum(data_hists[1:], data_hists[0].copy())

        # use MC with twice ttbar as pseudo data for the closure test
        data_hist = mc_hist + tt_hist

        # start by copying the mc hist and reset it, then fill it at specific category slices
        hists[qcd_proc] = qcd_hist = mc_hist.copy().reset()
        hists[data_proc] = data_hist

        # remove the original data histogram from hists
        hists_to_delete = []
        for idx, (p, h) in enumerate(hists.items()):
            if p.is_data and idx == 0:
                hists_to_delete.append(p)

        for h in hists_to_delete:
            del hists[h]

        for group_name in complete_groups:
            group = qcd_groups[group_name]

            # get the corresponding histograms and convert them to number objects,
            # each one storing an array of values with uncertainties
            # shapes: (SHIFT, VAR)
            get_hist = lambda h, region_name: h[{"category": hist.loc(group[region_name].id)}]
            os_noniso_mc = hist_to_num(get_hist(mc_hist, "os_noniso"), "os_noniso_mc")
            ss_noniso_mc = hist_to_num(get_hist(mc_hist, "ss_noniso"), "ss_noniso_mc")
            ss_iso_mc = hist_to_num(get_hist(mc_hist, "ss_iso"), "ss_iso_mc")
            os_noniso_data = hist_to_num(get_hist(data_hist, "os_noniso"), "os_noniso_data")
            ss_noniso_data = hist_to_num(get_hist(data_hist, "ss_noniso"), "ss_noniso_data")
            ss_iso_data = hist_to_num(get_hist(data_hist, "ss_iso"), "ss_iso_data")

            # estimate qcd shapes in the three sideband regions
            # shapes: (SHIFT, VAR)
            os_noniso_qcd = os_noniso_data - os_noniso_mc
            ss_iso_qcd = ss_iso_data - ss_iso_mc
            ss_noniso_qcd = ss_noniso_data - ss_noniso_mc

            # get integrals in ss regions for the transfer factor
            # shapes: (SHIFT,)
            int_ss_iso = integrate_num(ss_iso_qcd, axis=1)
            int_ss_noniso = integrate_num(ss_noniso_qcd, axis=1)

            # complain about negative integrals
            int_ss_iso_neg = int_ss_iso <= 0
            int_ss_noniso_neg = int_ss_noniso <= 0
            if int_ss_iso_neg.any():
                shift_ids = list(map(mc_hist.axes["shift"].value, np.where(int_ss_iso_neg)[0]))
                shifts = list(map(config.get_shift, shift_ids))
                logger.warning(
                    f"negative QCD integral in ss_iso region for group {group_name} and shifts: "
                    f"{', '.join(map(str, shifts))}",
                )
            if int_ss_noniso_neg.any():
                shift_ids = list(map(mc_hist.axes["shift"].value, np.where(int_ss_noniso_neg)[0]))
                shifts = list(map(config.get_shift, shift_ids))
                logger.warning(
                    f"negative QCD integral in ss_noniso region for group {group_name} and shifts: "
                    f"{', '.join(map(str, shifts))}",
                )

            # ABCD method
            # shape: (SHIFT, VAR)
            os_iso_qcd = os_noniso_qcd * ((int_ss_iso / int_ss_noniso)[:, None])

            # combine uncertainties and store values in bare arrays
            os_iso_qcd_values = os_iso_qcd()
            os_iso_qcd_variances = os_iso_qcd(sn.UP, sn.ALL, unc=True)**2

            # define uncertainties
            unc_data = os_iso_qcd(sn.UP, ["os_noniso_data", "ss_iso_data", "ss_noniso_data"], unc=True)
            unc_mc = os_iso_qcd(sn.UP, ["os_noniso_mc", "ss_iso_mc", "ss_noniso_mc"], unc=True)
            unc_data_rel = abs(unc_data / os_iso_qcd_values)
            unc_mc_rel = abs(unc_mc / os_iso_qcd_values)

            # only keep the MC uncertainty if it is larger than the data uncertainty and larger than 15%
            keep_variance_mask = (
                np.isfinite(unc_mc_rel) &
                (unc_mc_rel > unc_data_rel) &
                (unc_mc_rel > 0.15)
            )
            os_iso_qcd_variances[keep_variance_mask] = unc_mc[keep_variance_mask]**2
            os_iso_qcd_variances[~keep_variance_mask] = 0

            # retro-actively set values to zero for shifts that had negative integrals
            neg_int_mask = int_ss_iso_neg | int_ss_noniso_neg
            os_iso_qcd_values[neg_int_mask] = 1e-5
            os_iso_qcd_variances[neg_int_mask] = 0

            # residual zero filling
            zero_mask = os_iso_qcd_values <= 0
            os_iso_qcd_values[zero_mask] = 1e-5
            os_iso_qcd_variances[zero_mask] = 0

            # insert values into the qcd histogram
            cat_axis = qcd_hist.axes["category"]
            for cat_index in range(cat_axis.size):
                if cat_axis.value(cat_index) == group.os_iso.id:
                    qcd_hist.view().value[cat_index, ...] = os_iso_qcd_values
                    qcd_hist.view().variance[cat_index, ...] = os_iso_qcd_variances
                    break
            else:
                raise RuntimeError(
                    f"could not find index of bin on 'category' axis of qcd histogram {qcd_hist} "
                    f"for category {group.os_iso}",
                )

        return hists

    def qcd_validation(task, hists):
        """
        This hook calculates qcd estimation (shape x factor) for a choosen category.
        In the command line call --categories {etau,mutau,tautau}_{incl,2j,...}__os__iso
        """

        # SETUP
        # --------------------------------------------------------------------------------------
        control_regions = ["ss_iso", "os_noniso", "ss_noniso"]
        sig_region = "os_iso"
        # choose region to extract qcd shape estimation
        shape_region = control_regions[1]
        # choose numerator and denominator for ratio factor calculation
        numerator = control_regions[0]
        denominator = control_regions[2]
        # choose MC minimum uncertainty theshold. Uncertainties bellow min_mc_unc are disregraded
        min_mc_unc = 0.15
        # --------------------------------------------------------------------------------------

        if len(control_regions) != 3:
            raise ValueError("Please define exactly 3 control regions!")
        if sig_region in control_regions:
            raise ValueError("Signal region must not be in control regions set!")
        if shape_region == numerator or shape_region == denominator:
            raise ValueError("shape_region must not be the same as numerator or denominator!")
        if numerator == denominator:
            raise ValueError("Numerator and denominator for ratio factor cannot be the same region!")
        if not hists:
            return hists

        # get the qcd process
        qcd_proc = config.get_process("qcd", default=None)
        if not qcd_proc:
            return hists

        # extract all unique category ids and verify that the axis order is exactly
        # "category -> shift -> variable" which is needed to insert values at the end
        CAT_AXIS, SHIFT_AXIS, VAR_AXIS = range(3)
        category_ids = set()
        for proc, h in hists.items():
            # validate axes
            assert len(h.axes) == 3
            assert h.axes[CAT_AXIS].name == "category"
            assert h.axes[SHIFT_AXIS].name == "shift"
            # get the category axis
            cat_ax = h.axes["category"]
            for cat_index in range(cat_ax.size):
                category_ids.add(cat_ax.value(cat_index))

        # create qcd groups
        qcd_groups: dict[str, dict[str, od.Category]] = defaultdict(DotDict)
        for cat_id in category_ids:
            cat_inst = config.get_category(cat_id)
            if cat_inst.has_tag({"os", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_iso = cat_inst
            elif cat_inst.has_tag({"os", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_noniso = cat_inst
            elif cat_inst.has_tag({"ss", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_iso = cat_inst
            elif cat_inst.has_tag({"ss_pos", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_pos_iso = cat_inst
            elif cat_inst.has_tag({"ss_neg", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_neg_iso = cat_inst
            elif cat_inst.has_tag({"ss", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_noniso = cat_inst
            elif cat_inst.has_tag({"ss_pos", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_pos_noniso = cat_inst
            elif cat_inst.has_tag({"ss_neg", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_neg_noniso = cat_inst

        # get complete qcd groups
        complete_groups = [name for name, cats in qcd_groups.items() if len(cats) == 4]

        # nothing to do if there are no complete groups
        if not complete_groups:
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

        # initializing dictionary for later use
        dict_hists = {}

        for group_name in complete_groups:
            group = qcd_groups[group_name]

            # helper
            get_hist = lambda h, region_name: h[{"category": hist.loc(group[region_name].id)}]

            for region in control_regions:
                # get the corresponding histograms and convert them to number objects,
                # each one storing an array of values with uncertainties
                # shapes: (SHIFT, VAR)
                hist_mc = hist_to_num(get_hist(mc_hist, region), region + "_mc")
                hist_data = hist_to_num(get_hist(data_hist, region), region + "_data")
                hist_qcd = hist_data - hist_mc

                # save hists
                dict_hists[region + "_mc"] = (hist_mc)
                dict_hists[region + "_data"] = (hist_data)
                dict_hists[region + "_qcd"] = (hist_qcd)

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
            signal_region = getattr(group, sig_region)

            # sum over bins
            # shapes: (SHIFT,)
            int_num = integrate_num(num_region, axis=1)
            int_den = integrate_num(den_region, axis=1)

            # complain about negative integrals
            int_num_neg = int_num <= 0
            int_den_neg = int_den <= 0
            if int_num_neg.any():
                shift_ids = list(map(mc_hist.axes["shift"].value, np.where(int_num_neg)[0]))
                shifts = list(map(config.get_shift, shift_ids))
                logger.warning(
                    f"negative QCD integral in ss_iso region for group {group_name} and shifts: "
                    f"{', '.join(map(str, shifts))}",
                )
            if int_den_neg.any():
                shift_ids = list(map(mc_hist.axes["shift"].value, np.where(int_den_neg)[0]))
                shifts = list(map(config.get_shift, shift_ids))
                logger.warning(
                    f"negative QCD integral in ss_noniso region for group {group_name} and shifts: "
                    f"{', '.join(map(str, shifts))}",
                )

            # ABCD method
            # shape: (SHIFT, VAR)
            qcd_estimation = shape_estimation * ((int_num / int_den)[:, None])

            # combine uncertainties and store values in bare arrays
            qcd_estimation_values = qcd_estimation()
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

            # insert qcd estimation into signal region histograms
            cat_axis = qcd_hist.axes["category"]
            for cat_index in range(cat_axis.size):
                if cat_axis.value(cat_index) == signal_region.id:
                    qcd_hist.view().value[cat_index, ...] = qcd_estimation_values
                    qcd_hist.view().variance[cat_index, ...] = qcd_estimation_variances
                    break
            else:
                raise RuntimeError(
                    f"could not find index of bin on 'category' axis of qcd histogram {qcd_hist} "
                    f"for category {signal_region}",
                )

        return hists

    def pos_factor(task, hists):
        """
        This hook calculates the per bin ratio factor for a choosen category.
        In the command line call --categories {etau,mutau,tautau}_{incl,2j,...}__os__iso
        """

        # SETUP
        # --------------------------------------------------------------------------------------
        control_regions_pos = ["ss_pos_iso", "os_noniso", "ss_pos_noniso"]  # noqa: F841
        control_regions_neg = ["ss_neg_iso", "os_noniso", "ss_neg_noniso"]  # noqa: F841
        sig_region = "os_iso"
        # choose which control regions to use (SS+ or SS-)
        control_regions = control_regions_pos
        # choose numerator and denominator for ratio factor calculation
        numerator = control_regions[1]
        denominator = control_regions[2]
        # --------------------------------------------------------------------------------------

        if len(control_regions) != 3:
            raise ValueError("Please define exactly 3 control regions!")
        if sig_region in control_regions:
            raise ValueError("Signal region must not be in control regions set!")
        if numerator == denominator:
            raise ValueError("Numerator and denominator for ratio factor cannot be the same region!")
        if not hists:
            return hists

        # get dummy processes
        factor_bin = config.get_process("qcd", default=None)
        if not factor_bin:
            return hists
        factor_int = config.get_process("dy", default=None)
        if not factor_int:
            return hists

        # extract all unique category ids and verify that the axis order is exactly
        # "category -> shift -> variable" which is needed to insert values at the end
        CAT_AXIS, SHIFT_AXIS, VAR_AXIS = range(3)
        category_ids = set()
        for proc, h in hists.items():
            # validate axes
            assert len(h.axes) == 3
            assert h.axes[CAT_AXIS].name == "category"
            assert h.axes[SHIFT_AXIS].name == "shift"
            # get the category axis
            cat_ax = h.axes["category"]
            for cat_index in range(cat_ax.size):
                category_ids.add(cat_ax.value(cat_index))

        # create qcd groups
        qcd_groups: dict[str, dict[str, od.Category]] = defaultdict(DotDict)
        for cat_id in category_ids:
            cat_inst = config.get_category(cat_id)
            if cat_inst.has_tag({"os", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_iso = cat_inst
            elif cat_inst.has_tag({"os", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_noniso = cat_inst
            elif cat_inst.has_tag({"ss", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_iso = cat_inst
            elif cat_inst.has_tag({"ss", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_noniso = cat_inst
            elif cat_inst.has_tag({"ss_pos", "iso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_pos_iso = cat_inst
            elif cat_inst.has_tag({"ss_pos", "noniso"}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_pos_noniso = cat_inst

        # get complete qcd groups
        complete_groups = [name for name, cats in qcd_groups.items() if len(cats) == 4]
        # nothing to do if there are no complete groups
        if not complete_groups:
            return hists

        # sum up mc and data histograms, stop early when empty
        mc_hists = [h for p, h in hists.items() if p.is_mc and not p.has_tag("signal")]
        data_hists = [h for p, h in hists.items() if p.is_data]
        if not mc_hists or not data_hists:
            return hists
        mc_hist = sum(mc_hists[1:], mc_hists[0].copy())
        data_hist = sum(data_hists[1:], data_hists[0].copy())

        # start by copying the mc hist and reset it, then fill it at specific category slices
        hists = {}
        hists[factor_bin] = factor_hist = mc_hist.copy().reset()
        hists[factor_int] = factor_hist_int = mc_hist.copy().reset()

        # initializing dictionary for later use
        dict_hists = {}

        from IPython import embed
        embed()
        for group_name in complete_groups:
            group = qcd_groups[group_name]
            # helper
            get_hist = lambda h, region_name: h[{"category": hist.loc(group[region_name].id)}]

            for region in control_regions:
                # get the corresponding histograms and convert them to number objects,
                # each one storing an array of values with uncertainties
                # shapes: (SHIFT, VAR)
                hist_mc = hist_to_num(get_hist(mc_hist, region), region + "_mc")
                hist_data = hist_to_num(get_hist(data_hist, region), region + "_data")
                hist_qcd = hist_data - hist_mc

                # save hists
                dict_hists[region + "_mc"] = (hist_mc)
                dict_hists[region + "_data"] = (hist_data)
                dict_hists[region + "_qcd"] = (hist_qcd)

            num_region = dict_hists[numerator + "_qcd"]
            den_region = dict_hists[denominator + "_qcd"]
            signal_region = getattr(group, sig_region)

            # calculate the ratio factor per bin
            factor = (num_region / den_region)[:, None]
            factor_values = np.squeeze(np.nan_to_num(factor()), axis=0)
            factor_variances = factor(sn.UP, sn.ALL, unc=True)**2

            # calculate the average ratio factor summed over bins
            int_num = integrate_num(num_region, axis=1)
            int_den = integrate_num(den_region, axis=1)
            factor_int = (int_num / int_den)[0, None]
            # change shape of factor_int for plotting
            factor_int_values = factor_values.copy()
            factor_int_values.fill(factor_int()[0])

            # insert qcd estimation into the qcd histogram
            cat_axis = factor_hist.axes["category"]
            for cat_index in range(cat_axis.size):
                if cat_axis.value(cat_index) == signal_region.id:
                    factor_hist.view().value[cat_index, ...] = factor_values
                    factor_hist.view().variance[cat_index, ...] = factor_variances
                    factor_hist_int.view().value[cat_index, ...] = factor_int_values
                    break
            else:
                raise RuntimeError(
                    f"could not find index of bin on 'category' axis of qcd histogram {factor_hist} "
                    f"for category {signal_region}",
                )
        return hists

    # add all hooks
    config.x.hist_hooks.abcd_stats = abcd_stats
    config.x.hist_hooks.qcd = qcd_estimation
    config.x.hist_hooks.factor = factor
    config.x.hist_hooks.factor_incl = factor_incl

    # in development
    config.x.hist_hooks.closure = closure_test
    config.x.hist_hooks.pos_factor = pos_factor
    config.x.hist_hooks.qcd_validation = qcd_validation
