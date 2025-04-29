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

    def abcd_stats(
        task,
        hists,
        all_channels: bool = False,
        is_validation: bool = True,
    ):
        """
        hist hook to plot the statistics in each of the ABCD qcd regions.

        In the command line always call --categories all_incl

        To plot the ABCD regions for all decay channels, flag *all_channels* to True.
        Else, specify the decay channel in the *channel* variable in SETUP.

        Note:
        The plotting style of the x-axis must be set in l190 of columnflow/columnflow/tasks/plotting.py
        over/underflow of histograms must be commented out in l287-290 of columnflow/columnflow/plotting/plot_util.py
        """

        # SETUP
        # --------------------------------------------------------------------------------------
        # choose the decay channel
        channel = "tautau"

        # define kinematic regions
        kin_region = "incl"

        # define sign regions
        sign_region = ["os", "ss"]

        # define isolation regions
        if is_validation is False:
            iso_region = ["iso", "noniso"]
        elif is_validation is True:
            iso_region = ["vvl_vl", "vvvl_vvl"]
        # --------------------------------------------------------------------------------------

        cats_to_plot = [f"{kin_region}__{s}__{i}" for s in sign_region for i in iso_region]

        if all_channels is False:
            cats_to_plot = [f"{channel}__{c}" for c in cats_to_plot]

        cats = [task.config_inst.get_category(c) for c in cats_to_plot]

        # initialize objects for later use
        results = {}
        CAT_AXIS, SHIFT_AXIS, VAR_AXIS = range(3)
        category_ids = set()

        # get the histograms for each ABCD regions
        for proc, h in hists.items():
            # validate axes
            assert len(h.axes) == 3
            assert h.axes[CAT_AXIS].name == "category"
            assert h.axes[SHIFT_AXIS].name == "shift"
            # get the category axis
            cat_ax = h.axes["category"]
            for cat_index in range(cat_ax.size):
                category_ids.add(cat_ax.value(cat_index))

            # create new histogram to be filled with event statistics
            h_new = hist.Hist(
                hist.axes.StrCategory([c.name for c in cats], name=h.axes[-1].name),
                hist.axes.IntCategory([0], name="shift"),
                hist.axes.IntCategory([101], name="category"),  # 101: all_incl. do not change!
                storage=hist.storage.Weight(),
            )

            for ind, big_cat in enumerate(cats):
                # get histograms from all decay channels and sum them up
                if all_channels is True:
                    h_sum = h[
                        {
                            "category": [hist.loc(cat.id) for cat in big_cat.get_leaf_categories()],
                            "shift": sum,
                        }
                    ].sum()

                # get histograms from single decay channel

                elif (all_channels is False and channel in big_cat.name):
                    h_sum = h[
                        {
                            "category": [hist.loc(big_cat.id)],
                            "shift": sum,
                        }
                    ].sum()

                # fill histogram
                h_new.values()[ind][0][0] = h_sum.value
                h_new.variances()[ind][0][0] = h_sum.variance

            # save results
            results[proc] = h_new
        return results

    def factor_shape(
        task,
        hists,
        perbin: bool = False,
        method_I: bool = False,
        qcd_shape: bool = False,
        is_validation: bool = True,
        return_hists: bool = True,
        iso_tag_name: str = "iso",
        noniso_tag_name: str = "noniso",
    ):
        """
        This hook calculates the qcd estimation (shape x factor) for a choosen decay channel
        and kinematic category and plots it.

        If perbin is set to True, factor values will be applied per bin, otherwise a single summed factor
        will be applied to all bins.

        The qcd estimation can be calculated by two methods, depending on which control region is choosen to extract
        the shape estimation, and which one is choosen as the numerator for the factor calculation. This can be
        controlled by the method_I boolean.

        If qcd_shape is set to True, the final qcd estimation histogram (shape * factor) will be plotted.
        Otherwise, only the factors will be plotted.

        In the command line call --categories {etau,mutau,tautau}__{incl,2j,...}__os__{iso,vvl_vl,...}
        """

        # SETUP
        # --------------------------------------------------------------------------------------
        # choose decay channel and kinematic region
        channel = "tautau"
        kin_region = "incl"
        # choose category tags to get histograms
        if is_validation is False:
            iso_tag = "iso"
            noniso_tag = "noniso"
        elif is_validation is True:
            iso_tag = iso_tag_name
            noniso_tag = noniso_tag_name
        os_tag = "os"
        ss_tag = "ss"
        # --------------------------------------------------------------------------------------

        # defining ABCD regions
        control_regions = ["ss_iso", "os_noniso", "ss_noniso"]
        sig_region = "os_iso"
        denominator = "ss_noniso"
        # defining ABCD methods
        if method_I:
            shape_region = "os_noniso"
            numerator = "ss_iso"
        elif not method_I:
            shape_region = "ss_iso"
            numerator = "os_noniso"

        # sanity checks
        if shape_region == numerator or shape_region == denominator:
            raise ValueError("shape_region must not be the same as numerator or denominator!")
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
            # get qcd groups for a single decay channel in a specific kinematic region
            if f"{channel}__{kin_region}" in cat_inst.name:
                if cat_inst.has_tag({os_tag, iso_tag}, mode=all):
                    qcd_groups[cat_inst.x.qcd_group].os_iso = cat_inst
                elif cat_inst.has_tag({os_tag, noniso_tag}, mode=all):
                    qcd_groups[cat_inst.x.qcd_group].os_noniso = cat_inst
                elif cat_inst.has_tag({ss_tag, iso_tag}, mode=all):
                    qcd_groups[cat_inst.x.qcd_group].ss_iso = cat_inst
                elif cat_inst.has_tag({ss_tag, noniso_tag}, mode=all):
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
            shape_estimation = dict_hists[shape_region + "_qcd"]
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
            factor_int_variances = factor_int(sn.UP, sn.ALL, unc=True)**2

            # -----------------------------------------------
            # ABCD method
            # shape: (SHIFT, VAR)
            if perbin:
                qcd_estimation = shape_estimation * factor
            elif not perbin:
                qcd_estimation = shape_estimation * factor_int

            qcd_estimation_values = np.squeeze(np.nan_to_num(qcd_estimation()), axis=0)
            qcd_estimation_variances = qcd_estimation(sn.UP, sn.ALL, unc=True)**2
            shape_estimation_values = np.squeeze(np.nan_to_num(shape_estimation()), axis=0)
            shape_estimation_variances = shape_estimation(sn.UP, sn.ALL, unc=True)**2

            if qcd_shape:
                plot_values = qcd_estimation_values
                plot_variances = qcd_estimation_variances
                plot_values_v2 = shape_estimation_values
                plot_variances_v2 = shape_estimation_variances
            elif not qcd_shape:
                plot_values = factor_values
                plot_variances = factor_variances
                plot_values_v2 = factor_int_values
                plot_variances_v2 = factor_int_variances

            # insert qcd estimation into the qcd histogram
            cat_axis = factor_hist.axes["category"]
            for cat_index in range(cat_axis.size):
                if cat_axis.value(cat_index) == signal_region.id:
                    factor_hist.view().value[cat_index, ...] = plot_values
                    factor_hist.view().variance[cat_index, ...] = plot_variances
                    factor_hist_int.view().value[cat_index, ...] = plot_values_v2
                    factor_hist_int.view().variance[cat_index, ...] = plot_variances_v2
                    break
            else:
                raise RuntimeError(
                    f"could not find index of bin on 'category' axis of qcd histogram {factor_hist} "
                    f"for category {signal_region}",
                )
        if return_hists:
            print("-----------------------------")
            print("Int Factor: ", factor_int)
            print("Int Factor variances: ", factor_int_variances)
            print("Int num: ", int_num)
            print("Int den: ", int_den)
            print("-----------------------------")
            print("Int Factor values: ", plot_values)
            print("Int Factor variances: ", plot_variances)
            print("Int Factor values (per bin): ", plot_values_v2)
            print("Int Factor variances (per bin): ", plot_variances_v2)
            return hists
        else:
            return factor_int, factor_int_variances, int_num, int_den

    def get_WP_factors(task, hists):

        tags = [
            ("vvloose_vloose", "vvvloose_vvloose"),
            ("vloose_loose", "vvloose_vloose"),
            ("loose_medium", "vloose_loose"),
            ("iso", "loose_medium"),
            ("iso", "noniso"),
        ]

        for tag in tags:
            values = factor_shape(
                task,
                hists,
                perbin=False,
                method_I=True,
                qcd_shape=False,
                is_validation=True,
                return_hists=False,
                iso_tag_name=tag[0],
                noniso_tag_name=tag[1],
            )

            factor_int = values[0]
            factor_int_variances = values[1]
            int_num = values[2]
            int_den = values[3]

            print("-----------------------------")
            print("iso Tag: ", tag[0])
            print("noniso Tag: ", tag[1])
            print("Int Factor: ", factor_int)
            print("Int Factor variances: ", factor_int_variances)
            print("Int num: ", int_num)
            print("Int den: ", int_den)

        """
        This hook calculates the qcd estimation (shape x factor) for a choosen decay channel
        and kinematic category and adds it to all other existing MC histograms.

        If *perbin* is set to True, factor values will be applied per bin, otherwise a single summed factor
        will be applied to all bins.

        The qcd estimation can be calculated by two methods, depending on which control region is choosen to extract
        the shape estimation, and which one is choosen as the numerator for the factor calculation.
        The methods can be switched by setting the *method_I* boolean.

        In the command line call --categories {etau,mutau,tautau}_{incl,2j,...}__os__iso
        """

        # SETUP
        # --------------------------------------------------------------------------------------
        # choose decay channel and kinematic region
        channel = "tautau"
        kin_region = "incl"
        # choose category tags to get histograms
        if is_validation is False:
            iso_tag = "iso"
            noniso_tag = "noniso"
        elif is_validation is True:
            iso_tag = "vvloose_vloose"
            noniso_tag = "vvvloose_vvloose"
        os_tag = "os"
        ss_tag = "ss"
        # choose MC minimum uncertainty theshold. Uncertainties bellow min_mc_unc are disregraded
        min_mc_unc = 0.15
        # --------------------------------------------------------------------------------------

        # defining ABCD regions
        control_regions = ["ss_iso", "os_noniso", "ss_noniso"]
        sig_region = "os_iso"
        denominator = "ss_noniso"
        # defining ABCD methods
        if method_I:
            shape_region = "os_noniso"
            numerator = "ss_iso"
        elif not method_I:
            shape_region = "ss_iso"
            numerator = "os_noniso"

        # sanity checks
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
            # get qcd groups for all decay channels in a specific kinematic region
            if cat_inst.has_tag({os_tag, iso_tag}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_iso = cat_inst
            elif cat_inst.has_tag({os_tag, noniso_tag}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].os_noniso = cat_inst
            elif cat_inst.has_tag({ss_tag, iso_tag}, mode=all):
                qcd_groups[cat_inst.x.qcd_group].ss_iso = cat_inst
            elif cat_inst.has_tag({ss_tag, noniso_tag}, mode=all):
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
        hists[qcd_proc] = qcd_hist = mc_hist.copy().reset()

        # initializing dictionary for later use
        dict_hists = {}

        # helper function to get the corresponding histograms in single decay channels
        # if all_channels is False:
        #   get_hist = lambda h, group, region_name: h[{"category": hist.loc(group[region_name].id)}]
        # elif all_channels is True:
        #   channels_id = []
        #   # get all decay channels for a single kinematic region
        #   filtered_groups = [group_name for group_name in complete_groups if f"{kin_region}" in group_name]
        #   for group_name in filtered_groups:
        #       for region_name in control_regions:
        #           group = qcd_groups[group_name]
        #           channel_id = group[region_name].id
        #           print(group[region_name])
        #           channels_id.append(channel_id)
        #        get_hist = lambda h, region_name: h[{"category": hist.loc(id) for id in channels_id}].sum("category")  #noqa: E501

        # calculate the qcd estimation for a sing decay channels in a specific kinematic region
        for group_name in complete_groups:
            if (all_channels is False and group_name == f"{channel}__{kin_region}"):
                group = qcd_groups[group_name]

                # helper function to get the corresponding histograms in single decay channels
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

                # ------------------------------------------------------
                # ABCD method
                # shape: (SHIFT, VAR)
                factor_int = (int_num / int_den)[0, None]
                factor = (num_region / den_region)[:, None]
                if perbin:
                    qcd_estimation = shape_estimation * factor
                elif not perbin:
                    qcd_estimation = shape_estimation * factor_int
                # ------------------------------------------------------

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

                # insert qcd estimation into signal region histogram
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

        print(qcd_estimation_values)
        return hists

    # add all hooks
    config.x.hist_hooks.abcd_stats = abcd_stats
    config.x.hist_hooks.factor_shape = factor_shape
    config.x.hist_hooks.get_WP_factors = get_WP_factors
