# coding: utf-8

"""
Histogram hooks.
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


def add_hist_hooks(config: od.Config) -> None:
    """
    Add histogram hooks to a configuration.
    """
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

    def qcd_estimation(task, hists):
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
        hists[qcd_proc] = qcd_hist = mc_hist.copy().reset()
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

<<<<<<< Updated upstream
=======
    def flat_s(task ,hists: Dict[hist.Histogram]) -> Dict[hist.Histogram]:
        """Rebinnig of the signal histogram to have a flat distribution in the DNN output.

        Args:
            task (_type_): _description_
            hists (Dict[hist.Histogram]): A dictionary of histograms using Process instances as keys

        Returns:
            Dict[hist.Histogram]: A dictionary of histograms using Process instances as keys
        """
        def find_edges(signal_x, signal_y, background_x, background_y, n_bins) -> Tuple[np.ndarray, np.ndarray]:
            """
            Helper function to determine new bin edges that result in a flat signal distribution.
            The edges are determined by the signal distribution, while the background distribution
            is used to ensure that the background yield in each bin is sufficient.
            """
            def get_integral(cumulative_weights, stop, offset=0):
                """
                Helper to calculate the integral of *cumulative_weights* between the *offset* (included)
                and the *stop* index (not included)
                """
                return cumulative_weights[stop - 1] - (0 if offset == 0 else cumulative_weights[offset - 1])

            def print_flat(cumulative, indices):
                """
                TODO REMOVE at the END
                Helper to print the flat signal weights.
                """
                stop = indices
                start = np.insert(indices[:-1],0,0.)
                for i,j in zip(start,stop):
                    print(f"Integral for offset: {i}, stop:{j}: ",get_integral(cumulative, stop=j, offset=i))

            def count_events(arr, edges, weight=None):
                """
                TODO REMOVE at the END
                Helper function to quickly count events in intervals defined by *edges* in an array *arr*.
                If *weight* is given, the events are weighted accordingly.
                """
                if weight is None:
                    weight = np.ones_like(arr)

                for ind in range(len(edges)-1):
                    above = arr > edges[ind]
                    below = arr < edges[ind + 1]
                    events_in_interval = np.sum((above & below) * weight)
                    print(f"{edges[ind]} < x < {edges[ind+1]}: {events_in_interval}")

            breakpoint(header="find_edges")
            # flip arrays to start from the right
            r_signal_x, r_signal_y = np.flip(signal_x, axis=-1), np.flip(signal_y, axis=-1)
            r_cumulative_y = np.cumsum(r_signal_y, axis=0)
            # prepare parameters
            low_edge, max_edge = 0, 1
            bin_edges = [max_edge, ]
            num_events = len(r_cumulative_y)
            n_bins = 10
            # calculate desired yield per bin, last bin of cum is the total yield
            full_cum = r_cumulative_y[-1]
            y_per_bin = full_cum / n_bins
            # accumulated signal yield up to the current index
            already_binned_y = 0.0
            min_y = 1.0e-5
            # during binning, do not remove leading entries, but remember the index that denotes the start of the bin
            offset = 0
            # bookkeep reasons for stopping binning
            stop_reason = ""

            indices_gathering = [0, ]

            # start binning
            while len(bin_edges) < n_bins:
                # stopping condition 1: reached end of events
                if offset >= num_events:
                    stop_reason = "no more events left"
                    break
                # stopping condition 2: remaining signal yield too small
                # this would lead to a bin complelty filled with background
                remaining_y = full_cum - already_binned_y
                if remaining_y < min_y:
                    stop_reason = "remaining signal yield insufficient"
                    break
                # find the index of the event that would result in a signal yield increase of more
                # than the expected per-bin yield;
                # this index would mark the start of the next bin given all constraints are met
                if remaining_y >= y_per_bin:
                    threshold = already_binned_y + y_per_bin
                    print(f"threshold: {threshold}")
                    # get indices of array of values above threshold
                    # first entry defines the next bin edge
                    # shift next idx by offset
                    next_idx = offset + np.where(r_cumulative_y[offset:] > threshold)[0][0]
                else:
                    breakpoint(header="SPECIAL-CASE")
                    # special case: remaining signal yield smaller than the expected per-bin yield,
                    # so find the last event
                    next_idx = offset + np.where(r_cumulative_y[offset:])[0][-1] + 1

                # advance the index until backgrounds constraints are met
                #breakpoint(header="find_new_edge - background")
                    # Background constraints
                # while next_idx < num_events:
                #     # get the number of tt events and their yield
                #     n_tt = get_integral(rec.tt_count_cs, next_idx)
                #     y_tt = get_integral(rec.tt_weight_cs, next_idx)
                #     # get the number of dy events and their yield
                #     n_dy = get_integral(rec.dy_count_cs, next_idx)
                #     y_dy = get_integral(rec.dy_weight_cs, next_idx)
                #     # evaluate constraints
                #     # TODO: potentially relax constraints here, e.g when there are 3 (4?) tt events, drop the constraint
                #     #       on dy, and vice-versa
                #     constraints_met = (
                #         # tt and dy events
                #         n_tt >= 1 and
                #         n_dy >= 1 and
                #         n_tt + n_dy >= 4 and
                #         # yields must be positive to avoid negative sums of weights per process
                #         y_tt > 0 and
                #         y_dy > 0
                #     )
                #     if constraints_met:
                #         # TODO: maybe also check if the background conditions are just barely met and advance next_idx
                #         # to the middle between the current value and the next one that would change anything about the
                #         # background predictions; this might be more stable as the current implementation can highly
                #         # depend on the exact value of a single event (the one that tips the constraints over the edge
                #         # to fulfillment)

                #         # bin found, stop
                #         break
                #     # constraints not met, advance index to include the next tt or dy event and try again
                #     # next_bkg_indices = np.where(rec.process[next_idx:] != HH)[0]
                #     if len(next_bkg_indices) == 0:
                #         # no more background events left, move to the last position and let the stopping condition 3
                #         # below handle the rest
                #         next_idx = len(rec)
                #     else:
                #         next_idx += next_bkg_indices[0] + 1
                # else:
                #     # stopping condition 3: no more events left, so the last bin (most left one) does not fullfill
                #     # constraints; however, this should practically never happen
                #     stop_reason = "no more events left while trying to fulfill constraints"
                #     break


                # next_idx found, update values
                # get next edge or set to low edge if end is reached
                if next_idx == num_events:
                    edge_value = low_edge
                else:
                    # calculate bin center as new edge
                    edge_value = float(r_signal_x[next_idx - 1:next_idx + 1].mean())
                # prevent out of bounds values and push them to the boundaries
                bin_edges.append(max(min(edge_value, max_edge), low_edge))

                already_binned_y += get_integral(r_cumulative_y, next_idx, offset)
                offset = next_idx
                indices_gathering.append(next_idx)


            # make sure the lower dnn_output (max events) is included
            if bin_edges[-1] != low_edge:
                if len(bin_edges) > n_bins:
                    raise RuntimeError(f"number of bins reached and initial bin edge is not minimal bin edge (edges: {bin_edges})")
                bin_edges.append(low_edge)
                indices_gathering.append(num_events)


            # some debugging output
            n_bins_actual = len(bin_edges) - 1
            if n_bins_actual > n_bins:
                raise Exception("number of actual bins ended up larger than requested (implementation bug)")
            if n_bins_actual < n_bins:
                print(
                    f"  reducing n_bins from {n_bins} to {n_bins_actual} \n"
                    f"    -> reason: {stop_reason or 'NO REASON!?'}",
                )
                n_bins = n_bins_actual


            # flip indices to the right order
            breakpoint(header="indices_gathering")
            indices_gathering = (np.flip(indices_gathering) - num_events)*-1
            return np.flip(np.array(bin_edges), axis=-1), indices_gathering

            # breakpoint(header="find_new_edge")
            # bin_floor = np.floor(cumulative / threshold)
            # # compare with neibouring bins to find the best edge
            # indices = np.where(np.roll(bin_floor,1)!=bin_floor)[0]
            # #get edges of the bins
            # edges = edges[indices]
            # # replace the first index with 1 and last with 0
            # edges[-1] = low_edge
            # edges[0] = high_edge

            # # get the integral to the edge
            # integral_to_n_edge = cumulative[indices][1:]
            # integral_to_n_minus_1_edge = np.insert(integral_to_n_edge[:-1], 0, 0)
            # flat_signal_weight = integral_to_n_edge - integral_to_n_minus_1_edge

            # return np.flip(np.array(edges)), flat_signal_weight
            # while threshold < 1:
            #     ind = np.argmax(cumulative > threshold)
            #     choosen_edge = cumulative[ind]
            #     new_edges.append(choosen_edge)
            #     threshold += choosen_edge



            # for edge, cum in zip(edges,cumulative):
            #     # TODO add constraints
            #     if cum > threshold:
            #         new_edges.append(edge)
            #         threshold += cum
            # new_edges.append(low_edge)
            # return np.flip(np.array(new_edges))

        def apply_edges(h: hist.Hist, edges: np.ndarray, weights, n_bins: int = 10) -> hist.Hist:
            """
            Helper function that takes a histogram h and applies new bin edges to it.
            Rebinned histogram is returned.

            Args:
            h (hist.Hist): hist Histogram that is to be rebinned
            edges (np.ndarray): a array of ascending bin edges

            Returns:
                hist.Hist: _description_
            """
            from columnflow.columnar_util import fill_hist
            # get variables from hist
            not_variables = ("category", "shift")
            variables = [name for name in h.axes.name if name not in not_variables]

            # create new hist and add axes with coresponding edges
            edges, indices = edges

            axes = (
                [h.axes[not_variable] for not_variable in not_variables] +
                [hist.axis.Variable(edges, name=variable ,label=f"{variable}-flat-s") for variable in variables]
                )
            new_hist = hist.Hist(*axes, storage=hist.storage.Weight())

            # fill histogram, fill_data is a dict with keys being the axes names
            breakpoint(header="fill_data")
            # calculate the new values for rebinning for given edge
            new_compare = new_hist.copy()

            # slice the old histogram with new edges to sum bin contents and get the new bin content
            slices = [slice(int(indices[index]), int(indices[index + 1])) for index in range(0,len(indices) - 1)]
            slice_array = [np.sum(h.view()[..., _slice], axis=-1, keepdims=True) for _slice in slices]
            # concatenate the slices to get the new bin content and  store in new histogram storage view
            np.concatenate(slice_array, axis=-1, out=new_compare.view())



            fill_data = {}
            # copy axes for not_variables
            for axis_name in not_variables:
                axis = h.axes[axis_name]
                bin_range = range(0, axis.size)
                # fill data expects axis 1 to be existent
                bins = np.array([axis.bin(i) for i in bin_range])
                fill_data[axis_name] = np.reshape(bins,(1,-1))
            for variable in variables:
                # normalize to num of bins
                axis = h.axes[variable]
                fill_data[variable] = np.reshape(h.axes["res_dnn_hh_fine"].centers, (1,-1)) # TODO 1D data
            # fill not the construction proxy, but the actual histogram
            breakpoint(header="fill_hist")
            histogram_weight = new_hist.Weight()

            fill_hist(histogram_weight, fill_data, fill_kwargs={"weight": h.counts()})
            return histogram_weight

        import hist
        n_bins = 10
        # get the histogramπ of signal, only 1 signal process is allowed
        background_proc, background_hist = [], []
        for process, histogram in hists.items():
            if process.has_tag("signal"):
                signal_proc = process
                signal_hist = histogram
            else:
                background_proc.append(process)
                background_hist.append(histogram)
        if not signal_proc:
            logger.warning(f"could not find any signal process, return hist unchanged")
            return hists

        breakpoint(header="Start flat_s")
        # 1. preparation
        # get the leaf categories (e.g. {etau,mutau}__os__iso)
        leaf_cats = task.config_inst.get_category(task.branch_data.category).get_leaf_categories()

        # get category ids to filter and combine leaf categories
        hist_loc_cat_ids = [hist.loc(category.id) for category in leaf_cats]
        combined_signal_hist = signal_hist[{"category": hist_loc_cat_ids}]
        combined_signal_hist = combined_signal_hist[{"category": sum}]

        # remove shift axis, since its always nominal
        combined_signal_hist = combined_signal_hist[{"shift": hist.loc(0)}]

        # fine binned histograms bin centers are approx equivalent to dnn output
        variables_axis = combined_signal_hist.axes[task.variables[0]]
        signal_x = variables_axis.centers
        # weight in a weighted histogram is equal to the number of events
        signal_y = combined_signal_hist.counts()

        # signal_yield are sorted by default (since the histogram axis is sorted)
        # background yields and weights


        # 2. determine bin edges
        new_edges = find_edges(
            signal_x = signal_x,
            signal_y = signal_y,
            background_x = None, # TODO add background yields
            background_y = None,
            n_bins=n_bins,
            )

        # 3. apply to hists

        rebinned_hist = apply_edges(signal_hist, new_edges, weights=signal_y, n_bins=n_bins)
        hists[signal_proc] = rebinned_hist
        return hists
        # done




    # # insert counts and weights into columns for correct processes
    #         # (faster than creating arrays above which then get copied anyway when the recarray is created)
    #         HH, TT, DY = range(3)
    #         rec.hh_count_cs[rec.process == HH] = 1
    #         rec.tt_count_cs[rec.process == TT] = 1
    #         rec.dy_count_cs[rec.process == DY] = 1
    #         rec.hh_weight_cs[rec.process == HH] = hh_weights
    #         rec.tt_weight_cs[rec.process == TT] = tt_weights
    #         rec.dy_weight_cs[rec.process == DY] = dy_weights
    #         # sort by decreasing value to start binning from "the right" later on
    #         rec.sort(order="value")
    #         rec = np.flip(rec, axis=0)
    #         # replace counts and weights with their cumulative sums
    #         rec.hh_count_cs[:] = np.cumsum(rec.hh_count_cs)
    #         rec.tt_count_cs[:] = np.cumsum(rec.tt_count_cs)
    #         rec.dy_count_cs[:] = np.cumsum(rec.dy_count_cs)
    #         rec.hh_weight_cs[:] = np.cumsum(rec.hh_weight_cs)
    #         rec.tt_weight_cs[:] = np.cumsum(rec.tt_weight_cs)
    #         rec.dy_weight_cs[:] = np.cumsum(rec.dy_weight_cs)
    #         # eager cleanup
    #         del all_values, izeros, fzeros
    #         del hh_values, hh_weights
    #         del tt_values, tt_weights
    #         del dy_values, dy_weights
    #         # now, between any two possible discriminator values, we can easily extract the hh, tt and dy integrals,
    #         # as well as raw event counts without the need for additional, costly accumulation ops (sum, count, etc.),
    #         # but rather through simple subtraction of values at the respective indices instead

    #         #
    #         # step 2: binning
    #         #

    #         # determine the approximate hh yield per bin
    #         hh_yield_per_bin = rec.hh_weight_cs[-1] / n_bins
    #         # keep track of bin edges and the hh yield accumulated so far
    #         bin_edges = [x_max]
    #         hh_yield_binned = 0.0
    #         min_hh_yield = 1.0e-5
    #         # during binning, do not remove leading entries, but remember the index that denotes the start of the bin
    #         offset = 0
    #         # helper to extract a cumulative sum between the start offset (included) and the stop index (not included)
    #         get_integral = lambda cs, stop: cs[stop - 1] - (0 if offset == 0 else cs[offset - 1])
    #         # bookkeep reasons for stopping binning
    #         stop_reason = ""
    #         # start binning
    #         while len(bin_edges) < n_bins:
    #             # stopping condition 1: reached end of events
    #             if offset >= len(rec):
    #                 stop_reason = "no more events left"
    #                 break
    #             # stopping condition 2: remaining hh yield too small, so cause a background bin to be created
    #             remaining_hh_yield = rec.hh_weight_cs[-1] - hh_yield_binned
    #             if remaining_hh_yield < min_hh_yield:
    #                 stop_reason = "remaining signal yield insufficient"
    #                 break
    #             # find the index of the event that would result in a hh yield increase of more than the expected
    #             # per-bin yield; this index would mark the start of the next bin given all constraints are met
    #             if remaining_hh_yield >= hh_yield_per_bin:
    #                 threshold = hh_yield_binned + hh_yield_per_bin
    #                 next_idx = offset + np.where(rec.hh_weight_cs[offset:] > threshold)[0][0]
    #             else:
    #                 # special case: remaining hh yield smaller than the expected per-bin yield, so find the last event
    #                 next_idx = offset + np.where(rec.process[offset:] == HH)[0][-1] + 1
    #             # advance the index until backgrounds constraints are met
    #             while next_idx < len(rec):
    #                 # get the number of tt events and their yield
    #                 n_tt = get_integral(rec.tt_count_cs, next_idx)
    #                 y_tt = get_integral(rec.tt_weight_cs, next_idx)
    #                 # get the number of dy events and their yield
    #                 n_dy = get_integral(rec.dy_count_cs, next_idx)
    #                 y_dy = get_integral(rec.dy_weight_cs, next_idx)
    #                 # evaluate constraints
    #                 # TODO: potentially relax constraints here, e.g when there are 3 (4?) tt events, drop the constraint
    #                 #       on dy, and vice-versa
    #                 constraints_met = (
    #                     # tt and dy events
    #                     n_tt >= 1 and
    #                     n_dy >= 1 and
    #                     n_tt + n_dy >= 4 and
    #                     # yields must be positive to avoid negative sums of weights per process
    #                     y_tt > 0 and
    #                     y_dy > 0
    #                 )
    #                 if constraints_met:
    #                     # TODO: maybe also check if the background conditions are just barely met and advance next_idx
    #                     # to the middle between the current value and the next one that would change anything about the
    #                     # background predictions; this might be more stable as the current implementation can highly
    #                     # depend on the exact value of a single event (the one that tips the constraints over the edge
    #                     # to fulfillment)

    #                     # bin found, stop
    #                     break
    #                 # constraints not met, advance index to include the next tt or dy event and try again
    #                 next_bkg_indices = np.where(rec.process[next_idx:] != HH)[0]
    #                 if len(next_bkg_indices) == 0:
    #                     # no more background events left, move to the last position and let the stopping condition 3
    #                     # below handle the rest
    #                     next_idx = len(rec)
    #                 else:
    #                     next_idx += next_bkg_indices[0] + 1
    #             else:
    #                 # stopping condition 3: no more events left, so the last bin (most left one) does not fullfill
    #                 # constraints; however, this should practically never happen
    #                 stop_reason = "no more events left while trying to fulfill constraints"
    #                 break
    #             # next_idx found, update values
    #             edge_value = x_min if next_idx == 0 else float(rec.value[next_idx - 1:next_idx + 1].mean())
    #             bin_edges.append(max(min(edge_value, x_max), x_min))
    #             hh_yield_binned += get_integral(rec.hh_weight_cs, next_idx)
    #             offset = next_idx

    #         # make sure the minimum is included
    #         if bin_edges[-1] != x_min:
    #             if len(bin_edges) > n_bins:
    #                 raise RuntimeError(f"number of bins reached and initial bin edge is not x_min (edges: {bin_edges})")
    #             bin_edges.append(x_min)

    #         # reverse edges and optionally re-set n_bins
    #         bin_edges = sorted(set(bin_edges))
    #         n_bins_actual = len(bin_edges) - 1
    #         if n_bins_actual > n_bins:
    #             raise Exception("number of actual bins ended up larger than requested (implementation bug)")
    #         if n_bins_actual < n_bins:
    #             print(
    #                 f"  reducing n_bins from {n_bins} to {n_bins_actual} in ({category},{spin},{mass})\n"
    #                 f"    -> reason: {stop_reason or 'NO REASON!?'}",
    #             )
    #             n_bins = n_bins_actual

        return hists

>>>>>>> Stashed changes
    config.x.hist_hooks = {
        "qcd": qcd_estimation,
    }
