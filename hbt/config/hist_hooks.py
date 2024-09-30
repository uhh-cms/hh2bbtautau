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
        data_hist = sum(data_hists[19090:], data_hists[0].copy())

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


    def flat_s(task ,hists: Dict[hist.Histogram]) -> Dict[hist.Histogram]:
        """Rebinnig of the histograms in *hists* to archieve a flat-signal distribution.

        Args:
            task (TODO): task instance that contains the process informations
            hists (Dict[hist.Histogram]): A dictionary of histograms using Process instances as keys

        Returns:
            Dict[hist.Histogram]: A dictionary of histograms using Process instances as keys
        """
        def find_edges(signal_histogram, background_histograms, variable, n_bins) -> Tuple[np.ndarray, np.ndarray]:
            """
            Determine new bin edges that result in a flat signal distribution.
            The edges are determined by the signal distribution, while the background distribution
            is used to ensure that the background yield in each bin is sufficient.
            """
            def get_integral(cumulative_weights, stop, offset=0):
                """
                Helper to calculate the integral of *cumulative_weights* between the *offset* (included)
                and the *stop* index (not included)
                """
                return cumulative_weights[stop - 1] - (0 if offset == 0 else cumulative_weights[offset - 1])

            def prepare_background(histogram: hist.Histogram) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # noqa
                """
                Helper to extract information from background histograms.

                Returns:
                    tuple[np.ndarray]: A tuple containing the array that describe bin yield,
                    the number of equivalent bins and the cumulative bin yield.
                """
                bin_yield = histogram.counts()
                # y^2 / sigma^2, where y is the yield, sigma is the uncertainty
                # these are the number of events with weight 1 and same statistical fluctuation
                number_of_equivalent_bins = bin_yield**2 / histogram.variances()
                bin_yield = np.flip(bin_yield, axis=-1),
                cumulative_bin_yield = np.cumsum(bin_yield, axis=0)
                return (
                    bin_yield,
                    np.flip(number_of_equivalent_bins, axis=-1),
                    cumulative_bin_yield
                    )


            # signal_yield are sorted by default (since the histogram axis is sorted)
            # fine binned histograms bin centers are approx equivalent to dnn output
            # weight in a weighted histogram is equal to the number of events
            variables_axis = signal_histogram.axes[variable]
            signal_x = variables_axis.centers
            signal_y = combined_signal_hist.counts()

            # flip arrays to start from the right
            r_signal_x, r_signal_y = np.flip(signal_x, axis=-1), np.flip(signal_y, axis=-1)
            # calculate cumulative of reversed signal yield
            r_cumulative_signal_y = np.cumsum(r_signal_y, axis=0)

            # prepare parameters
            low_edge, max_edge = 0, 1
            bin_edges = [max_edge, ]
            num_events = len(r_cumulative_signal_y)
            n_bins = 10
            # calculate desired yield per bin, last bin of cum is the total yield
            full_cum = r_cumulative_signal_y[-1]
            y_per_bin = full_cum / n_bins
            # accumulated signal yield up to the current index
            already_binned_y = 0.0
            min_y = 1.0e-5
            # during binning, do not remove leading entries, but remember the index that denotes the start of the bin
            offset = 0
            # bookkeep reasons for stopping binning
            stop_reason = ""
            # accumulated signal yield up to the current index
            y_already_binned = 0.0
            y_min = 1.0e-5
            # during binning, do not remove leading entries
            # instead remember the index that denotes the start of the bin
            offset = 0

            # prepare signal
            # fine binned histograms bin centers are approx equivalent to dnn output
            # flip arrays to start from the right
            dnn_score_signal = np.flip(signal_histogram.axes[variable].centers, axis=-1)
            y_signal = np.flip(signal_histogram.counts(), axis=-1)

            # calculate cumulative of reversed signal yield and yield per bin
            cumulu_y_signal = np.cumsum(y_signal, axis=0)
            full_cum = cumulu_y_signal[-1]
            y_per_bin = full_cum / n_bins
            num_events = len(cumulu_y_signal)

            # prepare background

            for process, histogram in background_histograms.items():
                if process.name == "tt":
                    tt_y, tt_num_eq, cumulu_tt_y = prepare_background(histogram)
                elif process.name == "dy":
                    dy_y, dy_num_eq, cumulu_dy_y = prepare_background(histogram)

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
                    next_idx = offset + np.where(cumulu_y_signal[offset:] > threshold)[0][0]
                else:
                    # special case: remaining signal yield smaller than the expected per-bin yield,
                    # so find the last event
                    next_idx = offset + np.where(cumulu_y_signal[offset:])[0][-1] + 1

                # advance the index until backgrounds constraints are met
                #breakpoint(header="find_new_edge - background")

                # combine tt and dy histograms

                # Background constraints
                while next_idx < num_events:
                    # get the number of monte carlo tt and dy events
                    tt_num_events = get_integral(r_tt_num, next_idx, offset)
                    dy_num_events = get_integral(r_dy_num, next_idx, offset)

                    tt_yield = get_integral(cumulu_tt_y, next_idx, offset)
                    dy_yield = get_integral(cumulu_dy_y, next_idx, offset)


                #     # evaluate constraints
                    # TODO: potentially relax constraints here, e.g when there are 3 (4?) tt events, drop the constraint
                    #       on dy, and vice-versa
                    constraints_met = (
                        # have atleast 1 tt, 1 dy and atleast 4 background events
                        # scale by lumi ratio to be more fair to the smaller dataset
                        num_tt_events >= 1 * lumi_ratio and
                        num_dy_events >= 1 * lumi_ratio and
                        num_tt_events + num_dy_events >= 4 * lumi_ratio and

                        # yields must be positive to avoid negative sums of weights per process
                        tt_yield > 0 and
                        dy_yield > 0
                    )
                    if constraints_met:
                        # TODO: maybe also check if the background conditions are just barely met and advance next_idx
                        # to the middle between the current value and the next one that would change anything about the
                        # background predictions; this might be more stable as the current implementation can highly
                        # depend on the exact value of a single event (the one that tips the constraints over the edge
                        # to fulfillment)
                        print("constraint met")
                        # bin found, stop
                        break

                    # constraints not met, advance index to include the next tt or dy event and try again
                    next_idx += 1
                else:
                    # stopping condition 3: no more events left, so the last bin (most left one) does not fullfill
                    # constraints; however, this should practically never happen
                    stop_reason = "no more events left while trying to fulfill constraints"
                    break


                # next_idx found, update values
                # get next edge or set to low edge if end is reached
                if next_idx == num_events:
                    edge_value = low_edge
                else:
                    # calculate bin center as new edge
                    signal_edge_value = float(dnn_score[next_idx - 1:next_idx + 1].mean())
                # prevent out of bounds values and push them to the boundaries
                bin_edges.append(max(min(edge_value, max_edge), low_edge))

                y_already_binned += get_integral(cumulu_y_signal, next_idx, offset)
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
            indices_gathering = (np.flip(indices_gathering) - num_events)*-1
            return np.flip(np.array(bin_edges), axis=-1), indices_gathering


        def apply_edges(h: hist.Hist, edges: np.ndarray, indices: np.ndarray, variable: Tuple[str]) -> hist.Hist:
            """
            Rebin the content axes determined by *variables* of a given hist histogram *h* to
            given *edges* and their *indices*.
            The rebinned histogram is returned.

            Args:
            h (hist.Hist): hist Histogram that is to be rebinned
            edges (np.ndarray): a array of ascending bin edges
            indices (np.ndarray): a array of indices that define the new bin edges
            variables (str): variable name that is rebinned

            Returns:
                hist.Hist: rebinned hist histogram
            """
            # sort edges and indices, by default they are sorted
            ascending_order = np.argsort(edges)
            edges, indices = edges[ascending_order], indices[ascending_order]

            # create new hist and add axes with coresponding edges
            # define new axes, from old histogram and rebinned variable with new axis
            axes = (
                [h.axes[axis] for axis in h.axes.name if axis not in variable] +
                [hist.axis.Variable(edges, name=variable ,label=f"{variable}-flat-s")]
                )

            new_hist = hist.Hist(*axes, storage=hist.storage.Weight())

            # slice the old histogram storage view with new edges
            # sum over sliced bin contents to get rebinned content
            slices = [slice(int(indices[index]), int(indices[index + 1])) for index in range(0, len(indices) - 1)]
            slice_array = [np.sum(h.view()[..., _slice], axis=-1, keepdims=True) for _slice in slices]
            # concatenate the slices to get the new bin content
            # store in new histogram storage view
            np.concatenate(slice_array, axis=-1, out=new_hist.view())

            return new_hist

        import hist
        n_bins = 10
        # find signal histogram for which you will optimize, only 1 signal process is allowed
        background_hists = {}
        for process, histogram in hists.items():
            if process.has_tag("signal"):
                signal_proc = process
                signal_hist = histogram
            else:
                background_hists[process] = histogram

        if not signal_proc:
            logger.warning(f"could not find any signal process, return hist unchanged")
            return hists

        # 1. preparation
        # get the leaf categories (e.g. {etau,mutau}__os__iso)
        leaf_cats = task.config_inst.get_category(task.branch_data.category).get_leaf_categories()

        # sum over different leaf categories
        cat_ids_locations = [hist.loc(category.id) for category in leaf_cats]
        combined_signal_hist = signal_hist[{"category": cat_ids_locations}]
        combined_signal_hist = combined_signal_hist[{"category": sum}]
        # remove shift axis, since its always nominal
        combined_signal_hist = combined_signal_hist[{"shift": hist.loc(0)}]

        # same for background
        for process, histogram in background_hists.items():
            combined_background_hist = histogram[{"category": cat_ids_locations}]
            combined_background_hist = combined_background_hist[{"category": sum}]
            combined_background_hist = combined_background_hist[{"shift": hist.loc(0)}]
            background_hists[process] = combined_background_hist

        # 2. determine bin edges
        flat_s_edges, flat_s_indices = find_edges(
            signal_histogram=combined_signal_hist,
            variable=task.variables[0],
            background_histograms=background_hists,
            n_bins=n_bins,

            )

        # 3. apply to hists
        for process, histogram in hists.items():
            hists[process] = apply_edges(
                histogram,
                flat_s_edges,
                flat_s_indices,
                task.variables[0]
            )

        return hists

    config.x.hist_hooks = {
        "qcd": qcd_estimation, "flat_s": flat_s
    }
