# coding: utf-8

"""
Histogram hooks.
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial

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

    def flat_s(task, hists: dict[od.Process, hist.Histogram]) -> dict[od.Process, hist.Histogram]:
        """Rebinnig of the histograms in *hists* to archieve a flat-signal distribution.

        :param task: task instance that contains the process informations
        :param hists: A dictionary of histograms using Process instances as keys

        :raises RuntimeError: If the wanted number of bins is reached and the initial
        bin edge is not minimal.
        :raises Exception: If the number of actual bins ended up larger than requested.
        :return: A dictionary of histograms using Process instances as keys
        """
        def find_edges(signal_histogram, background_histograms, variable, n_bins=10) -> tuple[np.ndarray, np.ndarray]:
            """
            Determine new bin edges that result in a flat signal distribution.
            The edges are determined by the signal distribution, while the background distribution
            is used to ensure that the background yield in each bin is sufficient.

            :param signal_histogram: The histogram that describes the signal distribution.
            :param background_histograms: A dictionary of histograms using the process as key
            that describe the background distribution.
            :param variable: The variable name that is rebinned.
            :param n_bins: The number of bins that the signal distribution should be rebinned to.

            :return: A tuple containing the new bin edges and the indices that define the new bin edges.
            """
            def get_integral(cumulative_weights, stop, offset=0):
                """
                Helper to calculate the integral of *cumulative_weights* between the *offset* (included)
                and the *stop* index (not included)

                :param cumulative_weights: The cumulative weights array.
                :param stop: The index up to which the integral is calculated.
                :param offset: The index from which the integral is calculated.

                :return The integral of the weights between the given indices.
                """
                return cumulative_weights[stop - 1] - (0 if offset == 0 else cumulative_weights[offset - 1])

            def prepare_background(histogram: hist.Histogram) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # noqa
                """
                Helper to extract information from background histograms.

                :param histogram: A histogram that describes the background distribution.

                :return: A tuple containing the array that describe bin yield,
                    the number of equivalent bins and the cumulative bin yield.
                """
                bin_yield = histogram.counts()
                # y^2 / sigma^2, where y is the yield, sigma is the uncertainty
                # these are the number of events with weight 1 and same statistical fluctuation
                number_of_equivalent_bins = bin_yield**2 / histogram.variances()
                bin_yield = np.flip(bin_yield, axis=-1)
                cumulative_bin_yield = np.cumsum(bin_yield, axis=0)
                return (
                    bin_yield,
                    np.flip(number_of_equivalent_bins, axis=-1),
                    cumulative_bin_yield,
                )
            # prepare parameters
            low_edge, max_edge = 0, 1
            bin_edges = [max_edge]
            indices_gathering = [0]

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
                if process.name == "ttbar":
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
                y_remaining = full_cum - y_already_binned
                if y_remaining < y_min:
                    stop_reason = "remaining signal yield insufficient"
                    break
                # find the index of the event that would result in a signal yield increase of more
                # than the expected per-bin yield;
                # this index would mark the start of the next bin given all constraints are met
                if y_remaining >= y_per_bin:
                    threshold = y_already_binned + y_per_bin
                    # get indices of array of values above threshold
                    # first entry defines the next bin edge
                    # shift next idx by offset
                    next_idx = offset + np.where(cumulu_y_signal[offset:] > threshold)[0][0]
                else:
                    # special case: remaining signal yield smaller than the expected per-bin yield,
                    # so find the last event
                    next_idx = offset + np.where(cumulu_y_signal[offset:])[0][-1] + 1

                # Background constraints
                while next_idx < num_events:
                    # get the number of monte carlo tt and dy events
                    tt_num_events = get_integral(tt_num_eq, next_idx, offset)
                    dy_num_events = get_integral(tt_num_eq, next_idx, offset)
                    tt_yield = get_integral(cumulu_tt_y, next_idx, offset)
                    dy_yield = get_integral(cumulu_dy_y, next_idx, offset)

                    # evaluate constraints
                    # TODO: potentially relax constraints here, e.g when there are 3 (4?) tt events, drop the constraint
                    #       on dy, and vice-versa
                    constraints_met = (
                        # have atleast 1 tt, 1 dy and atleast 4 background events
                        # scale by lumi ratio to be more fair to the smaller dataset
                        tt_num_events >= 1 and
                        dy_num_events >= 1 and
                        tt_num_events + dy_num_events >= 4 and

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
                    edge_value = float(dnn_score_signal[next_idx - 1:next_idx + 1].mean())
                # prevent out of bounds values and push them to the boundaries
                bin_edges.append(max(min(edge_value, max_edge), low_edge))

                y_already_binned += get_integral(cumulu_y_signal, next_idx, offset)
                offset = next_idx
                indices_gathering.append(next_idx)

            # make sure the lower dnn_output (max events) is included
            if bin_edges[-1] != low_edge:
                if len(bin_edges) > n_bins:
                    raise RuntimeError(
                        "number of bins reached and initial bin edge"
                        f" is not minimal bin edge (edges: {bin_edges})",
                    )
                bin_edges.append(low_edge)
                indices_gathering.append(num_events)

            # some debugging output
            n_bins_actual = len(bin_edges) - 1
            if n_bins_actual > n_bins:
                raise Exception("number of actual bins ended up larger than requested (implementation bug)")
            if n_bins_actual < n_bins:
                print(
                    f"  started from {num_events} bins, targeted {n_bins} but ended at {n_bins_actual} bins\n"
                    f"    -> reason: {stop_reason or 'NO REASON!?'}",
                )
                n_bins = n_bins_actual

            # flip indices to the right order
            indices_gathering = (np.flip(indices_gathering) - num_events) * -1
            return np.flip(np.array(bin_edges), axis=-1), indices_gathering

        def apply_edges(h: hist.Hist, edges: np.ndarray, indices: np.ndarray, variable: str) -> hist.Hist:
            """
            Rebin the content axes determined by *variable* of a given hist histogram *h* to
            given *edges* and their *indices*.
            The rebinned histogram is returned.

            :param h: hist Histogram that is to be rebinned
            :param edges: a array of ascending bin edges
            :param indices: a array of indices that define the new bin edges
            :param variable: variable name that is rebinned

            :return: rebinned hist histogram
            """
            # sort edges and indices, by default they are sorted
            ascending_order = np.argsort(edges)
            edges, indices = edges[ascending_order], indices[ascending_order]

            # create new hist and add axes with coresponding edges
            # define new axes, from old histogram and rebinned variable with new axis
            axes = (
                [h.axes[axis] for axis in h.axes.name if axis not in variable] +
                [hist.axis.Variable(edges, name=variable, label=f"{variable}-flat-s")]
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
            logger.warning("could not find any signal process, return hist unchanged")
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
                task.variables[0],
            )

        return hists

    def general_higgs_morphing(
        task,
        hists,
        production_channel="ggf",
        guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1"],
        target_point="kl5_kt1",
        morphing_type="average",
        weights=False,
    ):
        """
        General morphing of Higgs production processes.

        params:
            task: law task
            hists: dict of histograms
            production_channel: str, "vbf" or "ggf", production channel of the Higgs boson
            guidance_points: list of str, guidance points to be used for morphing
            target_point: str, target point to be morphed to
            morphing_type: str, "exact", "average" or "fit", type of morphing to be used
        """

        # check conditions for morphing
        if morphing_type not in ["exact", "average", "fit"]:
            raise ValueError(f"morphing type {morphing_type} not recognized")

        if production_channel not in ["ggf", "vbf"]:
            raise ValueError(f"production channel {production_channel} not recognized")

        if morphing_type == "exact":
            if production_channel == "ggf":
                if len(guidance_points) != 3:
                    raise ValueError("exact morphing requires exactly three guidance points")
            if production_channel == "vbf":
                if len(guidance_points) != 6:
                    raise ValueError("exact morphing requires exactly six guidance points")

        if morphing_type == "average" or morphing_type == "fit":
            if production_channel == "ggf":
                if len(guidance_points) < 4:
                    raise ValueError(f"{morphing_type} morphing requires at least four guidance points")
            if production_channel == "vbf":
                if len(guidance_points) < 7:
                    raise ValueError(f"{morphing_type} morphing requires at least seven guidance points")

        # define helper functions
        def general_read_coupling_values(couplings, poi=["kl", "kt", "c2"]):
            coupling_list = couplings.split("_")
            coupling_dict = {}
            for coupling in coupling_list:
                for param in poi:
                    if coupling.startswith(param):
                        if param == "k2v":
                            coupling_dict[param] = coupling[3:].replace("p", ".").replace("m", "-")
                        else:
                            coupling_dict[param] = coupling[2:].replace("p", ".").replace("m", "-")
            # if "c2" in poi:
            #     if "c2" not in coupling_dict.keys():
            #         coupling_dict["c2"] = "0"
            if "kl" in poi:
                if "kl" not in coupling_dict.keys():
                    raise ValueError("kl not found in couplings")
            if "kt" in poi:
                if "kt" not in coupling_dict.keys():
                    raise ValueError("kt not found in couplings")
            if "kv" in poi:
                if "kv" not in coupling_dict.keys():
                    raise ValueError("kv not found in couplings")
            if "k2v" in poi:
                if "k2v" not in coupling_dict.keys():
                    raise ValueError("k2v not found in couplings")
            if "kl" in poi:
                if "kl" not in coupling_dict.keys():
                    raise ValueError("kl not found in couplings")
            return coupling_dict

        def get_coeffs(guidance_point_dict, parameters_oi):
            """
            Get coefficients from physical parameters for the morphing of the Higgs production processes.
            ggf assumes kl, kt, c2 as ordering in the parameters_oi list
            vbf assumes kv, k2v, kl as ordering in the parameters_oi list
            """
            if parameters_oi == ["kl", "kt", "c2"]:
                # (formula for any kt,kl: (kt**2*kl**2,kt**4,kt**3*kl))
                # assuming parameters_oi = ["kl", "kt", "c2"], c2 is not used in the morphing for now
                coeffs = [
                    (guidance_point_dict[parameters_oi[0]]**2) * (guidance_point_dict[parameters_oi[1]]**2),
                    guidance_point_dict[parameters_oi[1]]**4,
                    guidance_point_dict[parameters_oi[0]] * guidance_point_dict[parameters_oi[1]]**3,
                ]
            elif parameters_oi == ["kv", "k2v", "kl"]:
                # (formula for any kv,k2v,kl: (kv**4,k2v**2,kv**2*kl**2,kv**2*k2v,kv**3*kl,k2v*kv*kl)
                # assuming parameters_oi = ["kv", "k2v", "kl"]
                coeffs = [
                    guidance_point_dict[parameters_oi[0]]**4,
                    guidance_point_dict[parameters_oi[1]]**2,
                    (guidance_point_dict[parameters_oi[0]]**2) * (guidance_point_dict[parameters_oi[2]]**2),
                    (guidance_point_dict[parameters_oi[0]]**2) * guidance_point_dict[parameters_oi[1]],
                    (guidance_point_dict[parameters_oi[0]]**3) * guidance_point_dict[parameters_oi[2]],
                    guidance_point_dict[parameters_oi[0]] * guidance_point_dict[parameters_oi[1]] * guidance_point_dict[parameters_oi[2]],  # noqa
                ]
            else:
                raise ValueError("parameters of interest not recognized, expected [kl, kt, c2] or [kv, k2v, kl]")
            return coeffs

        # define parameters of interest for the production channel, following the order used in get_coeffs
        if production_channel == "ggf":
            parameters_oi = ["kl", "kt", "c2"]  # c2 is not used in the morphing for now
        if production_channel == "vbf":
            parameters_oi = ["kv", "k2v", "kl"]

        # get all guidance points
        guidance_points_list = [general_read_coupling_values(coupling, parameters_oi) for coupling in guidance_points]
        # for coupling in guidance_points:
        #     guidance_points_dict.update(general_read_coupling_values(coupling, parameters_oi))
        guidance_points_list_float = [
            {key: float(sample[key]) for key in sample.keys()} for sample in guidance_points_list
        ]
        # guidance_points_dict_float = {
        #     key: [float(coupling_value) for coupling_value in guidance_points_dict[key]]
        #     for key in guidance_points_dict.keys()
        # }

        # get all model processes
        model_procs = []
        for sample in guidance_points:
            model_proc = config.get_process(f"hh_{production_channel}_hbb_htt_{sample}", default=None)
            model_procs.append(model_proc)
        if not all(model_procs):
            return hists

        # verify that the axis order is exactly "category -> shift -> variable"
        # which is needed to insert values at the end
        CAT_AXIS, SHIFT_AXIS, VAR_AXIS = range(3)
        for h in hists.values():
            # validate axes
            assert len(h.axes) == 3
            assert h.axes[CAT_AXIS].name == "category"
            assert h.axes[SHIFT_AXIS].name == "shift"

        # get model histograms
        model_hists = [h for p, h in hists.items() if p in model_procs]
        if len(model_hists) != len(guidance_points):
            raise Exception("not all model histograms present, morphing cannot occur")

        # new coefficients for the newly created point
        target_params_oi = general_read_coupling_values(target_point, parameters_oi)
        target_params_oi_float = {key: float(target_params_oi[key]) for key in target_params_oi.keys()}
        new_coefficients = np.array(get_coeffs(target_params_oi_float, parameters_oi))

        if production_channel == "ggf":
            # create the new process for the morphed histogram, TODO: make customized id?
            new_proc = od.Process(
                f"hh_{production_channel}_hbb_htt_{target_point}_{morphing_type}_morphed",
                id=21130,
                label=r"morphed $HH_{ggf} \rightarrow bb\tau\tau$ "
                "\n"
                r"($\kappa_{\lambda}=$" + target_params_oi["kl"] + r", $\kappa_{t}=$" + target_params_oi["kt"] + ")",
            )
        elif production_channel == "vbf":
            # create the new process for the morphed histogram, TODO: make customized id?
            new_proc = od.Process(
                f"hh_{production_channel}_hbb_htt_{target_point}_{morphing_type}_morphed",
                id=21130,
                label=r"morphed $HH_{vbf} \rightarrow bb\tau\tau$ "
                "\n"
                r"($C_{V}=$" + target_params_oi["kv"] + r", $C_{2V}=$" + target_params_oi["k2v"] +
                r", $C_{3}=$" + target_params_oi["kl"] + ")",
            )

        # create the new hist
        new_hist = model_hists[0].copy().reset()

        # prepare morphing here
        # be careful with the order, the categories can be shuffled around in the different histograms
        # so sort the values by the categories
        model_values = np.array([
            model_hists[i].view().value[np.argsort(model_hists[i].axes[0])] for i in range(len(guidance_points))
        ])

        model_variances = np.array([
            model_hists[i].view().variance[np.argsort(model_hists[i].axes[0])] for i in range(len(guidance_points))
        ])

        original_hist_shape = new_hist.view().value.shape

        if morphing_type == "exact" or morphing_type == "average":
            # build guidance matrix from guidance points
            guidance_matrix = np.array([
                get_coeffs(guidance_points_list_float[i], parameters_oi)
                for i in range(len(guidance_points))
            ])

            if morphing_type == "exact":
                # inverse guidance matrix
                inv_guidance_matrix = np.linalg.inv(guidance_matrix)
            elif morphing_type == "average":
                # pseudo inverse guidance matrix
                inv_guidance_matrix = np.linalg.pinv(guidance_matrix)

            # morphing
            # morphed values
            morphed_values = np.matmul(
                np.matmul(new_coefficients, inv_guidance_matrix),
                model_values.reshape(len(guidance_points), -1),
            ).reshape(original_hist_shape)

            # morphed values and variances, using covariance matrix
            reshaped_model_variances = model_variances.reshape(len(guidance_points), -1)
            covariances = np.array([
                np.diag(reshaped_model_variances[:, i]) for i in range(reshaped_model_variances.shape[1])
            ])
            sigma_v_vector = np.matmul(
                np.matmul(
                    inv_guidance_matrix,
                    covariances.T,
                ).T,
                inv_guidance_matrix.T,
            )

            morphed_variances = np.matmul(
                new_coefficients,
                np.matmul(
                    sigma_v_vector,
                    new_coefficients,
                ).T,
            ).reshape(original_hist_shape)

            # # alternative method for morphing, using the actual equation, even for non invertible matrices:
            # # taken from blobel statistics book
            # # leads to exact same results as above
            # # (C^T * C) v = C^T sigma
            # # v = (C^T * C)^-1 * C^T sigma
            # inv_guidance_matrix_v2 = np.linalg.inv(np.matmul(guidance_matrix.T, guidance_matrix))
            # factor_v = np.matmul(inv_guidance_matrix_v2, guidance_matrix.T)
            # morphed_values_v2 = np.matmul(
            #     new_coefficients,
            #     np.matmul(
            #         factor_v,
            #         model_values.reshape(len(guidance_points), -1),
            #     ),
            # ).reshape(original_hist_shape)

            # sigma_v_vector_v2 = np.matmul(
            #     np.matmul(
            #         factor_v,
            #         covariances.T,
            #     ).T,
            #     factor_v.T,
            # )

            # morphed_variances_v2 = np.matmul(
            #     new_coefficients,
            #     np.matmul(
            #         sigma_v_vector_v2,
            #         new_coefficients,
            #     ).T,
            # ).reshape(original_hist_shape)

            if weights:
                # morphing with weights matrix, needs equation from blobel statistics book
                # actual equation also possible with weights
                # v = (C^T * W * C)^-1 * C^T * W * sigma

                # define weights
                weights = np.array([
                    np.diag(1 / reshaped_model_variances[:, i]) for i in range(reshaped_model_variances.shape[1])
                ])

                inv_guidance_matrix_v3 = np.linalg.inv(
                    np.matmul(
                        np.matmul(guidance_matrix.T, weights),
                        guidance_matrix,
                    ),
                )

                factor_v_2 = np.matmul(
                    np.matmul(inv_guidance_matrix_v3, guidance_matrix.T),
                    weights,
                )

                # at this point, matrix multiplication becomes difficult due to extended dimensionality
                # we want to multiply the matrices for each bin separately

                # # method number 1: custom multiplication by hand in loop
                # vector_v_3 = np.array([
                #     np.matmul(factor_v_2[i], model_values.reshape(len(guidance_points), -1)[:, i]) for i in range(len(factor_v_2))  # noqa
                # ])

                # morphed_values_with_weights = np.matmul(
                #     new_coefficients,
                #     vector_v_3.T,
                # ).reshape(original_hist_shape)

                # method number 2: tensor multiplication, then masking the additional dimensions
                tensordot_res = np.tensordot(
                    factor_v_2,
                    model_values.reshape(len(guidance_points), -1),
                    axes=1,
                )
                # mask creation
                mask = np.zeros_like(tensordot_res, dtype=bool)
                for i in range(len(factor_v_2)):
                    mask[i, :, i] = True

                vector_v_3_v2 = tensordot_res[mask].reshape(len(factor_v_2), -1)

                morphed_values_with_weights = np.matmul(
                    new_coefficients,
                    vector_v_3_v2.T,
                ).reshape(original_hist_shape)

                # calculate the morphed variances with weights

                # start with the first multiplication of the covariances with the prefactor of v
                tensordot_res_variances_part1 = np.tensordot(
                    factor_v_2,
                    covariances.T,
                    axes=1,
                )

                mask_variances_part1 = np.zeros_like(tensordot_res_variances_part1, dtype=bool)
                for i in range(len(factor_v_2)):
                    mask_variances_part1[i, :, :, i] = True

                sigma_vector_v_3_part1 = tensordot_res_variances_part1[mask_variances_part1].reshape(
                    len(factor_v_2),
                    len(new_coefficients),
                    -1,
                )

                # Then do the second multiplication of the covariances with the prefactor of v

                tensordot_res_variances_part2 = np.tensordot(
                    sigma_vector_v_3_part1,
                    factor_v_2.T,
                    axes=1,
                )

                mask_variances_part2 = np.zeros_like(tensordot_res_variances_part2, dtype=bool)
                for i in range(len(factor_v_2)):
                    mask_variances_part2[i, :, :, i] = True

                sigma_v_vector_v3 = tensordot_res_variances_part2[mask_variances_part2].reshape(
                    len(factor_v_2),
                    len(new_coefficients),
                    len(new_coefficients),
                )

                # finally, do the last multiplication with the new coefficients to get the morphed variances
                morphed_variances_with_weights = np.matmul(
                    new_coefficients,
                    np.matmul(
                        sigma_v_vector_v3,
                        new_coefficients,
                    ).T,
                ).reshape(original_hist_shape)

                morphed_values = morphed_values_with_weights
                morphed_variances = morphed_variances_with_weights

        if morphing_type == "fit":
            # create function and define params
            from scipy.optimize import curve_fit
            x = []
            for i in range(len(guidance_points)):
                x.append([float(guidance_points_list[i][key]) for key in guidance_points_list[i].keys()])
            x = np.array(x)

            if production_channel == "ggf":
                def LO_fit_function(values, a, b, c):
                    kl, kt = values
                    return a * kt**2 * kl**2 + b * kt**4 + c * kl * kt**3

                def LO_fit_variances(values, a, b, c):
                    kl, kt = values
                    return a * (kt**2 * kl**2)**2 + b * (kt**4)**2 + c * (kl * kt**3)**2

                def LO_primitive_funtion(values, a, b, c):
                    kl, kt = values
                    return a * kt**3 * kl**3 / 9 + b * kt**5 / 5 + c * kl**2 * kt**4 / 8

            elif production_channel == "vbf":
                def LO_fit_function(values, a, b, c, d, e, f):
                    kv, k2v, kl = values
                    return a * kv**4 + b * k2v**2 + c * (kv**2) * (kl**2) + d * (kv**2) * k2v + e * (kv**3) * kl + f * k2v * kv * kl  # noqa

                def LO_fit_variances(values, a, b, c, d, e, f):
                    kv, k2v, kl = values
                    return a * (kv**4)**2 + b * (k2v**2)**2 + c * (kv**2 * kl**2)**2 + d * (kv**2 * k2v)**2 + e * (kv**3 * kl)**2 + f * (k2v * kv * kl)**2  # noqa

                def LO_primitive_funtion(values, a, b, c, d, e, f):
                    kv, k2v, kl = values

                    return a / 5 * kv**5 + b / 3 * k2v**3 + c / 9 * (kv**3) * (kl**3) + d / 6 * (kv**3) * (k2v**2) + e / 8 * (kv**4) * (kl**2) + f / 8 * (k2v**2) * (kv**2) * (kl**2)  # noqa

            morphed_values = np.zeros(model_values[0].shape)
            morphed_variances = np.zeros(model_values[0].shape)

            pcovs = []
            # minuit_pcovs = []
            # chi2_infos = {}  # should include chi2 value, ndf, p-value, cov and fit status

            # loop over bins and calculate the morphed value
            for cat_ in range(len(model_hists[0].axes[0].centers)):
                for shift in range(len(model_hists[0].axes[1].centers)):
                    for bin_ in range(len(model_hists[0].axes[2].centers)):
                        # fit the bin values to the function
                        y = model_values[:, cat_, shift, bin_]
                        variances = model_variances[:, cat_, shift, bin_]
                        if production_channel == "ggf":
                            p0_fit = [0, 0, 0]
                        elif production_channel == "vbf":
                            p0_fit = [0, 0, 0, 0, 0, 0]

                        # y = np.array([model_hists[i].view().value[bin] for i in range(len(guidance_points))])
                        popt, pcov, infodict_, mesg_, ier_ = curve_fit(
                            LO_fit_function,
                            x.T,
                            y,
                            sigma=np.sqrt(variances),
                            absolute_sigma=True,
                            p0=p0_fit,
                            full_output=True,
                        )

                        # # alternative least square fitting with iminuit
                        # from iminuit import Minuit
                        # # we also need a cost function to fit and import the LeastSquares function
                        # from iminuit.cost import LeastSquares
                        # least_squares = LeastSquares(x.T, y, np.sqrt(variances), LO_fit_function)
                        # if production_channel == "ggf":
                        #     initial_params_dict = {"a": 0, "b": 0, "c": 0}
                        # elif production_channel == "vbf":
                        #     initial_params_dict = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0, "f": 0}
                        # m = Minuit(least_squares, **initial_params_dict)
                        # # m.errordef = Minuit.LIKELIHOOD
                        # m.migrad()  # finds minimum of least_squares function
                        # m.hesse()  # accurately computes uncertainties
                        # fitted_params = np.array(m.values)
                        # # fitted_param_errors = list(m.errors)
                        # fitted_cov = np.array(m.covariance)

                        # TODO : WIP
                        # # alternative fitting with likelihood
                        # from iminuit import Minuit
                        # # we also need a cost function to fit and import the LeastSquares function
                        # from iminuit.cost import ExtendedUnbinnedNLL

                        # def general_pdf(x, *args, channel=production_channel):
                        #     # from PDG Higgs section:
                        #     # https://pdg.lbl.gov/2024/web/viewer.html?file=../reviews/rpp2024-rev-higgs-boson.pdf
                        #     # kt is measured at 1.01 +- 0.11 by CMS and 0.94 +- 0.11 by ATLAS
                        #     # Therefore, we will constrain it at the first integer above 5\sigma: [0, 2]
                        #     #
                        #     # kw is measured at 1.02 +- 0.08 by CMS and 1.05 +- 0.06 by ATLAS
                        #     # kz is measured at 1.04 +- 0.07 by CMS and 0.99 +- 0.06 by ATLAS
                        #     # as the kv variations in the samples are much larger, we will constrain it to [-2.5,2]
                        #     # to fit all samples inside the range
                        #     #
                        #     # kl is constrained to [-1.2,6.5] by CMS and [-0.6, 6.6] by ATLAS at 95% CL in HH analyses
                        #     # (ATLAS has [-0.4, 6.3] for single Higgs analyses)
                        #     # as the kl variations in the samples are much larger, we will constrain it to [-20,15]
                        #     # to fit all samples inside the range
                        #     #
                        #     # k2v is constrained to [-0.1, 2.2] by CMS and [-0.1,2.0] by ATLAS at 95% CL
                        #     # To emulate a 5\sigma CL for a gaussian, we will constrain it to [-3,4]
                        #     # which fits all samples inside the range

                        #     if channel == "ggf":
                        #         kl, kt = x
                        #         c_kl = [-20, 15]
                        #         c_kt = [0, 2]
                        #         if kl < c_kl[0] or kl > c_kl[1] or kt < c_kt[0] or kt > c_kt[1]:
                        #             return 0
                        #         else:
                        #             integral_1 = LO_primitive_funtion([c_kl[1], c_kt[1]], *args)
                        #             integral_2 = LO_primitive_funtion([c_kl[0], c_kt[1]], *args)
                        #             integral_3 = LO_primitive_funtion([c_kl[1], c_kt[0]], *args)
                        #             integral_4 = LO_primitive_funtion([c_kl[0], c_kt[0]], *args)
                        #             integral = integral_1 - integral_2 - integral_3 + integral_4
                        #     elif channel == "vbf":
                        #         kv, k2v, kl = x
                        #         c_kv = [-2.5, 2]
                        #         c_k2v = [-3, 4]
                        #         c_kl = [-20, 15]
                        #         if kv < c_kv[0] or kv > c_kv[1] or k2v < c_k2v[0] or k2v > c_k2v[1] or kl < c_kl[0] or kl > c_kl[1]:  # noqa
                        #             return 0
                        #         else:
                        #             integral_1 = LO_primitive_funtion([c_kv[1], c_k2v[1], c_kl[1]], *args)
                        #             integral_2 = LO_primitive_funtion([c_kv[1], c_k2v[1], c_kl[0]], *args)
                        #             integral_3 = LO_primitive_funtion([c_kv[1], c_k2v[0], c_kl[1]], *args)
                        #             integral_4 = LO_primitive_funtion([c_kv[0], c_k2v[1], c_kl[1]], *args)
                        #             integral_5 = LO_primitive_funtion([c_kv[0], c_k2v[0], c_kl[1]], *args)
                        #             integral_6 = LO_primitive_funtion([c_kv[0], c_k2v[1], c_kl[0]], *args)
                        #             integral_7 = LO_primitive_funtion([c_kv[1], c_k2v[0], c_kl[0]], *args)
                        #             integral_8 = LO_primitive_funtion([c_kv[0], c_k2v[0], c_kl[0]], *args)
                        #             integral = integral_1 - integral_2 - integral_3 - integral_4 + integral_5 + integral_6 + integral_7 - integral_8  # noqa
                        #     return LO_fit_function(x, *args) / integral

                        # pdf = partial(general_pdf, channel=production_channel)

                        # likelihood = ExtendedUnbinnedNLL(y, pdf)
                        # # from iminuit.cost import LeastSquares
                        # # least_squares = LeastSquares(x.T, y, np.sqrt(variances), LO_fit_function)
                        # m = Minuit(likelihood, a=0, b=0, c=0)
                        # # m.errordef = Minuit.LIKELIHOOD
                        # m.migrad()  # finds minimum of least_squares function
                        # m.hesse()  # accurately computes uncertainties
                        # popt = list(m.values)
                        # from IPython import embed; embed(header="fitting, check errors")
                        # # pcov = list(m.errors)

                        # calculate the morphed value
                        target_param_values = [float(target_params_oi[key]) for key in target_params_oi.keys()]
                        morphed_value = LO_fit_function(target_param_values, *popt)
                        fit_variance = np.diag(pcov)
                        morphed_variance = LO_fit_variances(target_param_values, *fit_variance)

                        morphed_values[cat_, shift, bin_] = morphed_value
                        morphed_variances[cat_, shift, bin_] = morphed_variance
                        # if cat_ == 18 and bin_ == 5:
                        #     print(f"cat: {cat_}, shift: {shift}, bin: {bin_}")
                        #     print(f"morphed_value: {morphed_value}")
                        #     print(f"morphed_variance: {morphed_variance}")
                        #     from IPython import embed; embed()
                        pcovs.append(pcov)
                        # minuit_pcovs.append(fitted_cov)

            pcovs = np.array(pcovs)
            # minuit_pcovs = np.array(minuit_pcovs)
            morphed_variances = np.matmul(
                new_coefficients,
                np.matmul(
                    pcovs,
                    new_coefficients,
                ).T,
            ).reshape(original_hist_shape)

            # TODO: check why uncertainties fit and weights method are not the same
            # from IPython import embed; embed(header="fitting")

        # reshape the values to the correct categorization
        morphed_values_correct_categorization = morphed_values[np.argsort(np.argsort(model_hists[0].axes[0]))]
        morphed_variances_correct_categorization = morphed_variances[np.argsort(np.argsort(model_hists[0].axes[0]))]

        # insert the new hist into the hists dict
        new_proc.x.guidance_points_procs = model_procs
        new_proc.x.morphing_type = morphing_type
        new_proc.x.target_point = target_point
        new_proc.x.production_channel = production_channel
        new_proc.x.guidance_points_values = guidance_points_list_float

        # difficult to use for plotting since the categories to be added are not known
        # so maybe add as a separate histogram?
        # new_proc.x.chi2_infos = chi2_infos

        hists[new_proc] = new_hist

        # insert values into the new histogram
        hists[new_proc].view().value = morphed_values_correct_categorization
        hists[new_proc].view().variance = morphed_variances_correct_categorization

        # from IPython import embed; embed(header="morphing")

        return hists

    config.x.hist_hooks = {
        "qcd": qcd_estimation,
        "flat_s": flat_s,

        "hh_ggf_exact_morphing_kl0_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl0_kt1",
            morphing_type="exact",
        ),
        "hh_ggf_exact_morphing_kl1_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl1_kt1",
            morphing_type="exact",
        ),
        "hh_ggf_exact_morphing_kl2_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1"],
            target_point="kl2_kt1",
            morphing_type="exact",
        ),
        "hh_ggf_exact_morphing_kl2p45_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl5_kt1"],
            target_point="kl2p45_kt1",
            morphing_type="exact",
        ),
        "hh_ggf_exact_morphing_kl3_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1"],
            target_point="kl3_kt1",
            morphing_type="exact",
        ),
        "hh_ggf_exact_morphing_kl4_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1"],
            target_point="kl4_kt1",
            morphing_type="exact",
        ),
        "hh_ggf_exact_morphing_kl5_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1"],
            target_point="kl5_kt1",
            morphing_type="exact",
        ),

        "hh_ggf_average_morphing_kl0_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl0_kt1",
            morphing_type="average",
        ),
        "hh_ggf_average_morphing_kl1_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl1_kt1",
            morphing_type="average",
        ),
        "hh_ggf_average_morphing_kl2_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl2_kt1",
            morphing_type="average",
        ),
        "hh_ggf_average_morphing_kl2p45_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl2p45_kt1",
            morphing_type="average",
        ),
        "hh_ggf_average_morphing_kl3_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl3_kt1",
            morphing_type="average",
        ),
        "hh_ggf_average_morphing_kl4_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl4_kt1",
            morphing_type="average",
        ),
        "hh_ggf_average_morphing_kl5_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl5_kt1",
            morphing_type="average",
        ),

        "hh_ggf_weight_morphing_kl0_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl0_kt1",
            morphing_type="average",
            weights=True,
        ),

        "hh_ggf_weight_morphing_kl1_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl1_kt1",
            morphing_type="average",
            weights=True,
        ),

        "hh_ggf_weight_morphing_kl2_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl2_kt1",
            morphing_type="average",
            weights=True,
        ),

        "hh_ggf_weight_morphing_kl2p45_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl2p45_kt1",
            morphing_type="average",
            weights=True,
        ),

        "hh_ggf_weight_morphing_kl3_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl3_kt1",
            morphing_type="average",
            weights=True,
        ),

        "hh_ggf_weight_morphing_kl4_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl4_kt1",
            morphing_type="average",
            weights=True,
        ),

        "hh_ggf_weight_morphing_kl5_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl5_kt1",
            morphing_type="average",
            weights=True,
        ),


        "hh_ggf_fit_morphing_kl0_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl0_kt1",
            morphing_type="fit",
        ),
        "hh_ggf_fit_morphing_kl1_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl1_kt1",
            morphing_type="fit",
        ),
        "hh_ggf_fit_morphing_kl2_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl2_kt1",
            morphing_type="fit",
        ),
        "hh_ggf_fit_morphing_kl2p45_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl2p45_kt1",
            morphing_type="fit",
        ),
        "hh_ggf_fit_morphing_kl3_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl3_kt1",
            morphing_type="fit",
        ),
        "hh_ggf_fit_morphing_kl4_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl4_kt1",
            morphing_type="fit",
        ),
        "hh_ggf_fit_morphing_kl5_kt1": partial(
            general_higgs_morphing,
            production_channel="ggf",
            guidance_points=["kl0_kt1", "kl1_kt1", "kl2p45_kt1", "kl5_kt1"],
            target_point="kl5_kt1",
            morphing_type="fit",
        ),

        "hh_vbf_exact_morphing_kv1_k2v1_kl2": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm1p21_k2v1p94_klm0p94",  # kvm0p012_k2v0p03_kl10p2
            ],
            target_point="kv1_k2v1_kl2",
            morphing_type="exact",
        ),

        "hh_vbf_exact_morphing_kv1_k2v1_kl1": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v0_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm1p21_k2v1p94_klm0p94", "kvm0p012_k2v0p03_kl10p2",
                "kvm0p962_k2v0p959_klm1p43",
            ],
            target_point="kv1_k2v1_kl1",
            morphing_type="exact",
        ),

        # all points
        # "hh_vbf_average_morphing_kv1_k2v1_kl5": partial(
        #     general_higgs_morphing,
        #     production_channel="vbf",
        #     guidance_points=[
        #         "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
        #         "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
        #         "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
        #         "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
        #         "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96"
        #     ],
        #     target_point="kv1_k2v1_kl5",
        #     morphing_type="fit",
        # ),

        # all points except SM
        "hh_vbf_average_morphing_kv1_k2v1_kl5": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v1_kl5",
            morphing_type="average",
        ),

        "hh_vbf_average_morphing_kv1_k2v1_kl1": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v1_kl1",
            morphing_type="average",
        ),

        "hh_vbf_fit_morphing_kv1_k2v1_kl5": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v1_kl5",
            morphing_type="fit",
        ),

        "hh_vbf_fit_morphing_kv1_k2v1_kl1": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v1_kl1",
            morphing_type="fit",
        ),

        # all points except the one to be morphed
        "hh_vbf_average_morphing_kv1_k2v0_kl1": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v0_kl1",
            morphing_type="average",
        ),

        "hh_vbf_fit_morphing_kv1_k2v0_kl1": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v0_kl1",
            morphing_type="fit",
        ),

        "hh_vbf_average_morphing_kv1_k2v1_kl2": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v1_kl2",
            morphing_type="average",
        ),

        "hh_vbf_fit_morphing_kv1_k2v1_kl2": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v1_kl2",
            morphing_type="fit",
        ),

        "hh_vbf_average_morphing_kv1_k2v2_kl1": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v2_kl1",
            morphing_type="average",
        ),

        "hh_vbf_fit_morphing_kv1_k2v2_kl1": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v2_kl1",
            morphing_type="fit",
        ),

        "hh_vbf_average_morphing_kv1p74_k2v1p37_kl14p4": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1p74_k2v1p37_kl14p4",
            morphing_type="average",
        ),

        "hh_vbf_fit_morphing_kv1p74_k2v1p37_kl14p4": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1p74_k2v1p37_kl14p4",
            morphing_type="fit",
        ),

        "hh_vbf_average_morphing_kvm0p758_k2v1p44_klm19p3": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kvm0p758_k2v1p44_klm19p3",
            morphing_type="average",
        ),

        "hh_vbf_fit_morphing_kvm0p758_k2v1p44_klm19p3": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kvm0p758_k2v1p44_klm19p3",
            morphing_type="fit",
        ),

        "hh_vbf_average_morphing_kvm0p012_k2v0p03_kl10p2": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kvm0p012_k2v0p03_kl10p2",
            morphing_type="average",
        ),

        "hh_vbf_fit_morphing_kvm0p012_k2v0p03_kl10p2": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kvm0p012_k2v0p03_kl10p2",
            morphing_type="fit",
        ),

        "hh_vbf_average_morphing_kvm0p962_k2v0p959_klm1p43": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kvm0p962_k2v0p959_klm1p43",
            morphing_type="average",
        ),

        "hh_vbf_fit_morphing_kvm0p962_k2v0p959_klm1p43": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kvm0p962_k2v0p959_klm1p43",
            morphing_type="fit",
        ),

        "hh_vbf_average_morphing_kvm1p21_k2v1p94_klm0p94": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kvm1p21_k2v1p94_klm0p94",
            morphing_type="average",
        ),

        "hh_vbf_fit_morphing_kvm1p21_k2v1p94_klm0p94": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kvm1p21_k2v1p94_klm0p94",
            morphing_type="fit",
        ),

        "hh_vbf_average_morphing_kvm1p6_k2v2p72_klm1p36": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kvm1p6_k2v2p72_klm1p36",
            morphing_type="average",
        ),

        "hh_vbf_fit_morphing_kvm1p6_k2v2p72_klm1p36": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kvm1p6_k2v2p72_klm1p36",
            morphing_type="fit",
        ),

        "hh_vbf_average_morphing_kvm1p83_k2v3p57_klm3p39": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kvm1p83_k2v3p57_klm3p39",
            morphing_type="average",
        ),

        "hh_vbf_fit_morphing_kvm1p83_k2v3p57_klm3p39": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kvm1p83_k2v3p57_klm3p39",
            morphing_type="fit",
        ),

        "hh_vbf_average_morphing_kvm2p12_k2v3p87_klm5p96": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39",
            ],
            target_point="kvm2p12_k2v3p87_klm5p96",
            morphing_type="average",
        ),

        "hh_vbf_fit_morphing_kvm2p12_k2v3p87_klm5p96": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39",
            ],
            target_point="kvm2p12_k2v3p87_klm5p96",
            morphing_type="fit",
        ),


        # all points
        "hh_vbf_average_morphing_kv1_k2v1_kl0_all_points": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v1_kl0",
            morphing_type="average",
        ),

        "hh_vbf_fit_morphing_kv1_k2v1_kl0_all_points": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v1_kl0",
            morphing_type="fit",
        ),

        "hh_vbf_average_morphing_kv1_k2v1_kl1_all_points": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v1_kl1",
            morphing_type="average",
        ),

        "hh_vbf_weight_morphing_kv1_k2v1_kl1_all_points": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v1_kl1",
            morphing_type="average",
            weights=True,
        ),

        "hh_vbf_fit_morphing_kv1_k2v1_kl1_all_points": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v1_kl1",
            morphing_type="fit",
        ),

        "hh_vbf_average_morphing_kv1_k2v1_kl5_all_points": partial(
            general_higgs_morphing,
            production_channel="vbf",
            guidance_points=[
                "kv1_k2v1_kl1", "kv1_k2v0_kl1", "kv1_k2v1_kl2", "kv1_k2v2_kl1",
                "kv1p74_k2v1p37_kl14p4", "kvm0p758_k2v1p44_klm19p3",
                "kvm0p012_k2v0p03_kl10p2", "kvm0p962_k2v0p959_klm1p43",
                "kvm1p21_k2v1p94_klm0p94", "kvm1p6_k2v2p72_klm1p36",
                "kvm1p83_k2v3p57_klm3p39", "kvm2p12_k2v3p87_klm5p96",
            ],
            target_point="kv1_k2v1_kl5",
            morphing_type="average",
        ),
    }

# Remark on new Process for morphing: single ID for all morphed processes, but different names
# in order to be able to distinguish them in the plot functions, maybe need to find a way to create
# a unique ID for each morphed process.
