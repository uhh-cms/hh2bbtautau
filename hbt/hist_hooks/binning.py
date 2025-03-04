# coding: utf-8

"""
Histogram hooks for binning changes.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import law
import order as od

from columnflow.util import maybe_import
from columnflow.types import Any, Callable

np = maybe_import("numpy")
hist = maybe_import("hist")


logger = law.logger.get_logger(__name__)


@dataclass
class BinCount:
    # actual bin content
    val: float
    # variance of bin content
    var: float
    # number of equivalent event entries that went into the bin
    # when zero, it is inferred from bin content and error through the usual assumptions
    num: float = 0.0

    def __post_init__(self):
        # compute number of entries assuming that each event in the bin had a similar weight
        if self.num == 0 and self.var != 0:
            self.num = self.val**2 / self.var


@dataclass
class BinningConstraint:
    # tags that identify processes that are relevant for the constraint
    process_tags: list[str]
    # function taking a dictionary {"procA": (yieldA, uncA), ...} and returning true when met
    check: Callable[[dict[str, BinCount]], bool]


# helper to extract the name of the requested category and variable
def get_task_infos(task) -> dict[str, Any]:
    # datacard task
    if "config_category" in task.branch_data:
        return {
            "category_name": task.branch_data.config_category,
            "variable_name": task.branch_data.config_variable,
        }

    # plotting task
    if "category" in task.branch_data:
        return {
            "category_name": task.branch_data.category,
            "variable_name": task.branch_data.variable[0],
        }

    raise Exception(f"cannot determine task infos of unhandled task: {task!r}")


def add_hooks(config: od.Config) -> None:
    """
    Add histogram hooks to a configuration.
    """
    def flat_s(
        task,
        hists: dict[od.Process, hist.Histogram],
        signal_process_name: str = "",
        n_bins: int = 10,
        constraint: BinningConstraint | None = None,
    ) -> dict[od.Process, hist.Histogram]:
        """Rebinnig of the histograms in *hists* to archieve a flat-signal distribution.

        :param task: task instance that contains the process informations
        :param hists: A dictionary of histograms using Process instances as keys

        :raises RuntimeError: If the wanted number of bins is reached and the initial
        bin edge is not minimal.
        :raises Exception: If the number of actual bins ended up larger than requested.
        :return: A dictionary of histograms using Process instances as keys
        """

        # edge finding helper
        def find_edges(
            signal_hist: hist.Hist,
            background_hists: dict[od.Process, hist.Hist],
            variable: str,
            n_bins: int = 10,
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            Determine new bin edges that result in a flat signal distribution.
            The edges are determined by the signal distribution, while the background distribution
            is used to ensure that the background yield in each bin is sufficient.

            :param signal_hist: The histogram that describes the signal distribution.
            :param background_hists: A dictionary of histograms using the process as key
            that describe the background distribution.
            :param variable: The variable name that is rebinned.
            :param n_bins: The number of bins that the signal distribution should be rebinned to.

            :return: A tuple containing the new bin edges and the indices that define the new bin edges.
            """
            # prepare parameters
            low_edge, max_edge = 0, 1
            bin_edges = [max_edge]
            indices_gathering = [0]

            # bookkeep reasons for stopping binning
            stop_reason = ""
            # accumulated signal yield up to the current index
            y_already_binned = 0.0
            y_min = 1.0e-5

            # prepare signal
            # fine binned histograms bin centers are approx equivalent to dnn output
            # flip arrays to start from the right
            dnn_score_signal = np.flip(signal_hist.axes[variable].centers, axis=-1)
            y = np.flip(signal_hist.counts(), axis=-1)

            # set negative yields to zero and warn about it
            neg_mask = y < 0
            if neg_mask.any():
                y[neg_mask] = 0
                if neg_mask.mean() > 0.05:
                    logger.warning(
                        f"found {neg_mask.mean() * 100:.1f}% of the signal bins to be negative",
                    )

            # calculate cumulative of reversed signal yield and yield per bin
            y_cumsum = np.cumsum(y, axis=0)
            y_sum = y_cumsum[-1]
            y_per_bin = y_sum / n_bins
            num_bins_orig = len(y)

            # prepare backgrounds for constraint
            if constraint:
                constraint_data = {}
                for tag in constraint.process_tags:
                    # start with empty data for counts and variances
                    constraint_data[tag] = [np.zeros_like(y), np.zeros_like(y)]
                    # loop over histograms and check if they fit the process
                    for proc, h in background_hists.items():
                        if proc.has_tag(tag):
                            constraint_data[tag][0] += np.flip(h.counts(), axis=-1)
                            constraint_data[tag][1] += np.flip(h.variances(), axis=-1)

            # start binning
            start_idx = 0
            while len(bin_edges) < n_bins:
                # stopping condition 1: reached end of original bins
                if start_idx >= num_bins_orig:
                    stop_reason = "no more source bins left"
                    break
                # stopping condition 2: remaining signal yield too small
                # this would lead to a bin completely filled with background
                y_remaining = y_sum - y_already_binned
                if y_remaining < y_min:
                    stop_reason = "remaining signal yield insufficient"
                    break
                # find the index "stop_idx" of the source bin that marks the start of the next
                # merged bin
                if y_remaining >= y_per_bin:
                    threshold = y_already_binned + y_per_bin
                    # get indices of array of values above threshold
                    # first entry defines the next bin edge
                    # shift by start_idx
                    stop_idx = start_idx + max(np.where(y_cumsum[start_idx:] > threshold)[0][0], 1)
                else:
                    # special case: remaining signal yield smaller than the expected per-bin yield,
                    # so find the last bin
                    stop_idx = start_idx + np.where(y_cumsum[start_idx:])[0][-1] + 1

                # check background constraints
                if constraint:
                    while stop_idx < num_bins_orig:
                        # create the per-bin count objects
                        counts = {
                            tag: BinCount(
                                val=float(constraint_data[tag][0][start_idx:stop_idx].sum()),
                                var=float(constraint_data[tag][1][start_idx:stop_idx].sum()),
                            )
                            for tag in constraint.process_tags
                        }

                        # check if the constraint is met
                        if constraint.check(counts):
                            # TODO: maybe also check if the background conditions are just barely met and advance
                            # stop_idx to the middle between the current value and the next one that would change
                            # anything about the background predictions; this might be more stable as the current
                            # implementation can highly depend on the exact value of a single event (the one that
                            # tips the constraints over the edge to fulfillment)
                            break

                        # constraints not met, advance index to include the next bin and try again
                        stop_idx += 1

                    else:
                        # stopping condition 3: no more source bins left, so the last bin (most left one) does not
                        # fullfill constraints; however, this should practically never happen
                        stop_reason = "no more bins left while trying to fulfill constraints"
                        break

                # stop_idx found, update values
                # get next edge or set to low edge if end is reached
                if stop_idx == num_bins_orig:
                    edge_value = low_edge
                else:
                    # calculate bin center as new edge
                    edge_value = float(dnn_score_signal[stop_idx - 1:stop_idx + 1].mean())
                # prevent out of bounds values and push them to the boundaries
                bin_edges.append(max(min(edge_value, max_edge), low_edge))

                y_already_binned += y[start_idx:stop_idx].sum()
                start_idx = stop_idx
                indices_gathering.append(stop_idx)

            # make sure the lower dnn_output (max events) is included
            if bin_edges[-1] != low_edge:
                if len(bin_edges) > n_bins:
                    raise RuntimeError(
                        "number of bins reached and initial bin edge"
                        f" is not minimal bin edge (edges: {bin_edges})",
                    )
                bin_edges.append(low_edge)
                indices_gathering.append(num_bins_orig)

            # some debugging output
            n_bins_actual = len(bin_edges) - 1
            if n_bins_actual > n_bins:
                raise Exception("number of actual bins ended up larger than requested (implementation bug)")
            if n_bins_actual < n_bins:
                print(
                    f"  started from {num_bins_orig} bins, targeted {n_bins} but ended at {n_bins_actual} bins\n"
                    f"    -> reason: {stop_reason or 'NO REASON!?'}",
                )
                n_bins = n_bins_actual

            # flip indices to the right order
            indices_gathering = (np.flip(indices_gathering) - num_bins_orig) * -1
            return np.flip(np.array(bin_edges), axis=-1), indices_gathering

        # rebinning helper
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

        # extract task infos
        task_infos = get_task_infos(task)

        # find signal histogram for which you will optimize, only 1 signal process is allowed
        signal_proc = None
        signal_hist = None
        background_hists = {}
        for process, j in hists.items():
            if process.has_tag("signal") and (signal_process_name in (process.name, "")):
                if signal_proc:
                    logger.warning("more than one signal process found, use the first one")
                else:
                    signal_proc = process
                    signal_hist = j
            elif process.is_mc:
                background_hists[process] = j

        if not signal_proc:
            logger.warning("could not find any signal process, return hist unchanged")
            return hists

        # 1. preparation
        # get the leaf categories (e.g. {etau,mutau}__os__iso)
        category_inst = task.config_inst.get_category(task_infos["category_name"])
        leaf_cats = (
            [category_inst]
            if category_inst.is_leaf_category
            else category_inst.get_leaf_categories()
        )

        # filter categories not existing in histogram
        cat_ids_locations = [hist.loc(c.id) for c in leaf_cats if c.id in signal_hist.axes["category"]]

        # sum over different leaf categories
        combined_signal_hist = signal_hist[{"category": cat_ids_locations}][{"category": sum}]
        combined_signal_hist = combined_signal_hist[{"shift": hist.loc(0)}]

        # same for background
        for process, j in background_hists.items():
            combined_background_hist = j[{"category": cat_ids_locations}][{"category": sum}]
            combined_background_hist = combined_background_hist[{"shift": hist.loc(0)}]
            background_hists[process] = combined_background_hist

        # 2. determine bin edges
        flat_s_edges, flat_s_indices = find_edges(
            signal_hist=combined_signal_hist,
            background_hists=background_hists,
            variable=task_infos["variable_name"],
            n_bins=n_bins,
        )

        # 3. apply to hists
        for process, histogram in hists.items():
            hists[process] = apply_edges(
                histogram,
                flat_s_edges,
                flat_s_indices,
                task_infos["variable_name"],
            )

        return hists

    # some usual binning constraints
    def constrain_tt_dy(counts: dict[str, BinCount]) -> bool:
        # have at least one tt, one dy, and four total background events
        # as well as positive yields
        return (
            counts["tt"].num >= 1 and
            counts["dy"].num >= 1 and
            counts["tt"].num + counts["dy"].num >= 4 and
            counts["tt"].val > 0 and
            counts["dy"].val > 0
        )

    # add hooks
    config.x.hist_hooks.flats = flat_s
    config.x.hist_hooks.flats_kl1_n10 = functools.partial(
        flat_s,
        signal_process_name="hh_ggf_hbb_htt_kl1_kt1",
        n_bins=10,
    )
    config.x.hist_hooks.flats_kl1_n10_guarded = functools.partial(
        flat_s,
        signal_process_name="hh_ggf_hbb_htt_kl1_kt1",
        n_bins=10,
        constraint=BinningConstraint(["tt", "dy"], constrain_tt_dy),
    )
