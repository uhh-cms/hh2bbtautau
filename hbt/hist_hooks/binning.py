# coding: utf-8

"""
Histogram hooks for binning changes.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from collections import defaultdict

import law
import order as od

from columnflow.util import maybe_import
from columnflow.types import TYPE_CHECKING, Callable

if TYPE_CHECKING:
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


def add_hooks(analysis_inst: od.Analysis) -> None:
    """
    Add histogram hooks to a analysis.
    """
    def flat_s(
        task: law.Task,
        hists: dict[od.Config, dict[od.Process, hist.Hist]],
        category_name: str,
        variable_name: str,
        signal_process_name: str = "",
        n_bins: int = 10,
        constraint: BinningConstraint | None = None,
    ) -> dict[od.Config, dict[od.Process, hist.Hist]]:
        """
        Rebinnig of the histograms in *hists* to archieve a flat-signal distribution.
        """
        import numpy as np
        import hist

        # edge finding helper
        def find_edges(
            signal_hist: hist.Hist,
            background_hists: list[tuple[od.Process, hist.Hist]],
            n_bins: int = 10,
        ) -> np.ndarray:
            """
            Determine new bin edges that result in a flat signal distribution. The edges are determined by the signal
            distribution, while the background distribution is used to ensure that the background yield in each bin is
            sufficient.
            """
            # prepare parameters
            low_edge, max_edge = 0, 1
            bin_edges = [max_edge]

            # bookkeep reasons for stopping binning
            stop_reason = ""
            # accumulated signal yield up to the current index
            y_already_binned = 0.0
            y_min = 1.0e-5

            # prepare signal
            # fine binned histograms bin centers are approx equivalent to dnn output
            # flip arrays to start from the right
            dnn_score_signal = np.flip(signal_hist.axes[-1].centers, axis=-1)
            y = np.flip(signal_hist.counts(), axis=-1)

            # set negative yields to zero and warn about it
            neg_mask = y < 0
            if neg_mask.any():
                y[neg_mask] = 0
                if neg_mask.mean() > 0.05:
                    logger.warning(f"found {neg_mask.mean() * 100:.1f}% of the signal bins to be negative")

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
                    for proc, h in background_hists:
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
                # find the index "stop_idx" of the source bin that marks the start of the next merged bin
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
                            # implementation can highly depend on the exact value of a single event (the one that tips
                            # the constraints over the edge to fulfillment)
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

            # make sure the lower dnn_output (max events) is included
            if bin_edges[-1] != low_edge:
                if len(bin_edges) > n_bins:
                    raise RuntimeError(
                        f"number of bins reached and initial bin edge is not minimal bin edge (edges: {bin_edges})",
                    )
                bin_edges.append(low_edge)

            # some debugging output
            n_bins_actual = len(bin_edges) - 1
            if n_bins_actual > n_bins:
                raise Exception("number of actual bins ended up larger than requested (implementation bug)")
            if n_bins_actual < n_bins:
                print(
                    f"  started from {num_bins_orig} bins, targeted {n_bins} but ended at {n_bins_actual} bins\n"
                    f"    -> reason: {stop_reason or 'NO REASON!?'}",
                )

            # flip back
            return np.flip(np.array(bin_edges), axis=-1)

        # find signal and background histograms
        signal_hist: dict[od.Config, hist.Hist] = {}
        background_hists: dict[od.Config, dict[od.Process, hist.Hist]] = defaultdict(dict)
        for config_inst, proc_hists in hists.items():
            for process_inst, h in proc_hists.items():
                if process_inst.has_tag("signal") and (signal_process_name in (process_inst.name, "")):
                    if config_inst in signal_hist:
                        logger.warning("more than one signal histogram found, use the first one")
                    else:
                        signal_hist[config_inst] = h
                elif process_inst.is_mc:
                    background_hists[config_inst][process_inst] = h
            if config_inst not in signal_hist:
                logger.warning(f"could not find any signal process for config {config_inst}, skip flat_s hook")
                return hists

        # 1. select and sum over requested categories
        for config_inst in hists:
            # get the leaf categories
            category_inst = config_inst.get_category(category_name)
            leaf_cats = (
                [category_inst]
                if category_inst.is_leaf_category
                else category_inst.get_leaf_categories()
            )

            # select leaf categories and nominal shift
            def select(h: hist.Hist) -> hist.Hist:
                # filter to existing categories
                h = h[{"category": [hist.loc(c.name) for c in leaf_cats if c.name in h.axes["category"]]}]
                # sum over categories and select nominal shift
                h = h[{"category": sum, "shift": hist.loc("nominal")}]
                return h

            # sum over different leaf categories and select the nominal shift
            signal_hist[config_inst] = select(signal_hist[config_inst])
            for process_inst, h in background_hists[config_inst].items():
                background_hists[config_inst][process_inst] = select(h)

        # 2. determine bin edges, considering signal and background sums over all configs
        # note: for signal, this assumes that variable axes have the same name, but they probably always will
        signal_sum = sum((signal_hists := list(signal_hist.values()))[1:], signal_hists[0].copy())
        background_sum = sum((list(proc_hists.items()) for proc_hists in background_hists.values()), [])
        flat_s_edges = find_edges(
            signal_hist=signal_sum,
            background_hists=background_sum,
            n_bins=n_bins,
        )
        print(f"edges in {category_name}: {flat_s_edges.tolist()}")

        # 3. apply to hists
        for config_inst, proc_hists in hists.items():
            for process_inst, h in proc_hists.items():
                proc_hists[process_inst] = h[{h.axes[-1].name: hist.rebin(edges=flat_s_edges)}]

        return hists

    # some usual binning constraints
    def constrain_tt_dy(counts: dict[str, BinCount], n_tt: int = 1, n_dy: int = 1, n_sum: int = 4) -> bool:
        # have at least one tt, one dy, and four total background events as well as positive yields
        return (
            counts["tt"].num >= n_tt and
            counts["dy"].num >= n_dy and
            (counts["tt"].num + counts["dy"].num) >= n_sum and
            counts["tt"].val > 0 and
            counts["dy"].val > 0
        )

    # add hooks
    analysis_inst.x.hist_hooks.flats = flat_s
    for n_bins in [10, 15, 20, 30, 40]:
        for kl in ["0", "1", "2p45", "5"]:
            analysis_inst.x.hist_hooks[f"flats_kl{kl}_n{n_bins}"] = functools.partial(
                flat_s,
                signal_process_name=f"hh_ggf_hbb_htt_kl{kl}_kt1",
                n_bins=n_bins,
            )
            analysis_inst.x.hist_hooks[f"flats_kl{kl}_n{n_bins}_guarded"] = functools.partial(
                flat_s,
                signal_process_name=f"hh_ggf_hbb_htt_kl{kl}_kt1",
                n_bins=n_bins,
                constraint=BinningConstraint(["tt", "dy"], constrain_tt_dy),
            )
            analysis_inst.x.hist_hooks[f"flats_kl{kl}_n{n_bins}_guarded5"] = functools.partial(
                flat_s,
                signal_process_name=f"hh_ggf_hbb_htt_kl{kl}_kt1",
                n_bins=n_bins,
                constraint=BinningConstraint(["tt", "dy"], functools.partial(constrain_tt_dy, n_tt=5, n_dy=5, n_sum=10)),
            )
