# coding: utf-8

"""
Histogram hooks for the in-situ QCD estimation.

The ABCD method is split into two parts:

  - this hook computes the transfer factor TF from the same-sign regions C/D and writes the histograms of the
    opposite-sign, non-isolated region B into the signal region A, scaled by TF, as separate "control region"
    processes: one for data and one per background process
  - the subtraction QCD_A = TF * (data_B - sum(bkg_B)) is left to the inference tool, which applies a frozen factor of
    -1 to the background copies (see hbt/inference/qcd_in_situ.py), so that systematic variations of the backgrounds
    in A and in B are correlated in the fit

The statistical uncertainty of TF is written as the up/down variations of the shift source *tf_shift_source* to all
control region histograms.
"""

from __future__ import annotations

import collections
import functools

import law
import order as od
import scinum as sn

from columnflow.util import maybe_import, DotDict
from columnflow.hist_util import sum_hists, sum_hists_shift_aware
from columnflow.types import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    hist = maybe_import("hist")


logger = law.logger.get_logger(__name__)

# name of the config process representing data in the qcd control region, and name pattern of the processes
# representing a background process in that region
DATA_CR_PROCESS = "qcd_cr_data"
BKG_CR_PROCESS = "qcd_cr_{name}"


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


# helper to ensure that a specific value exists on a categorical axis of a histogram
def ensure_axis_value(h: hist.Histogram, axis_name: str, value: str) -> hist.Histogram:
    ax = h.axes[axis_name]
    if value in ax:
        return h
    dummy_fill = {_ax.name: _ax[0] for _ax in h.axes if _ax.name != axis_name}
    h.fill(**dummy_fill, **{axis_name: value}, weight=0.0)
    return h


def add_hooks(analysis_inst: od.Analysis) -> None:
    """
    Add histogram hooks to a analysis.
    """
    def qcd_in_situ_per_config(
        task: law.Task,
        config_inst: od.Config,
        hists: dict[od.Process, Any],
        requested_category: str | None = None,
        # whether to do sum leaf categories first for all data and mc histograms and then performing the ABCD method,
        # opposed to doing the ABCD method per leaf category and then summing up the resulting qcd histograms
        sum_leaves_first: bool = True,
        # strategy for shape transfer
        shape_transfer: Literal["from_os_noniso", "from_ss_iso"] = "from_os_noniso",
        # name of the shift source whose up/down variations receive the statistical uncertainty of the transfer factor
        tf_shift_source: str = "qcd_tf",
        **kwargs,
    ) -> dict[od.Process, Any]:
        import numpy as np
        import hist

        if shape_transfer not in (known_shape_transfers := {"from_os_noniso", "from_ss_iso"}):
            raise ValueError(
                f"unknown shape transfer strategy '{shape_transfer}', known strategies are: "
                f"{', '.join(known_shape_transfers)}",
            )

        # get the process representing data in the control region, do nothing if it does not exist
        data_cr_proc = config_inst.get_process(DATA_CR_PROCESS, default=None)
        if not data_cr_proc:
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

        # get ABCD categories corresponding to the requested category
        requested_group = None
        # determine the group corresponding to the requested category
        # (disable this check if the qcd estimation should be done on granular leaf categories again)
        if sum_leaves_first and requested_category:
            def find_cat(cat_name, cat_tags, check_group=None):
                cat_inst = config_inst.get_category(cat_name)
                if not cat_inst.has_tag(cat_tags, mode=all):
                    raise ValueError(
                        f"requested category {cat_name} does not have the required tags {cat_tags} for the "
                        f"ABCD method.",
                    )
                if check_group and cat_inst.x.qcd_group != check_group:
                    raise ValueError(
                        f"requested category {cat_name} is not part of the same ABCD group {check_group} as the "
                        f"other categories.",
                    )
                return cat_inst

            cat_inst_req = DotDict()
            cat_inst_req.os_iso = find_cat(requested_category, {"os", "iso"})
            requested_group = cat_inst_req.os_iso.x.qcd_group
            cat_inst_req.ss_iso = find_cat(requested_category.replace("os__iso", "ss__iso"), {"ss", "iso"}, requested_group)  # noqa: E501
            cat_inst_req.os_noniso = find_cat(requested_category.replace("os__iso", "os__noniso"), {"os", "noniso"}, requested_group)  # noqa: E501
            cat_inst_req.ss_noniso = find_cat(requested_category.replace("os__iso", "ss__noniso"), {"ss", "noniso"}, requested_group)  # noqa: E501

        # create qcd groups
        qcd_groups: dict[str, dict[str, list[od.Category]]] = collections.defaultdict(DotDict)
        for cat_name in category_names:
            cat_inst = config_inst.get_category(cat_name)
            # store references to the four category objects
            region_key = None
            if cat_inst.has_tag({"os", "iso"}, mode=all):
                region_key = "os_iso"
            elif cat_inst.has_tag({"os", "noniso"}, mode=all):
                region_key = "os_noniso"
            elif cat_inst.has_tag({"ss", "iso"}, mode=all):
                region_key = "ss_iso"
            elif cat_inst.has_tag({"ss", "noniso"}, mode=all):
                region_key = "ss_noniso"
            else:
                continue

            qcd_groups[cat_inst.x.qcd_group].setdefault(region_key, []).append(cat_inst)

            # store the group corresponding to the requested category (if set)
            if requested_group:
                if any(c.has_category(cat_name, deep=True) for c in cat_inst_req.values()):
                    qcd_groups[requested_group].setdefault(region_key, []).append(cat_inst)

        # get complete qcd groups, potentially only selecting the one corresponding to the requested category
        if requested_group:
            complete_groups = [requested_group] if len(qcd_groups[requested_group]) == 4 else []
        else:
            complete_groups = [
                name for name, cats in qcd_groups.items()
                if len(cats) == 4
            ]

        # nothing to do if there are no complete groups
        if not complete_groups:
            task.logger.warning("no complete ABCD groups found, skipping in-situ QCD estimation")
            return hists

        # sum up mc and data histograms, stop early when empty
        mc_procs = [p for p in hists if p.is_mc and not p.has_tag("signal")]
        if not mc_procs:
            task.logger.warning("no MC histograms found, skipping in-situ QCD estimation")
            return hists
        data_hists = [h for p, h in hists.items() if p.is_data]
        if not data_hists:
            task.logger.warning("no data histograms found, skipping in-situ QCD estimation")
            return hists
        mc_hist = sum_hists_shift_aware([hists[p] for p in mc_procs])
        data_hist = sum_hists(data_hists)

        # find the control region counterpart of each background process
        bkg_cr_procs: dict[od.Process, od.Process] = {}
        for proc in mc_procs:
            cr_proc = config_inst.get_process(BKG_CR_PROCESS.format(name=proc.name), default=None)
            if not cr_proc:
                raise ValueError(
                    f"no control region process '{BKG_CR_PROCESS.format(name=proc.name)}' found in config "
                    f"'{config_inst.name}' for background process '{proc.name}'",
                )
            bkg_cr_procs[proc] = cr_proc

        # names of the nominal shift and the shifts carrying the transfer factor uncertainty
        nominal = config_inst.get_shift("nominal").name
        tf_shifts = [f"{tf_shift_source}_{d}" for d in ["up", "down"]]

        # create empty output histograms with the same axes as the input ones, plus the transfer factor shifts
        def prepare_output(h: hist.Histogram) -> hist.Histogram:
            h = h.copy().reset()
            for shift_name in tf_shifts:
                h = ensure_axis_value(h, "shift", shift_name)
            return h

        cr_hists: dict[od.Process, hist.Histogram] = {data_cr_proc: prepare_output(data_hist)}
        cr_hists.update({cr_proc: prepare_output(hists[proc]) for proc, cr_proc in bkg_cr_procs.items()})

        for group_name in complete_groups:
            group = qcd_groups[group_name]

            # define the target category
            if requested_group:
                target_category = requested_category
            else:
                for key, cats in group.items():
                    if len(cats) > 1:
                        raise ValueError(f"ABCD group {group_name} has multiple categories for region {key}")
                target_category = group.os_iso[0].name

            # helper to get the histogram of a region, summed over its categories
            # shapes: (SHIFT, VAR)
            def get_hist(h: hist.Histogram, region_name: str) -> hist.Histogram | None:
                cat_names = [cat.name for cat in group[region_name] if cat.name in h.axes["category"]]
                if not cat_names:
                    return None
                h = h[{"category": [hist.loc(cat_name) for cat_name in cat_names]}]
                h = h[{"category": sum}]
                return h

            # helper to integrate the nominal histogram of a region into a number with uncertainty
            def integrate_region(h: hist.Histogram, region_name: str, unc_name: str) -> sn.Number:
                h = get_hist(h, region_name)
                if h is None:
                    return sn.Number(0.0, {unc_name: 0.0})
                return integrate_num(hist_to_num(h[{"shift": hist.loc(nominal)}], unc_name))

            # assign regions depending on the shape transfer strategy
            # (B: shape transfer, C/D: transfer factor)
            if shape_transfer == "from_os_noniso":
                b_region, c_region, d_region = "os_noniso", "ss_iso", "ss_noniso"
            else:  # from_ss_iso
                b_region, c_region, d_region = "ss_iso", "os_noniso", "ss_noniso"

            # transfer factor from nominal C and D integrals, with statistical uncertainties from data and mc
            # shapes: ()
            c_int = (
                integrate_region(data_hist, c_region, f"{c_region}_data") -
                integrate_region(mc_hist, c_region, f"{c_region}_mc")
            )
            d_int = (
                integrate_region(data_hist, d_region, f"{d_region}_data") -
                integrate_region(mc_hist, d_region, f"{d_region}_mc")
            )
            # log nominal ingredients of the estimation
            task.logger.info(
                f"in-situ QCD estimation for group {group_name} (nominal): "
                f"B data = {float(integrate_region(data_hist, b_region, 'b_data')()):.3f}, "
                f"B mc = {float(integrate_region(mc_hist, b_region, 'b_mc')()):.3f}, "
                f"C data - mc = {float(c_int()):.3f}, D data - mc = {float(d_int()):.3f}; categories: " +
                ", ".join(f"{region}: {[c.name for c in cats]}" for region, cats in group.items()),
            )
            if not (np.isfinite(c_int()) and np.isfinite(d_int()) and c_int() > 0 and d_int() > 0):
                task.logger.warning(
                    f"non-positive QCD integrals in group {group_name} ({c_region}: {float(c_int())}, "
                    f"{d_region}: {float(d_int())}), transfer factor set to zero",
                )
                tf_nom = tf_up = tf_down = 0.0
            else:
                tf = c_int / d_int
                tf_nom = float(tf())
                tf_up = tf_nom + float(tf(sn.UP, unc=True))
                tf_down = tf_nom - float(tf(sn.DOWN, unc=True))
                task.logger.info(
                    f"QCD transfer factor in group {group_name}: {tf_nom:.4f} +{tf_up - tf_nom:.4f} "
                    f"-{tf_nom - tf_down:.4f}",
                )

            # helper to insert the B histogram, scaled by the transfer factor, into the target category of the output
            def fill_output(h_out: hist.Histogram, h_in: hist.Histogram) -> None:
                h_b = get_hist(h_in, b_region)
                if h_b is None:
                    return
                h_out = ensure_axis_value(h_out, "category", target_category)
                cat_index = h_out.axes["category"].index(target_category)
                out_view = h_out.view(flow=True)
                b_view = h_b.view(flow=True)

                # all shifts of the input histogram, scaled by the nominal transfer factor
                for shift_name in h_b.axes["shift"]:
                    out_index = h_out.axes["shift"].index(shift_name)
                    in_index = h_b.axes["shift"].index(shift_name)
                    out_view.value[cat_index, out_index] = b_view.value[in_index] * tf_nom
                    out_view.variance[cat_index, out_index] = b_view.variance[in_index] * tf_nom**2

                # nominal input, scaled by the varied transfer factors
                in_index = h_b.axes["shift"].index(nominal)
                for shift_name, tf_var in zip(tf_shifts, [tf_up, tf_down]):
                    out_index = h_out.axes["shift"].index(shift_name)
                    out_view.value[cat_index, out_index] = b_view.value[in_index] * tf_var
                    out_view.variance[cat_index, out_index] = b_view.variance[in_index] * tf_var**2

            fill_output(cr_hists[data_cr_proc], data_hist)
            for proc, cr_proc in bkg_cr_procs.items():
                fill_output(cr_hists[cr_proc], hists[proc])

        # add the control region histograms
        hists.update(cr_hists)

        return hists

    def qcd_in_situ(
        task: law.Task,
        hists: dict[od.Config, dict[od.Process, Any]],
        category_name: str,
        variable_name: str,
        **kwargs,
    ) -> dict[od.Config, dict[od.Process, Any]]:
        return {
            config_inst: qcd_in_situ_per_config(
                task,
                config_inst,
                hists[config_inst],
                requested_category=category_name,
                **kwargs,
            )
            for config_inst in hists.keys()
        }

    # add different hook variations
    analysis_inst.x.hist_hooks.qcd_in_situ = qcd_in_situ
    analysis_inst.x.hist_hooks.qcd_in_situ_from_ss_iso = functools.partial(qcd_in_situ, shape_transfer="from_ss_iso")
