# coding: utf-8

"""
Histogram hooks for the in-situ QCD estimation.

The ABCD method is split into two parts:^

  - this hook computes the transfer factor TF from the same-sign regions C/D and writes the histograms of the
    opposite-sign, non-isolated region B into the signal region A, scaled by TF, as separate "control region"
    processes: one for data and one per background process
  - the subtraction QCD_A = TF * (data_B - sum(bkg_B)) is left to the inference tool, which applies a frozen factor of
    -1 to the background copies (see hbt/inference/qcd_in_situ.py), so that systematic variations of the backgrounds
    in A and in B are correlated in the fit

Uncertainty treatment (see also hbt/inference/qcd_in_situ.py for the datacard side):

  - Bin-by-bin statistics: every control region template carries the variance of its B histogram scaled by TF^2
    (data: Poisson counts, MC: sum of squared weights). The fit combines them per bin via autoMCStats. With
    *variance_strategy* "mc" the data template is written with zero variances and therefore does not contribute to
    autoMCStats (only the MC subtraction does); "none" drops all variances. This mirrors the option of the a-priori
    hook, whose default "mc_data" keeps both as well.
  - TF statistics: TF = (data_C - MC_C) / (data_D - MC_D) is computed from the nominal, integrated C and D regions
    with Poisson uncertainties for data and sum-of-squared-weights uncertainties for MC, propagated uncorrelated
    through the ratio. The result is written as the up/down variations of the shift source *tf_shift_source* to all
    control region templates, data and backgrounds alike, in log-normal form: nominal B histogram times
    TF * (1 + sigma_TF / TF) and TF / (1 + sigma_TF / TF). The inference model converts their integral effect into
    one rate (lnN) nuisance per era and category. The log-normal form keeps the templates positive and the lnN
    encodable also when sigma_TF / TF exceeds 100% (a warning is issued above 50%).
  - MC systematic shifts: for every systematic shift present in the MC histograms, TF is recomputed from the shifted
    MC in C and D (data stays nominal). Background copies receive their shifted B histogram times TF(shift); the data
    copy receives its nominal B histogram times TF(shift) for every MC shift. The datacard attaches all shape
    nuisances acting on a background copy to the data copy as well, so a variation moves TF * (data_B - sum(bkg_B))
    coherently. Rate (lnN) nuisances of the datacard do not enter TF.
  - A systematic shift with a non-positive C or D integral falls back to the nominal TF with a warning; a non-positive
    nominal C or D sets TF and its variations to zero.
  - No clipping: negative bins of data_B - sum(bkg_B) are kept; only the total expectation per bin is relevant in the
    fit.
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
        # whether to recompute the transfer factor for every systematic shift of the mc histograms and use it for the
        # corresponding shifted background templates (the data template always uses the nominal transfer factor)
        tf_per_shift: bool = True,
        # which per-bin variances to keep in the control region templates (same meaning as in the a-priori hook):
        # "mc_data" keeps all, "mc" sets the variances of the data template to zero so that only mc statistics enter
        # autoMCStats, "none" sets all variances to zero
        variance_strategy: Literal["mc_data", "mc", "none"] = "mc_data",
        **kwargs,
    ) -> dict[od.Process, Any]:
        import numpy as np
        import hist

        if variance_strategy not in (known_variance_strategies := {"mc_data", "mc", "none"}):
            raise ValueError(
                f"unknown variance strategy '{variance_strategy}', known strategies are: "
                f"{', '.join(known_variance_strategies)}",
            )

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
        # (the data copy additionally receives all shifts of the mc histograms, as its transfer factor depends on them)
        def prepare_output(h: hist.Histogram, extra_shifts: list[str]) -> hist.Histogram:
            h = h.copy().reset()
            for shift_name in list(extra_shifts) + tf_shifts:
                h = ensure_axis_value(h, "shift", shift_name)
            return h

        mc_shift_names = [s for s in mc_hist.axes["shift"] if s not in tf_shifts]
        cr_hists: dict[od.Process, hist.Histogram] = {data_cr_proc: prepare_output(data_hist, mc_shift_names)}
        cr_hists.update({cr_proc: prepare_output(hists[proc], []) for proc, cr_proc in bkg_cr_procs.items()})

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
                # relative statistical uncertainty, written as log-normal variations TF * (1 + rel) and
                # TF / (1 + rel) so that the templates stay positive and the inference model can encode the integral
                # effect as a (symmetric) lnN nuisance even when the relative uncertainty exceeds 100%
                tf_rel_unc = float(tf(sn.UP, unc=True)) / tf_nom
                tf_up = tf_nom * (1.0 + tf_rel_unc)
                tf_down = tf_nom / (1.0 + tf_rel_unc)
                task.logger.info(
                    f"QCD transfer factor in group {group_name}: {tf_nom:.4f} +-{tf_nom * tf_rel_unc:.4f} "
                    f"({100 * tf_rel_unc:.1f}%, log-normal variations {tf_down:.4f} / {tf_up:.4f})",
                )
                if tf_rel_unc > 0.5:
                    task.logger.warning(
                        f"relative uncertainty of the QCD transfer factor in group {group_name} is "
                        f"{100 * tf_rel_unc:.0f}%, the C/D regions have too few events for a reliable estimate",
                    )

            # transfer factors per systematic shift of the mc histograms (data is only available for the nominal
            # shift), used for the shifted templates of all control region copies so that a variation of a background
            # also propagates through the mc subtraction in the C and D regions
            tf_shift: dict[str, float] = {shift_name: tf_nom for shift_name in mc_shift_names}
            if tf_per_shift:
                # helper to integrate the mc histogram of a region for a specific shift, falling back to nominal
                def integrate_mc_region(region_name: str, shift_name: str) -> float:
                    h = get_hist(mc_hist, region_name)
                    if h is None:
                        return 0.0
                    if shift_name not in h.axes["shift"]:
                        shift_name = nominal
                    return float(h[{"shift": hist.loc(shift_name)}].values().sum())

                c_data = float(integrate_region(data_hist, c_region, "c_data")())
                d_data = float(integrate_region(data_hist, d_region, "d_data")())
                for shift_name in mc_shift_names:
                    if shift_name == nominal:
                        continue
                    c_shift = c_data - integrate_mc_region(c_region, shift_name)
                    d_shift = d_data - integrate_mc_region(d_region, shift_name)
                    if not (np.isfinite(c_shift) and np.isfinite(d_shift) and c_shift > 0 and d_shift > 0):
                        task.logger.warning(
                            f"non-positive QCD integrals in group {group_name} for shift {shift_name} "
                            f"({c_region}: {c_shift}, {d_region}: {d_shift}), using the nominal transfer factor",
                        )
                    else:
                        tf_shift[shift_name] = c_shift / d_shift
                # report shifts that change the transfer factor noticeably
                if tf_nom > 0:
                    large = {
                        name: tf / tf_nom - 1
                        for name, tf in tf_shift.items()
                        if name != nominal and abs(tf / tf_nom - 1) > 0.05
                    }
                    if large:
                        task.logger.info(
                            f"QCD transfer factor in group {group_name} changes by more than 5% for shifts: " +
                            ", ".join(f"{name}: {rel:+.1%}" for name, rel in sorted(large.items())),
                        )

            # helper to insert the B histogram, scaled by the transfer factor, into the target category of the output
            # for backgrounds, each shift of the input is scaled by the transfer factor of that shift; for data, which
            # only has a nominal histogram, the nominal input is scaled by the transfer factor of every mc shift
            def fill_output(h_out: hist.Histogram, h_in: hist.Histogram, is_data: bool) -> None:
                h_b = get_hist(h_in, b_region)
                if h_b is None:
                    return
                h_out = ensure_axis_value(h_out, "category", target_category)
                cat_index = h_out.axes["category"].index(target_category)
                out_view = h_out.view(flow=True)
                b_view = h_b.view(flow=True)
                nominal_index = h_b.axes["shift"].index(nominal)
                # variances are dropped for data templates with the "mc" strategy and for all with "none"
                keep_variance = variance_strategy == "mc_data" or (variance_strategy == "mc" and not is_data)

                shift_names = mc_shift_names if is_data else list(h_b.axes["shift"])
                for shift_name in shift_names:
                    out_index = h_out.axes["shift"].index(shift_name)
                    in_index = nominal_index if is_data else h_b.axes["shift"].index(shift_name)
                    tf_val = tf_shift.get(shift_name, tf_nom)
                    out_view.value[cat_index, out_index] = b_view.value[in_index] * tf_val
                    out_view.variance[cat_index, out_index] = (
                        b_view.variance[in_index] * tf_val**2 if keep_variance else 0.0
                    )

                # nominal input, scaled by the varied transfer factors
                for shift_name, tf_var in zip(tf_shifts, [tf_up, tf_down]):
                    out_index = h_out.axes["shift"].index(shift_name)
                    out_view.value[cat_index, out_index] = b_view.value[nominal_index] * tf_var
                    out_view.variance[cat_index, out_index] = (
                        b_view.variance[nominal_index] * tf_var**2 if keep_variance else 0.0
                    )

            fill_output(cr_hists[data_cr_proc], data_hist, is_data=True)
            for proc, cr_proc in bkg_cr_procs.items():
                fill_output(cr_hists[cr_proc], hists[proc], is_data=False)

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
    # variant with only mc statistics in the control region templates (data template without autoMCStats)
    analysis_inst.x.hist_hooks.qcd_in_situ_mcvar = functools.partial(qcd_in_situ, variance_strategy="mc")
