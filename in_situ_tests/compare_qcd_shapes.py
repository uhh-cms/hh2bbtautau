# coding: utf-8

"""
Per-bin shape comparison of the QCD estimate in the signal region between

  (a) the in-situ datacards: QCD = QCD_data_<era>_QCDCR - sum(<bkg>_<era>_QCDCR), with the bin uncertainty from the
      summed variances of all control region templates,
  (b) the template datacards with the hooks in the order flats -> qcd (QCD_<era>_datadriven),
  (c) optionally, the template datacards with the old order qcd -> flats (disabled by default).

For every category a figure with absolute yields and the ratio to the in-situ estimate is written, as well as a text
table with the per-bin values (absolute and normalized to unit integral).

Run inside the columnflow sandbox (needs uproot, numpy, matplotlib).
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
except ImportError:  # plain matplotlib fallback
    hep = None


BASE = (
    "/data/dust/user/alvesand/hh2bbtautau/hbt_store/analysis_hbt/cf.cms.CreateDatacards/23post_v14/calib__default/"
    "sel__default/red__default/prod__default__run3_dnn_moe/hist__default"
)
FLATS = "flats_kl1_n10_guardedext0p25"

# variants: label -> (directory, hook string in the file name, categorical color)
VARIANTS = {
    "in situ": (f"{BASE}/inf__default_qcd_in_situ/prod28_insitu_v2", f"hooks_qcd_in_situ__{FLATS}", "#3f90da"),
    "flats -> qcd": (f"{BASE}/inf__default_no_shifts/prod28_abcd_v2", f"hooks_{FLATS}__qcd", "#e76300"),
    # old hook order, kept for reference
    # "qcd -> flats (old)": (f"{BASE}/inf__default_no_shifts/prod28_abcd_v2", f"hooks_qcd__{FLATS}", "#832db6"),
}
REFERENCE = "in situ"
OUT_DIR = "qcd_shape_comparison"

# relative statistical uncertainty of the transfer factor per category, sigma_TF / TF, taken from the hook logs
# (in the in-situ datacard this is a separate, fully correlated shape nuisance and therefore not part of the per-bin
# variances; it is added here as a second band so that both estimates show the same uncertainty content)
TF_REL_UNC = {
    "cat_2023BPix_etau_res1b": 0.0111 / 0.0786,
    "cat_2023BPix_etau_res2b": 0.0328 / 0.0288,
    "cat_2023BPix_mutau_res1b": 0.0081 / 0.0717,
    "cat_2023BPix_mutau_res2b": 0.0262 / 0.0774,
    "cat_2023BPix_tautau_res1b": 0.0079 / 0.0848,
    "cat_2023BPix_tautau_res2b": 0.0386 / 0.0992,
}

# integrated luminosity in fb^-1 and year label per era (from the config)
ERAS = {
    "2023BPix": (9.7, ""),
}
CHANNELS = {"etau": r"$e\tau_h$", "mutau": r"$\mu\tau_h$", "tautau": r"$\tau_h\tau_h$"}
PHASESPACES = {"res1b": "res1b", "res2b": "res2b", "boosted": "boosted"}
SERIES_LABELS = {
    "in situ": "in-situ ABCD",
    "flats -> qcd": "a-priori ABCD",
    "qcd -> flats (old)": "ABCD template, old hook order",
}


def pretty_category(cat: str) -> tuple[str, str]:
    # returns (era, description) for a category name like cat_2023BPix_tautau_res2b
    m = re.match(r"^cat_(\w+?)_(etau|mutau|tautau)_(\w+)$", cat)
    if not m:
        return "", cat
    era, channel, phasespace = m.groups()
    return era, f"{CHANNELS.get(channel, channel)}, {PHASESPACES.get(phasespace, phasespace)}"


def nominal_keys(directory) -> list[str]:
    return [k for k in directory.keys(cycle=False) if "__" not in k]


def read_qcd(variant: str, cat: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Returns (values, errors, edges) of the QCD estimate of *variant* in category *cat*, or None if missing.
    """
    directory, hooks, _ = VARIANTS[variant]
    path = os.path.join(directory, f"shapes__{cat}__{hooks}.root")
    if not os.path.exists(path):
        return None
    d = uproot.open(path)[cat]

    if variant == REFERENCE:
        cr_keys = [k for k in nominal_keys(d) if k.endswith("_QCDCR")]
        data_key = [k for k in cr_keys if k.startswith("QCD_data_")][0]
        bkg_keys = [k for k in cr_keys if k != data_key]
        values = d[data_key].values() - sum(d[k].values() for k in bkg_keys)
        variances = d[data_key].variances() + sum(d[k].variances() for k in bkg_keys)
        edges = d[data_key].axis().edges()
    else:
        key = [k for k in nominal_keys(d) if k.startswith("QCD_") and k.endswith("_datadriven")][0]
        values = d[key].values()
        variances = d[key].variances()
        edges = d[key].axis().edges()

    return values, variances**0.5, edges


def plot(cat: str, results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    ref_values, ref_errors, edges = results[REFERENCE]
    n = len(ref_values)
    x = np.arange(n + 1)  # equal-width bins in index space, edges shown as tick labels
    centers = 0.5 * (x[:-1] + x[1:])
    era, description = pretty_category(cat)

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(10, 11.5), sharex=True, gridspec_kw={"height_ratios": [4.5, 1.2], "hspace": 0.06},
    )

    # draw the reference last and thicker so that variants coinciding with it (e.g. flats -> qcd in categories
    # without negative bins) remain visible as a dashed line on top
    ordered = [l for l in results if l != REFERENCE] + [REFERENCE]
    for label in ordered:
        values, errors, _ = results[label]
        color = VARIANTS[label][2]
        is_ref = label == REFERENCE
        ax.stairs(
            values, x, color=color, label=SERIES_LABELS.get(label, label),
            linewidth=3 if is_ref else 2, linestyle="-" if is_ref else "--", zorder=2 if is_ref else 3,
        )
        # statistical uncertainty band: for the reference only the B region statistics, for the template the full
        # per-bin variance of the hook (which already includes the transfer factor term)
        lo = np.r_[values - errors, values[-1] - errors[-1]]
        hi = np.r_[values + errors, values[-1] + errors[-1]]
        ax.fill_between(
            x, lo, hi, step="post", color="#1f5fa8" if is_ref else color, alpha=0.3 if is_ref else 0.25,
            linewidth=0, label="stat. unc. (non-iso region)" if is_ref else "stat. unc.",
        )
        if is_ref and cat in TF_REL_UNC:
            # transfer factor uncertainty on top, in quadrature with the B region statistics
            tot = np.sqrt(errors**2 + (TF_REL_UNC[cat] * values)**2)
            lo_tot = np.r_[values - tot, values[-1] - tot[-1]]
            hi_tot = np.r_[values + tot, values[-1] + tot[-1]]
            ax.fill_between(
                x, lo_tot, hi_tot, step="post", facecolor="none", edgecolor="#8fc1ec", hatch="////",
                linewidth=0, alpha=0.9, label=rf"$\oplus$ TF unc. ({TF_REL_UNC[cat] * 100:.0f}%)",
            )
        if label != REFERENCE:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = values / ref_values
                ratio_err = errors / np.abs(ref_values)
            rax.errorbar(
                centers, ratio, yerr=ratio_err, fmt="o", color="black", ecolor="black", markerfacecolor="black",
                markeredgecolor="black", markersize=6, capsize=3, linewidth=1.5,
            )

    # y axis: log scale when all values are positive, symmetric log otherwise (negative bins)
    all_values = np.concatenate([v for v, _, _ in results.values()])
    if (all_values > 0).all():
        ax.set_yscale("log")
        ax.set_ylim(max(all_values.min() * 0.3, 1e-3), all_values.max() * 100)
    else:
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(-max(abs(all_values.min()) * 3, 1.0), all_values.max() * 100)
        ax.axhline(0, color="#6b6b6b", linewidth=1)
    ax.set_ylabel("Entries")
    ax.text(
        0.04, 0.95, description, transform=ax.transAxes, ha="left", va="top", fontsize=20,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=16)

    rax.axhline(1, color="#000000", linewidth=1)
    rax.set_ylabel("ABCD / in situ", fontsize=16, loc="center")
    rax.set_ylim(-0.5, 2.5)
    rax.set_yticks([0, 1, 2])
    rax.set_xticks(x)
    rax.set_xticklabels([f"{e:.4g}" for e in edges], rotation=45, ha="right", fontsize=18)
    rax.set_xlabel("HH output node", labelpad=22)
    rax.set_xlim(x[0], x[-1])

    if hep is not None:
        lumi, year = ERAS.get(era, (None, ""))
        hep.cms.label(ax=ax, llabel="Private Work", data=True, lumi=lumi, year=year or None, com=13.6, fontsize=20)
    else:
        ax.set_title(f"CMS Private Work, {era}", loc="left")

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"qcd_shape_{cat}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_table(cat: str, results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], f) -> None:
    labels = list(results)
    edges = results[REFERENCE][2]
    f.write(f"\n=== {cat} ===\n")
    f.write("absolute yields (value +- stat)\n")
    f.write(f"{'bin':>3s} {'low':>8s} {'high':>8s} " + " ".join(f"{l:>22s}" for l in labels) + "\n")
    for i in range(len(edges) - 1):
        row = " ".join(f"{results[l][0][i]:11.4f} +- {results[l][1][i]:8.4f}" for l in labels)
        f.write(f"{i:3d} {edges[i]:8.4g} {edges[i + 1]:8.4g} {row}\n")
    f.write("integrals: " + ", ".join(f"{l}: {results[l][0].sum():.3f}" for l in labels) + "\n")
    f.write("normalized shapes (bin fraction of the integral)\n")
    f.write(f"{'bin':>3s} " + " ".join(f"{l:>20s}" for l in labels) + "\n")
    for i in range(len(edges) - 1):
        row = " ".join(f"{results[l][0][i] / results[l][0].sum():20.4f}" for l in labels)
        f.write(f"{i:3d} {row}\n")


if __name__ == "__main__":
    ref_dir, ref_hooks, _ = VARIANTS[REFERENCE]
    cats = sorted(
        re.match(r"^shapes__(.+?)__hooks_", os.path.basename(p)).group(1)
        for p in glob.glob(os.path.join(ref_dir, f"shapes__*__{ref_hooks}.root"))
    )
    # start from a clean output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(OUT_DIR, "qcd_shape_*.*")):
        os.remove(old)
    with open(os.path.join(OUT_DIR, "qcd_shapes.txt"), "w") as f:
        for cat in cats:
            results = {}
            for variant in VARIANTS:
                r = read_qcd(variant, cat)
                if r is None:
                    print(f"{cat}: no shapes for variant '{variant}', skipping it")
                    continue
                results[variant] = r
            if REFERENCE not in results:
                continue
            plot(cat, results)
            write_table(cat, results, f)
            print(f"{cat}: " + ", ".join(f"{l} = {v[0].sum():.3f}" for l, v in results.items()))
    print(f"\nfigures and table written to {OUT_DIR}/")
