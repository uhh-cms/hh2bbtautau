# coding: utf-8

"""
Checks that the total expected yield per bin of the in-situ datacards stays positive, with the control region copies
entering with their sign (QCD_data_*_QCDCR positive, <bkg>_*_QCDCR negative), for

  - the nominal templates,
  - every systematic variation (Up/Down) found in the shapes file, varying only the processes that carry it,
  - every lnN nuisance of the datacard, applied as a rate factor to the processes that carry it,
  - a combined "all nuisances 1 and 3 sigma in the downward direction" estimate, adding the per-nuisance
    downward changes in quadrature.

Totals are evaluated without signal (background only) and with signal at r = 1.

Run inside the columnflow sandbox (needs uproot and numpy).
"""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict

import numpy as np
import uproot


BASE = (
    "/data/dust/user/alvesand/hh2bbtautau/hbt_store/analysis_hbt/cf.cms.CreateDatacards/23post_v14/calib__default/"
    "sel__default/red__default/prod__default__run3_dnn_moe/hist__default"
)
# directory and hook string of the in-situ datacards to check
IN_SITU_DIR = f"{BASE}/inf__default_qcd_in_situ/prod28_insitu_v2"
IN_SITU_HOOKS = "hooks_qcd_in_situ__flats_kl1_n10_guardedext0p25"

CR_SUFFIX = "_QCDCR"  # placed after the era, e.g. ttbar_2023BPix_QCDCR, ttH_13p6TeV_2023BPix_QCDCR_hbb
DATA_CR_PREFIX = "QCD_data_"
SIGNAL_PREFIXES = ("ggHH_", "qqHH_")


def is_signal(name: str) -> bool:
    return name.startswith(SIGNAL_PREFIXES)


def sign(name: str) -> float:
    # control region background copies enter with -1 (frozen QCD_scale rate parameter)
    return -1.0 if (CR_SUFFIX in name and not name.startswith(DATA_CR_PREFIX)) else 1.0


def read_lnn(datacard_path: str) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Returns lnN nuisances of a datacard as {name: {process: (down_factor, up_factor)}}.
    """
    lines = [l.split() for l in open(datacard_path).read().splitlines() if l.strip()]
    procs = [l[1:] for l in lines if l and l[0] == "process" and not l[1].lstrip("-").isdigit()][0]
    lnn = {}
    for l in lines:
        if len(l) < 3 or l[1] != "lnN":
            continue
        name, effects = l[0], l[2:]
        lnn[name] = {}
        for proc, eff in zip(procs, effects):
            if eff == "-":
                continue
            if "/" in eff:
                d, u = (float(x) for x in eff.split("/"))
            else:
                u = float(eff)
                d = 1.0 / u
            lnn[name][proc] = (d, u)
    return lnn


def check_category(path: str, cat: str, datacard_path: str | None) -> dict:
    d = uproot.open(path)[cat]
    keys = [k for k in d.keys(cycle=False) if k != "data_obs"]
    nominal = {k: d[k].values() for k in keys if "__" not in k}

    # systematic templates: (process, syst, direction) -> values
    syst = defaultdict(dict)
    for k in keys:
        if "__" not in k:
            continue
        proc, rest = k.split("__", 1)
        m = re.match(r"^(.+)(Up|Down)$", rest)
        if not m:
            continue
        syst[(m.group(1), m.group(2))][proc] = d[k].values()

    # lnN nuisances of the datacard, converted to varied templates of the affected processes
    if datacard_path and os.path.exists(datacard_path):
        for name, effects in read_lnn(datacard_path).items():
            for proc, (down, up) in effects.items():
                if proc not in nominal:
                    continue
                syst[(name, "Down")][proc] = nominal[proc] * down
                syst[(name, "Up")][proc] = nominal[proc] * up

    def total(values: dict[str, np.ndarray], with_signal: bool) -> np.ndarray:
        return sum(
            sign(p) * v
            for p, v in values.items()
            if with_signal or not is_signal(p)
        )

    result = {"nominal": {}, "systs": {}, "combined": {}}
    for with_signal in (False, True):
        label = "r=1" if with_signal else "r=0"
        tot_nom = total(nominal, with_signal)
        result["nominal"][label] = tot_nom

        # per systematic: replace the templates of the varied processes
        downs = []  # per nuisance, the largest decrease per bin over the two directions
        per_syst = {}
        by_name = defaultdict(dict)
        for (name, direction), procs in syst.items():
            varied = dict(nominal)
            varied.update(procs)
            tot_var = total(varied, with_signal)
            per_syst[(name, direction)] = tot_var
            by_name[name][direction] = tot_var
        for name, dirs in by_name.items():
            decrease = np.zeros_like(tot_nom)
            for tot_var in dirs.values():
                decrease = np.maximum(decrease, tot_nom - tot_var)
            downs.append(decrease)
        result["systs"][label] = per_syst
        quad = np.sqrt(sum(dd**2 for dd in downs)) if downs else np.zeros_like(tot_nom)
        result["combined"][label] = {"1sigma": tot_nom - quad, "3sigma": tot_nom - 3 * quad}

    result["n_bins"] = len(next(iter(nominal.values())))
    return result


if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(IN_SITU_DIR, f"shapes__*__{IN_SITU_HOOKS}.root")))
    if not paths:
        raise SystemExit(f"no shapes files found in {IN_SITU_DIR}")

    problems = 0
    for path in paths:
        cat = re.match(r"^shapes__(.+?)__hooks_", os.path.basename(path)).group(1)
        datacard_name = os.path.basename(path).replace("shapes__", "datacard__")[:-5] + ".txt"
        datacard_path = os.path.join(os.path.dirname(path), datacard_name)
        res = check_category(path, cat, datacard_path)
        print(f"\n=== {cat} ({res['n_bins']} bins) ===")
        for label in ("r=0", "r=1"):
            tot_nom = res["nominal"][label]
            print(f"  [{label}] nominal total per bin: min = {tot_nom.min():.4f} (bin {tot_nom.argmin()})")
            if (tot_nom <= 0).any():
                problems += 1
                print(f"  [{label}] !! non-positive nominal total in bins {np.where(tot_nom <= 0)[0].tolist()}")

            # systematics: report the ones with the smallest resulting total
            systs = res["systs"][label]
            if systs:
                worst = sorted(systs.items(), key=lambda kv: kv[1].min())[:5]
                print(f"  [{label}] {len(systs)} systematic templates; smallest totals:")
                for (name, direction), tot_var in worst:
                    flag = "  <-- non-positive" if (tot_var <= 0).any() else ""
                    if (tot_var <= 0).any():
                        problems += 1
                    print(f"      {name}{direction:5s}: min = {tot_var.min():.4f} (bin {tot_var.argmin()}){flag}")
            else:
                print(f"  [{label}] no systematic variations found (shapes file or datacard lnN)")

            comb = res["combined"][label]
            for k in ("1sigma", "3sigma"):
                flag = "  <-- non-positive" if (comb[k] <= 0).any() else ""
                print(f"  [{label}] all nuisances {k} down (quadrature): min = {comb[k].min():.4f}{flag}")

    print(f"\n{'OK: all totals positive' if problems == 0 else f'{problems} problem(s) found, see above'}")
