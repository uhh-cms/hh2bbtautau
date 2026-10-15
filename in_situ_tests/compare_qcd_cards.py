# coding: utf-8

"""
Per-bin comparison of the QCD estimate between datacards created with the "qcd" hook (pre-computed template)
and the "qcd_in_situ" hook (control region processes), for all categories found in the old directory.

Run inside the columnflow sandbox (needs uproot and numpy).
"""

import glob
import os
import re

import numpy as np
import uproot


OLD_DIR = (
    "/data/dust/user/alvesand/hh2bbtautau/hbt_store/analysis_hbt/cf.cms.CreateDatacards/23post_v14/calib__default/"
    "sel__default/red__default/prod__default__run3_dnn_moe/hist__default/inf__default_no_shifts/prod28_abcd_v2"
)
NEW_DIR = (
    "/data/dust/user/alvesand/hh2bbtautau/hbt_store/analysis_hbt/cf.cms.CreateDatacards/23post_v14/calib__default/"
    "sel__default/red__default/prod__default__run3_dnn_moe/hist__default/inf__default_qcd_in_situ/prod28_insitu_v2"
)
OLD_HOOKS = "hooks_qcd__"
NEW_HOOKS = "hooks_qcd_in_situ__"


def nominal_keys(directory) -> list[str]:
    # histogram names without systematic variations
    return [k for k in directory.keys(cycle=False) if "__" not in k]


summary = []
for old_path in sorted(glob.glob(os.path.join(OLD_DIR, f"shapes__*__{OLD_HOOKS}*.root"))):
    basename = os.path.basename(old_path)
    cat = re.match(r"^shapes__(.+?)__hooks_", basename).group(1)
    new_path = os.path.join(NEW_DIR, basename.replace(OLD_HOOKS, NEW_HOOKS))
    if not os.path.exists(new_path):
        print(f"{cat}: no in-situ shapes found at {new_path}, skipping")
        continue

    old = uproot.open(old_path)[cat]
    new = uproot.open(new_path)[cat]

    # old: pre-computed template
    qcd_old_key = [k for k in nominal_keys(old) if k.startswith("QCD_") and k.endswith("_datadriven")]
    assert len(qcd_old_key) == 1, f"{cat}: unexpected old qcd keys {qcd_old_key}"
    qcd_old = old[qcd_old_key[0]].values()

    # new: data in the control region minus background copies
    cr_keys = [k for k in nominal_keys(new) if k.endswith("_QCDCR")]
    data_cr_key = [k for k in cr_keys if k.startswith("QCD_data_")]
    assert len(data_cr_key) == 1, f"{cat}: unexpected data cr keys {data_cr_key}"
    bkg_cr_keys = [k for k in cr_keys if k not in data_cr_key]
    data_cr = new[data_cr_key[0]].values()
    bkg_cr = sum(new[k].values() for k in bkg_cr_keys)
    qcd_new = data_cr - bkg_cr

    # total expectation per bin in the in-situ card (all nominal processes, control region copies with their sign)
    total_new = sum(
        new[k].values() * (-1.0 if k in bkg_cr_keys else 1.0)
        for k in nominal_keys(new)
        if k != "data_obs"
    )

    print(f"\n=== {cat} ===")
    print(f"{'bin':>3s} {'old qcd':>10s} {'new qcd':>10s} {'data_B*TF':>10s} {'bkg_B*TF':>10s} {'total new':>10s}")
    for i in range(len(qcd_old)):
        flag = "  <-- negative" if qcd_new[i] < 0 else ""
        print(
            f"{i:3d} {qcd_old[i]:10.4f} {qcd_new[i]:10.4f} {data_cr[i]:10.4f} {bkg_cr[i]:10.4f} "
            f"{total_new[i]:10.4f}{flag}",
        )

    # relation expected if the only difference is the clipping of negative bins in the old hook
    pos = qcd_old > 1e-4
    max_dev = np.abs(qcd_old[pos] - qcd_new[pos]).max() if pos.any() else 0.0
    summary.append((
        cat, qcd_old.sum(), qcd_new.sum(), np.clip(qcd_new, 0, None).sum(), int((qcd_new < 0).sum()), max_dev,
        int((total_new <= 0).sum()),
    ))

print("\n=== summary ===")
print(
    f"{'category':40s} {'old sum':>9s} {'new sum':>9s} {'new pos.':>9s} {'neg bins':>8s} {'max|old-new| (old>0)':>22s} "
    f"{'bins tot<=0':>11s}",
)
for row in summary:
    print(f"{row[0]:40s} {row[1]:9.3f} {row[2]:9.3f} {row[3]:9.3f} {row[4]:8d} {row[5]:22.2e} {row[6]:11d}")
