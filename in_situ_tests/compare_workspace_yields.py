# coding: utf-8
"""
Compare the per-process yields printed by the patched combine (block "Per-process yields including all
normalisation factors" in the text2workspace output) with the "rate" line of the corresponding datacards.

Usage:
    python3 compare_workspace_yields.py <text2workspace_log> <datacard_dir>

The log may contain several channels (combined cards, bins named ch1, ch2, ...; the mapping to datacard files
is taken from the "combineCards.py ..." command line in the same log) or a single card (bin named after the
category).
"""
from __future__ import annotations

import glob
import os
import re
import sys


def parse_log(path: str) -> tuple[dict[str, dict[str, float]], list[str]]:
    yields: dict[str, dict[str, float]] = {}
    order: list[str] = []
    current = None
    in_table = False
    for line in open(path):
        m = re.search(r"combineCards\.py\s+(.*?)\s*>", line)
        if m:
            order = [os.path.basename(t) for t in m.group(1).split() if t.endswith(".txt")]
        m = re.search(r"Analysing bin errors for: prop_bin(\S+)", line)
        if m:
            current = m.group(1)
            yields[current] = {}
            in_table = False
            continue
        if current and "Per-process yields" in line:
            in_table = True
            continue
        if in_table:
            if line.startswith("=") or line.strip().startswith("Process"):
                if line.startswith("="):
                    in_table = False
                continue
            parts = line.split()
            if len(parts) == 3:
                yields[current][parts[0]] = float(parts[2])
    return yields, order


def parse_card(path: str) -> dict[str, float]:
    lines = open(path).read().splitlines()
    procs = [l.split()[1:] for l in lines if l.startswith("process") and not l.split()[1].lstrip("-").isdigit()][0]
    rates = [float(x) for x in [l for l in lines if l.startswith("rate")][0].split()[1:]]
    return dict(zip(procs, rates))


def main(log: str, card_dir: str) -> None:
    yields, order = parse_log(log)
    if not order:
        # the CombineDatacards task was not rerun, so no combineCards.py line is in the log; the inference tools
        # sort the resolved datacard paths, so chN follows the alphabetical order of the files
        order = sorted(os.path.basename(f) for f in glob.glob(os.path.join(card_dir, "datacard__*.txt")))
    for ch, ws in yields.items():
        if re.fullmatch(r"ch\d+", ch) and order:
            card_file = os.path.join(card_dir, order[int(ch[2:]) - 1])
        else:
            matches = glob.glob(os.path.join(card_dir, f"datacard__{ch}__*.txt"))
            card_file = matches[0] if matches else None
        if not card_file or not os.path.exists(card_file):
            print(f"\n=== {ch}: no datacard found in {card_dir}")
            continue
        card = parse_card(card_file)
        print(f"\n=== {ch}  ({os.path.basename(card_file)})")
        print(f"{'process':50s} {'card rate':>12s} {'workspace':>12s} {'diff':>10s}")
        for proc, w in ws.items():
            c = card.get(proc)
            if c is None:
                if abs(w) > 1e-6:
                    print(f"{proc:50s} {'(absent)':>12s} {w:12.5f}")
                continue
            if abs(abs(w) - c) > 1e-4:
                print(f"{proc:50s} {c:12.5f} {abs(w):12.5f} {abs(w) - c:+10.5f}")
        cr = [p for p in card if "_QCDCR" in p]
        data = [p for p in cr if p.startswith("QCD_data")]
        if data:
            card_net = card[data[0]] - sum(card[p] for p in cr if p not in data)
            ws_net = sum(ws.get(p, 0.0) for p in cr)
            print(f"{'net QCD (data copy - bkg copies)':50s} {card_net:12.4f} {ws_net:12.4f} {ws_net - card_net:+10.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
