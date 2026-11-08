#!/usr/bin/env python3
"""
vseg07 (2026-11-06): Seg2Track threshold from two hold-out folds (each fold's detector never saw its 2 sequences):
mean mask HOTA of fold A (~/mot_logs/seg2track_hold_mots) and fold B (seg2track_holdb_mots); ties -> lower threshold.
Prints the table to stderr, the threshold to stdout, records ~/mot_logs/seg2track_holdb_mots/selected_cv.json.
"""
import argparse
import json
import sys
from pathlib import Path

L = Path("~/mot_logs").expanduser()
ap = argparse.ArgumentParser()
ap.add_argument("--det", required=True)
ap.add_argument("--thrs", nargs="+", required=True)
a = ap.parse_args()
rows = []
for t in a.thrs:
    h = []
    for d in ("seg2track_hold_mots", "seg2track_holdb_mots"):
        p = L / d / f"{a.det}_l_thr{t}.json"
        if not p.exists():
            sys.exit(f"STOP: {p} missing")
        h.append(json.loads(p.read_text())["mean"]["HOTA"])
    rows.append((t, h[0], h[1], (h[0] + h[1]) / 2))
best = max(rows, key=lambda x: (round(x[3], 4), -float(x[0])))
for t, ha, hb, m in rows:
    print(f"cv    {a.det:<22} thr {t:<4} foldA {ha:.3f}  foldB {hb:.3f}  mean {m:.3f}{'  <- selected' if t == best[0] else ''}",
          file=sys.stderr)
p = L / "seg2track_holdb_mots" / "selected_cv.json"
sel = json.loads(p.read_text()) if p.exists() else {}
sel[a.det] = {"thr": float(best[0]), "mean_HOTA": best[3]}
p.write_text(json.dumps(sel, indent=1))
print(best[0])
