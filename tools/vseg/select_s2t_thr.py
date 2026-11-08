#!/usr/bin/env python3
"""
vseg04 (2026-11-02): pick the Seg2Track detection threshold of one detector on the TRAIN split (mask HOTA, mean of
car / pedestrian), the way the MOT stage chose UCMCTrack / AB3DMOT thresholds. Val is never looked at.

    python tools/vseg/select_s2t_thr.py --det cascade_rcnn --thrs 0.5 0.6 0.7 0.8

Reads ~/mot_logs/seg2track_train_mots/<det>_l_thr<t>.json (tools/vseg/eval_seg2track_mots.py --split train),
prints the table to stderr, the chosen threshold to stdout (ties -> the lower threshold, i.e. closer to the
official 0.5), and records it in ~/mot_logs/seg2track_train_mots/selected.json.
"""

import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="~/mot_logs/seg2track_train_mots", help="vseg06: ~/mot_logs/seg2track_hold_mots")
ap.add_argument("--det", required=True)
ap.add_argument("--thrs", nargs="+", required=True)
ap.add_argument("--size", default="l")
a = ap.parse_args()
D = Path(a.dir).expanduser()

rows = []
for t in a.thrs:
    p = D / f"{a.det}_{a.size}_thr{t}.json"
    if not p.exists():
        sys.exit(f"STOP: {p} missing")
    r = json.loads(p.read_text())
    tp, fp, fn = (sum(r[c][k] for c in ("car", "pedestrian")) for k in ("TP", "FP", "FN"))
    rows.append((t, r["mean"]["HOTA"], r["mean"]["DetA"], r["mean"]["AssA"], r["mean"]["sMOTSA"], tp, fp, fn))
best = max(rows, key=lambda x: (round(x[1], 4), -float(x[0])))
for t, h, d, s, m, tp, fp, fn in rows:
    print(f"{D.name.split('_')[1]:<5} {a.det:<22} thr {t:<4} HOTA {h:.3f}  DetA {d:.3f}  AssA {s:.3f}  sMOTSA {m:.3f}  "
          f"TP {tp} FP {fp} FN {fn}{'  <- selected' if t == best[0] else ''}", file=sys.stderr)
sel = json.loads((D / "selected.json").read_text()) if (D / "selected.json").exists() else {}
sel[a.det] = {"thr": float(best[0]), "train_HOTA": best[1], "candidates": [float(t) for t in a.thrs]}
(D / "selected.json").write_text(json.dumps(sel, indent=1))
print(best[0])
