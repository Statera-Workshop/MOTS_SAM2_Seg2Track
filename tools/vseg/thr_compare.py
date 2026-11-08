#!/usr/bin/env python3
"""vseg06: Seg2Track val mask HOTA at the official threshold, the train-selected one (vseg04/05) and the hold-out-
selected one (vseg06, detector trained without the held-out sequences)."""
import json
from pathlib import Path

L = Path("~/mot_logs").expanduser()


def val(det, t):
    run = f"{det}_l" if t is None or abs(t - 0.5) < 1e-9 else f"{det}_l_thr{t:g}"
    p = L / "seg2track_mots" / f"{run}.json"
    if not p.exists():
        return "-"
    r = json.loads(p.read_text())
    return f"{r['mean']['HOTA']:.3f} (FP {r['car']['FP'] + r['pedestrian']['FP']})"


def sel(d):
    p = L / d / "selected.json"
    return json.loads(p.read_text()) if p.exists() else {}


tr, ho = sel("seg2track_train_mots"), sel("seg2track_hold_mots")
p = L / "seg2track_holdb_mots" / "selected_cv.json"
cv = json.loads(p.read_text()) if p.exists() else {}
print(f"{'detector':<22}{'official 0.5':>20}{'train thr':>11}{'val':>20}{'hold thr':>10}{'val':>20}{'cv thr':>8}{'val':>20}")
for det in ["cascade_rcnn", "cascade_mask_rcnn_km", "faster_rcnn", "libra_rcnn", "v11", "gfl"]:
    t1 = tr.get(det, {}).get("thr")
    t2 = ho.get(det, {}).get("thr")
    cvt = "-" if det not in cv else f"{cv[det]['thr']:g}"
    print(f"{det:<22}{val(det, None):>20}{('-' if t1 is None else f'{t1:g}'):>11}{(val(det, t1) if t1 else '-'):>20}"
          f"{('-' if t2 is None else f'{t2:g}'):>10}{(val(det, t2) if t2 else '-'):>20}"
          f"{cvt:>8}{(val(det, cv[det]['thr']) if det in cv else '-'):>20}")
