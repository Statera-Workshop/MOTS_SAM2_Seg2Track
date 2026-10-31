#!/usr/bin/env python3
"""
MOT v1.1 (2026-10-30): evaluate Seg2Track-SAM2 runs (mot14_day1/run_seg2track.py) with the framework's evaluation.

    python tools/mot/eval_seg2track.py                      # every finished run under ~/mot_logs/seg2track
    python tools/mot/eval_seg2track.py --runs cascade_rcnn_l

Input : ~/mot_logs/seg2track/<det>_<size>/annotations_mot/Seg2Track/data/<seq>.txt (KITTI tracking lines: frame id
        class -1 -1 -1 x1 y1 x2 y2 ... score; boxes = mask extent) and run_info.json (seconds per sequence).
Output: ~/mot_logs/track/kittimots_<det>/seg2track.json, same schema as run_tracker.py (box HOTA / MOTA / IDF1 with
        common/mot/evaluate.py, so the numbers sit in the same table as the other trackers), summary.txt line.
Tracker time = Seg2Track wall time per frame (SAM2 + Seg2Track logic, GPU), detector time from the detections npz.
Masks are not evaluated here (box level, like every tracker of v1.0).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config as C  # noqa: E402
from common.mot.evaluate import KITTI_CLASSES, evaluate, load_kittimots  # noqa: E402

CLS = {"Car": 0, "Pedestrian": 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="~/mot_logs/seg2track")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--dets", default="~/mot_logs/dets/kittimots")
    ap.add_argument("--out", default="~/mot_logs/track")
    a = ap.parse_args()
    root = Path(a.root).expanduser()
    gt, _ = load_kittimots(C.DATASETS_ROOT / "KITTI_MOTS_VOC", "val")
    runs = a.runs or sorted(p.name for p in root.iterdir() if (p / "run_info.json").exists())
    for run in runs:
        rd = root / run
        if not all((rd / f"done_{s}").exists() for s in gt):
            print(f"{run:<24} skipped: not all {len(gt)} sequences finished")
            continue
        info = json.loads((rd / "run_info.json").read_text())
        det, size = run.rsplit("_", 1)[0], run.rsplit("_", 1)[1]
        if "_thr" in run:
            det = run.split("_l_thr")[0]
        pred = {}
        for seq, frames in gt.items():
            per = [[] for _ in frames]
            for line in (rd / "annotations_mot" / "Seg2Track" / "data" / f"{seq}.txt").read_text().splitlines():
                p = line.split()
                if not p or p[2] not in CLS:
                    continue
                f = int(p[0])
                if f < len(per):
                    per[f].append([float(p[6]), float(p[7]), float(p[8]), float(p[9]), int(p[1]), CLS[p[2]],
                                   float(p[17])])
            pred[seq] = [np.array(x, np.float64).reshape(-1, 7) for x in per]
        res = evaluate(gt, pred, KITTI_CLASSES)
        n_frames = sum(len(v) for v in gt.values())
        ms = 1000.0 * sum(info["sec"][s] for s in gt) / n_frames
        det_ms = None
        dp = Path(a.dets).expanduser() / f"{det}.npz"
        if dp.exists():
            z = np.load(dp, allow_pickle=False)
            d = np.concatenate([z[f"{s}__ms"] for s in gt])
            det_ms = {"median": float(np.median(d)), "p95": float(np.percentile(d, 95))}
        rec = {"tracker": "seg2track", "run": f"kittimots_{det}", "source": "kittimots", "split": "val",
               "detections": {"version": det}, "sam2_size": size, "seg2track_params": info.get("params"),
               "npz_layout": info.get("layout"),
               "tracker_ms": {"median": ms, "p95": ms, "frames": n_frames, "note": "mean wall time per frame (GPU)"},
               "detector_ms": det_ms, "reid_ms": None, "score_thr": None, "causal_check": None, **res}
        out = Path(a.out).expanduser() / f"kittimots_{det}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "seg2track.json").write_text(json.dumps(rec, indent=1))
        m = res["mean"]
        line = (f"{'seg2track':<12} HOTA {m['HOTA']:.4f}  DetA {m['DetA']:.4f}  AssA {m['AssA']:.4f}  "
                f"MOTA {m['MOTA']:.4f}  IDF1 {m['IDF1']:.4f}  IDSW {m['IDSW']:>5}  trk {ms:.1f} ms (SAM2.1-{size})"
                + "  | " + "  ".join(f"{k} HOTA {v['HOTA']:.3f}" for k, v in res["per_class"].items()))
        with open(out / "summary.txt", "a") as fh:
            fh.write(f"kittimots_{det}  ({time.strftime('%F %T')})\n{line}\n")
        print(f"{det:<16} {line}", flush=True)


if __name__ == "__main__":
    main()
