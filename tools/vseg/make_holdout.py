#!/usr/bin/env python3
"""
vseg05 (2026-11-03): hold two KITTI-MOTS TRAIN sequences out to choose the fine-tuning length of the *_km models
without looking at val (YOLO11-seg peaked at epoch 10 of 20 on val, see the vseg tab).

Rule, fixed before any run: among all pairs of the 12 train sequences whose frames are 10-20 % of the train frames,
take the pair whose pedestrian share of (non-ignore) objects is closest to the whole train split; ties -> fewer
frames. Writes KITTI_MOTS_VOC/ImageSets/Main/train10.txt and hold2.txt (ids "<seq>_<frame>", round-robin order of
trainval kept) and ~/vseg_logs/holdout.json. Used by train_seg_<v>_hold.py (UNIFIED_SEG_TRAIN_SPLIT / _TEST_SPLIT).
"""

import argparse
import itertools
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="", help="vseg07: suffix of a second fold, e.g. b -> train10b / hold2b / holdout_b.json")
ap.add_argument("--exclude", nargs="*", default=[], help="sequences that may not be held out (the first fold)")
args = ap.parse_args()
ROOT = Path("~/Physical_AI/datasets/KITTI_MOTS_VOC").expanduser()
main_dir = ROOT / "ImageSets" / "Main"
ids = [x for x in (main_dir / "trainval.txt").read_text().split() if x]
frames, objs = Counter(), defaultdict(Counter)
for i in ids:
    seq = i.split("_")[0]
    frames[seq] += 1
    for o in ET.parse(ROOT / "Annotations" / f"{i}.xml").getroot().iter("object"):
        if o.find("difficult").text.strip() == "0":
            objs[seq][o.find("name").text.strip().lower()] += 1
total = sum(frames.values())
ped_all = sum(c["pedestrian"] for c in objs.values()) / max(1, sum(sum(c.values()) for c in objs.values()))
best = None
for a, b in itertools.combinations(sorted(s for s in frames if s not in args.exclude), 2):
    f = (frames[a] + frames[b]) / total
    if not 0.10 <= f <= 0.20:
        continue
    n = objs[a] + objs[b]
    if not (n["car"] and n["pedestrian"]):
        continue
    ped = n["pedestrian"] / sum(n.values())
    key = (abs(ped - ped_all), f)
    if best is None or key < best[0]:
        best = (key, (a, b), f, ped)
(_, hold, f, ped) = best
(main_dir / f"hold2{args.name}.txt").write_text("\n".join(i for i in ids if i.split("_")[0] in hold) + "\n")
(main_dir / f"train10{args.name}.txt").write_text("\n".join(i for i in ids if i.split("_")[0] not in hold) + "\n")
info = {"hold": list(hold), "hold_frames": int(f * total), "hold_fraction": round(f, 3),
        "hold_ped_share": round(ped, 3), "train_ped_share": round(ped_all, 3), "train_frames": total}
out = Path(f"~/vseg_logs/holdout{'_' + args.name if args.name else ''}.json").expanduser()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(info, indent=1))
print(info)
