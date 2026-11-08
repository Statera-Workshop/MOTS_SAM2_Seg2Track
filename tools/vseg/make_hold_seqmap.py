#!/usr/bin/env python3
"""vseg06: KITTI_MOTS_VOC/ImageSets/MOT/hold2_seqmap.txt for the held-out train sequences (~/vseg_logs/holdout.json),
so dump_dets.py / dump_seg_dets.py --split hold2 work like the val split."""
import json
import sys
from pathlib import Path

name = sys.argv[1] if len(sys.argv) > 1 else ""            # vseg07: "b" -> holdout_b.json -> hold2b_seqmap.txt

root = Path("~/Physical_AI/datasets/KITTI_MOTS_VOC/ImageSets/MOT").expanduser()
hold = json.loads(Path(f"~/vseg_logs/holdout{'_' + name if name else ''}.json").expanduser().read_text())["hold"]
n = dict(line.split() for line in (root / "train_seqmap.txt").read_text().split("\n") if line.strip())
(root / f"hold2{name}_seqmap.txt").write_text("\n".join(f"{s} {n[s]}" for s in sorted(hold)) + "\n")
print({s: int(n[s]) for s in sorted(hold)})
