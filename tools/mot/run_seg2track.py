"""Seg2Track-SAM2를 우리 검출 결과(~/mot_logs/dets/kittimots/<det>.npz)로 KITTI-MOTS val에 돌립니다.

- 공식 Seg2Track 코드와 그 안의 수정된 sam2를 그대로 씁니다(PYTHONPATH로 지정, 아래 명령 참고).
- 설정은 공식 configs/kitti_example.json 값 그대로이고, 이미지 저장만 끕니다.
- npz 열 순서(frame, box, score, class)는 값으로 자동 판별하고, 판별이 애매하면 멈춥니다.
- 시퀀스마다 끝나면 done 표시를 남겨, 끊겨도 다시 실행하면 이어서 합니다.
출력: ~/mot_logs/seg2track/<det>_<size>/annotations_mot|annotations_mots/Seg2Track/data/<seq>.txt
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

HOME = os.path.expanduser("~")
VAL = ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]
SIZES = {"t": ("tiny", "t"), "s": ("small", "s"), "b+": ("base_plus", "b+"), "l": ("large", "l")}
OFFICIAL = {"detection_threshold": 0.5, "addition_threshold": {"1": 0.6, "2": 0.85},
            "removal_threshold": 0.1, "removal_tries": 5, "reprompt_threshold": 0.7,
            "removal_bool": False, "reprompt_bool": False, "dataset_type": "KITTI", "detection_type": "KITTI"}

ap = argparse.ArgumentParser()
ap.add_argument("--dets", nargs="+", required=True, help="검출기 이름 (npz 파일 이름)")
ap.add_argument("--size", default="l", choices=list(SIZES))
ap.add_argument("--seqs", nargs="+", default=VAL)
ap.add_argument("--det-thr", type=float, default=None, help="기본: 공식 값 0.5")
ap.add_argument("--mots", default=f"{HOME}/Physical_AI/datasets/KITTI_MOTS")
ap.add_argument("--s2t", default=f"{HOME}/Physical_AI/refs_mot/Seg2Track-SAM2")
ap.add_argument("--ckpt-dir", default=f"{HOME}/Physical_AI/refs_mot/sam2/checkpoints")
ap.add_argument("--out", default=f"{HOME}/mot_logs/seg2track")
a = ap.parse_args()

sys.path.insert(0, os.path.join(a.s2t, "Seg2Track"))
import torch                                   # noqa: E402
import sam2                                    # noqa: E402
from sam2.build_sam import build_sam2_video_predictor   # noqa: E402
from aux_functions import setup_device         # noqa: E402
from Seg2Track import Seg2Track                # noqa: E402

if not os.path.realpath(sam2.__file__).startswith(os.path.realpath(a.s2t)):
    sys.exit(f"STOP: 공식 sam2가 잡혔습니다 ({sam2.__file__}). PYTHONPATH={a.s2t}/sam2 로 실행하세요.")


def frames_jpg(seq):
    """PNG frame -> SAM2/Seg2Track이 읽는 000000.jpg (한 번만 변환, 검출기 간 공유)."""
    d = f"{a.out}/frames/{seq}"
    src = sorted(glob.glob(f"{a.mots}/training/image_02/{seq}/*.png"))
    if len(glob.glob(f"{d}/*.jpg")) != len(src):
        os.makedirs(d, exist_ok=True)
        for p in src:
            stem = os.path.splitext(os.path.basename(p))[0]
            cv2.imwrite(f"{d}/{int(stem):06d}.jpg", cv2.imread(p), [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = cv2.imread(src[0]).shape[:2]
    return d, len(src), h, w


def infer_layout(arr, nframes):
    """npz (N,7) 열 의미를 값으로 판별: frame / score / class / box 4열(xyxy 또는 xywh)."""
    isint = [np.allclose(arr[:, c], np.round(arr[:, c])) for c in range(arr.shape[1])]
    nuniq = [len(np.unique(arr[:, c])) for c in range(arr.shape[1])]
    frame = [c for c in range(7) if isint[c] and arr[:, c].min() >= 0 and arr[:, c].max() < nframes
             and nuniq[c] > 0.3 * nframes]
    score = [c for c in range(7) if not isint[c] and arr[:, c].min() >= 0 and arr[:, c].max() <= 1.0]
    cls = [c for c in range(7) if isint[c] and nuniq[c] <= 3 and c not in frame]
    if len(frame) != 1 or len(score) != 1 or len(cls) != 1:
        raise ValueError(f"열 판별 실패 frame={frame} score={score} cls={cls}")
    box = [c for c in range(7) if c not in frame + score + cls]
    b = arr[:, box]
    xyxy = (b[:, 2] > b[:, 0]).mean() > 0.99 and (b[:, 3] > b[:, 1]).mean() > 0.99 and b[:, 2].max() > 600
    return frame[0], score[0], cls[0], box, xyxy


def to_txt(det, seq, nframes, h, w, layout_log):
    d = np.load(f"{HOME}/mot_logs/dets/kittimots/{det}.npz", allow_pickle=True)
    arr = d[seq].astype(np.float64)
    fc, sc, cc, bc, xyxy = infer_layout(arr, nframes)
    b = arr[:, bc].copy()
    if not xyxy:
        b[:, 2:] += b[:, :2]
    # 일부 검출기는 이미지 밖으로 나간 box를 그대로 냅니다(clip 안 함). SAM2 prompt용으로 이미지 안으로 자릅니다.
    n_out = int(((b[:, 0] < 0) | (b[:, 1] < 0) | (b[:, 2] > w) | (b[:, 3] > h)).sum())
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w - 1)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h - 1)
    ok = (b[:, 2] - b[:, 0] >= 1) & (b[:, 3] - b[:, 1] >= 1)
    arr, b = arr[ok], b[ok]
    # class: 세로로 긴 쪽(h/w 큰 쪽) = Pedestrian(2), 나머지 = Car(1)
    ks = np.unique(arr[:, cc]).astype(int)
    ratio = {k: np.median((b[arr[:, cc] == k, 3] - b[arr[:, cc] == k, 1]) /
                          np.maximum(b[arr[:, cc] == k, 2] - b[arr[:, cc] == k, 0], 1)) for k in ks}
    ped = max(ratio, key=ratio.get)
    cmap = {k: (2 if k == ped else 1) for k in ks}
    layout_log[seq] = dict(frame=fc, score=sc, cls=cc, box=bc, xyxy=bool(xyxy),
                           class_map={int(k): v for k, v in cmap.items()},
                           hw_ratio={int(k): round(float(v), 2) for k, v in ratio.items()},
                           clipped=n_out, dropped=int((~ok).sum()))
    p = f"{a.out}/dets_txt/{det}/{seq}.txt"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for r, bb in zip(arr, b):
            f.write(f"{int(r[fc])} {bb[0]:.2f} {bb[1]:.2f} {bb[2]:.2f} {bb[3]:.2f} {r[sc]:.4f} {cmap[int(r[cc])]}\n")
    return p, str(d["meta"]) if "meta" in d.files else ""


setup_device()
long, short = SIZES[a.size]
predictor = build_sam2_video_predictor(f"configs/sam2.1/sam2.1_hiera_{short}.yaml",
                                       f"{a.ckpt_dir}/sam2.1_hiera_{long}.pt", device="cuda")

def run_det(det):
    if not os.path.exists(f"{HOME}/mot_logs/dets/kittimots/{det}.npz"):
        print(f"SKIP {det}: npz 없음", flush=True)
        return
    tag = f"{det}_{a.size}" + ("" if a.det_thr is None else f"_thr{a.det_thr}")
    params = dict(OFFICIAL, save_images=False, output_tag=tag)
    if a.det_thr is not None:
        params["detection_threshold"] = a.det_thr
    root = f"{a.out}/{tag}"
    os.makedirs(root, exist_ok=True)
    info_p = f"{root}/run_info.json"
    info = json.load(open(info_p)) if os.path.exists(info_p) else {"params": params, "layout": {}, "sec": {}}
    for seq in a.seqs:
        if os.path.exists(f"{root}/done_{seq}"):
            print(f"[{tag}] {seq} 이미 완료", flush=True)
            continue
        fdir, n, h, w = frames_jpg(seq)
        txt, meta = to_txt(det, seq, n, h, w, info["layout"])
        info["meta"] = meta
        print(f"[{tag}] {seq}: {n} frames, layout {info['layout'][seq]}", flush=True)
        t0 = time.time()
        state = predictor.init_state(video_path=fdir, offload_video_to_cpu=True, offload_state_to_cpu=True)
        Seg2Track(txt, predictor, state, fdir, params, a.out, sequence=seq)
        predictor.reset_state(state)
        del state
        torch.cuda.empty_cache()
        info["sec"][seq] = round(time.time() - t0, 1)
        json.dump(info, open(info_p, "w"), indent=1, ensure_ascii=False)
        open(f"{root}/done_{seq}", "w").close()
        print(f"[{tag}] {seq} done in {info['sec'][seq]:.0f}s ({n / info['sec'][seq]:.2f} fps)", flush=True)
    print(f"[{tag}] ALL DONE", flush=True)


for det in a.dets:
    try:
        run_det(det)
    except Exception as e:                       # 한 검출기의 오류가 나머지를 멈추지 않게
        import traceback
        traceback.print_exc()
        print(f"[{det}] FAILED: {e}", flush=True)
