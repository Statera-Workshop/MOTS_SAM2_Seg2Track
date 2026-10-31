# MOTS_SAM2_Seg2Track

**Seg2Track-SAM2** - zero-shot tracking by segmentation (SAM2.1-large prompted with the detections) on
KITTI-MOTS for the 25 detectors of the comparison, evaluated at box level like every tracker of v1.0.

- Paper / code: `hcmr-lab/Seg2Track-SAM2` (arXiv 2509.11772), on SAM2 (`facebookresearch/sam2`).
- This repository contains only the runner `tools/mot/run_seg2track.py` (our detections -> Seg2Track, official
  `kitti_example.json` values, image saving off) and the evaluation `tools/mot/eval_seg2track.py`. Seg2Track and its
  modified SAM2 are used as published and are **not redistributed** (no licence file for the Seg2Track code).

### Results (HOTA, KITTI-MOTS val, car / pedestrian mean)

| detector | **Seg2Track-SAM2** HOTA | DetA | AssA | best v1.0 tracker | difference | SAM2 ms / frame |
|---|---|---|---|---|---|---|
| gfl | **0.608** | 0.533 | 0.695 | hybridsort 0.522 | +0.086 | 690 |
| cascade_rcnn | **0.606** | 0.522 | 0.710 | tracktrack 0.625 | -0.019 | 920 |
| tood | **0.591** | 0.511 | 0.688 | hybridsort 0.506 | +0.085 | 807 |
| retinanet | **0.591** | 0.501 | 0.700 | botsort_reid 0.535 | +0.056 | 842 |
| atss | **0.581** | 0.480 | 0.705 | tracktrack 0.527 | +0.054 | 959 |
| v6 | **0.580** | 0.489 | 0.691 | botsort_reid 0.509 | +0.070 | 864 |
| v11 | **0.579** | 0.502 | 0.671 | hybridsort 0.435 | +0.144 | 705 |
| v10 | **0.573** | 0.500 | 0.660 | botsort_reid 0.455 | +0.118 | 689 |
| rtmdet | **0.563** | 0.485 | 0.657 | hybridsort 0.465 | +0.098 | 773 |
| ppyoloe | **0.558** | 0.463 | 0.676 | tracktrack 0.487 | +0.071 | 926 |
| centernet | **0.558** | 0.476 | 0.657 | hybridsort 0.395 | +0.163 | 593 |
| dynamic_rcnn | **0.556** | 0.441 | 0.714 | tracktrack 0.579 | -0.023 | 1095 |
| v9 | **0.556** | 0.460 | 0.673 | botsort_reid 0.460 | +0.096 | 707 |
| deformable_detr | **0.553** | 0.464 | 0.665 | botsort_reid 0.482 | +0.071 | 766 |
| x2 | **0.550** | 0.456 | 0.669 | hybridsort 0.452 | +0.098 | 819 |
| v8 | **0.549** | 0.472 | 0.647 | botsort_reid 0.443 | +0.107 | 671 |
| v12 | **0.546** | 0.457 | 0.655 | hybridsort 0.424 | +0.122 | 780 |
| v13 | **0.545** | 0.458 | 0.653 | tracktrack 0.372 | +0.173 | 695 |
| fcos | **0.526** | 0.415 | 0.667 | botsort_reid 0.493 | +0.033 | 725 |
| v26 | **0.522** | 0.411 | 0.671 | hybridsort 0.390 | +0.132 | 788 |
| x | **0.521** | 0.415 | 0.670 | botsort_reid 0.478 | +0.043 | 1050 |
| libra_rcnn | **0.512** | 0.401 | 0.675 | tracktrack 0.599 | -0.086 | 1294 |
| v1 | **0.512** | 0.388 | 0.683 | tracktrack 0.374 | +0.138 | 761 |
| faster_rcnn | **0.504** | 0.364 | 0.700 | tracktrack 0.582 | -0.078 | 1376 |
| ssd | **0.355** | 0.278 | 0.515 | botsort 0.288 | +0.068 | 628 |

Findings: best tracker on 21 of the 25 detectors and nearly detector-independent (HOTA 0.51-0.61 for 24 detectors,
against 0.33-0.63 for the best v1.0 tracker) - the v1.0 rule "detector > threshold > tracker" largely stops holding.
Two caveats: about 1 s per frame (offline use: automatic labelling), and the KITTI-MOTS boxes are mask extents, the
same definition as Seg2Track's output boxes, which favours it at box level.

## Usage

```bash
git clone https://github.com/hcmr-lab/Seg2Track-SAM2 && git clone https://github.com/facebookresearch/sam2
(cd sam2 && SAM2_BUILD_CUDA=0 pip install -e . && cd checkpoints && bash download_ckpts.sh)
pip install matplotlib scipy tqdm opencv-python
PYTHONPATH=Seg2Track-SAM2/sam2 python tools/mot/run_seg2track.py --size l --dets cascade_rcnn v6 --s2t Seg2Track-SAM2
python tools/mot/eval_seg2track.py
```

Part of the *Physical AI MOT* comparison of the **Statera Working Group** (v1.1 extension: UCMCTrack, CAMELTrack,
Seg2Track-SAM2, AB3DMOT). The v1.0 comparison (10 trackers x 25 detectors, JDT, E2E) and the v1.1 summary are in
`MOT_Detectors_KITTIMOTS` (`RESULTS.md`).

## Protocol (all MOT_* repositories)

- Online / causal only: every method sees the current frame and its own state; checked by re-running prefixes of
  every sequence (the output of frame t must not change when later frames are removed).
- Input 1216 x 384 (KITTI 1242 x 375 letterboxed), detectors trained from scratch on KITTI-MOTS train (12 sequences),
  30 epochs, Adam 1e-4, effective batch 8, seed 777 - the protocol of the *Physical AI Object Detection* series.
- Metrics: HOTA / DetA / AssA (TrackEval classes, car / pedestrian mean), MOTA, IDF1, IDSW.
- Latency: RTX 4070 Laptop GPU, batch 1, fp32, medians over val; pipeline = detector (forward + post-processing) +
  ReID (appearance trackers) + tracker (CPU); image decoding / letterboxing not included.
- Trackers run with their official default parameters; frame-based values given for 30 fps are scaled to 10 fps.
  A sensitivity table with one common score threshold for every tracker is reported separately (chosen on val, so it
  is analysis, not ranking).

## Data: KITTI-MOTS (box level, val = 9 sequences, 2,981 frames)

KITTI tracking images (21 training sequences) with instance / track annotations; boxes = tight boxes of the masks,
classes car and pedestrian, ignore regions (id 10000) neither rewarded nor penalised. The annotations are taken from
the **KITTI-STEP** panoptic maps (same sequences and split, car / person instances merged from KITTI-MOTS; frames,
track counts and pedestrian masks match the KITTI-MOTS paper, car masks differ by 39 of 26,899).

```bash
bash tools/mot/get_kitti_mots.sh            # KITTI tracking images + annotations (KITTI-STEP fallback)
python tools/mot/kittimots_to_voc.py        # VOC-layout copy with tracks + count report
UNIFIED_IMAGE_DATASET=kittimots python tools/mot/count_loader.py     # counts as the loaders read them
```

**Licence of the data and of every trained weight here: KITTI / KITTI-MOTS / KITTI-STEP are CC BY-NC-SA 3.0
(non-commercial).**

Results of this repository: `results/` (evaluation JSON per detector). Sources and licences: `THIRD_PARTY.md`.

## Global Collaboration

Hills Robotics is currently participating in multiple projects in Korea and internationally.

We are actively interested in building relationships with organizations that can complement our technology with global sales networks, distribution channels, system integration capabilities, manufacturing ecosystems, AI infrastructure, robotics technologies, research capabilities, or strategic investment.

Several technology and business collaborations are already underway. Some partners, projects, system architectures, and commercial programs cannot be publicly disclosed because they are subject to confidentiality and business agreements.

We believe that the emerging Physical AI ecosystem cannot be built by a single company.
It requires a global network connecting robotics companies, AI researchers, semiconductor and computing companies, sensor manufacturers, software developers, universities, system integrators, industrial customers, investors, and global distribution partners.

We have now begun our serious exploration of the Physical AI frontier, and we welcome connections with people and organizations who want to build this ecosystem together.

## Research, Contribution, and Donation

The repositories of the Statera Working Group are intended to contribute to the broader technical discussion around Physical AI.

Researchers, engineers, developers, companies, universities, and independent contributors interested in these subjects are welcome to connect with us.

If you would like to collaborate on Physical AI or autonomous robotics, participate in research or technology development, discuss international business cooperation, propose technology or platform integration, contribute technical knowledge, or support this GitHub knowledge initiative through sponsorship or donation, please contact us.

## Contact

**Richard Minsoo Go**
Head of AI, Hills Robotics
Statera Working Group

Personal / Research Contact:
[gominsoo888@gmail.com](mailto:gominsoo888@gmail.com)

Hills Robotics / Business Contact:
[minsoo.go@hillslogis.com](mailto:minsoo.go@hillslogis.com)

Hills Robotics:
https://hillslogis.com/

For inquiries regarding next-generation outdoor autonomous robotic platforms, Physical AI Spider Grid, Intelligence Cabinet, Edge AI, or On-Premise AI architectures that are not yet publicly presented on the company website, please contact us directly.
