# Third-party code

Methods are re-implemented after the official code named below (procedure, defaults); differences are marked
`[Unified]` in the module docstrings. Data: KITTI / KITTI-MOTS / KITTI-STEP, CC BY-NC-SA 3.0 (non-commercial); the
weights and results inherit this condition. This list is informational, not legal advice.

| source | licence (as published by the source) |
| --- | --- |
| hcmr-lab/Seg2Track-SAM2 (used as published, NOT redistributed: the repository has no licence file for its own code; clone it yourself) | none stated |
| facebookresearch/sam2 (checkpoints downloaded with the official script, not redistributed) | Apache-2.0 |
| JonathonLuiten/TrackEval (installed with pip, pinned commit 12c8791) | MIT |
| rlabbe/filterpy Kalman filter equations (re-implemented) | MIT |
| bochinski/iou-tracker (procedure re-implemented) | MIT |
| abewley/sort (procedure re-implemented independently; no code copied) | GPL-3.0 (original) |
| ifzhang/ByteTrack (procedure and defaults re-implemented) | MIT |
| noahcao/OC_SORT (procedure and defaults re-implemented) | MIT |
| ymzis69/HybridSORT (procedure and defaults re-implemented) | MIT |
| NirAharon/BoT-SORT (procedure and defaults re-implemented) | MIT |
| nwojke/deep_sort (procedure re-implemented independently; no code copied) | GPL-3.0 (original) |
| kamkyu94/TrackTrack (procedure and defaults re-implemented) | MIT |
| ifzhang/FairMOT (network heads, loss, association re-implemented) | MIT |
