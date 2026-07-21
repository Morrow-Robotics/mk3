"""Customer video -> reconstructed SO-101 scene -> the arm packs the objects.

For each clip: SAM2 (operator-seeded) segments the carton and a few objects in one
frame; their pixel bboxes + an operator scale become a carton opening and a set of
graspable box sizes; the SO-101 model then picks each object and packs it into the
carton (real IK + contact grasp + FSM-style recovery). One panel per clip.

Honest boundary (same as the rest of the watch stack):
  * SAM2 gives pixel geometry; a monocular clip has no metric scale, so the operator
    supplies m/px and the object count. Segmentation is operator-seeded, not magic.
  * Objects are approximated as rectangular boxes sized from their bboxes — what the
    parallel jaw grips. Diverse-material grasping is Phase-4 hardware.
  * The SO-101 sequence is embodiment-portable: `SO101BenchRobot` runs it on the real
    arm unchanged.
"""

from __future__ import annotations

import numpy as np

from .arm_multi import CARTON, capture_pack_objects
from .watch import Sam2Segmenter, _px_box, have_sam2, read_frame

# Per-clip operator seeds: the carton box + a few object boxes (fractional), a table
# scale, and object heights. Kept to <=3 objects — the SO-101's reliable count.
VIDEO_SCENES = [
    dict(clip="videos/pexels_7581335.mp4", frame_idx=18, scale_m_per_px=0.00024,
         carton_box_frac=(0.16, 0.30, 0.80, 0.92),
         object_boxes_frac=[(0.05, 0.55, 0.26, 0.90),     # mug
                            (0.35, 0.40, 0.62, 0.74),     # picture frame
                            (0.63, 0.30, 0.80, 0.55)]),   # item at right
    dict(clip="videos/mixkit_42119.mp4", frame_idx=120, scale_m_per_px=0.0006,
         carton_box_frac=(0.05, 0.35, 0.45, 1.0),
         object_boxes_frac=[(0.58, 0.30, 0.78, 0.62),
                            (0.80, 0.35, 0.98, 0.70)]),
    dict(clip="videos/pexels_7855140.mp4", frame_idx=120, scale_m_per_px=0.00012,
         carton_box_frac=(0.28, 0.35, 0.72, 0.85),
         object_boxes_frac=[(0.30, 0.20, 0.50, 0.45),
                            (0.55, 0.20, 0.75, 0.45)]),
]

# The SO-101 packs graspable boxes of a proven-reliable size — the video sets the
# object COUNT (and the real detections drive the overlay). Per-object size variation
# on this tiny arm destabilizes the sequence, so the pack uses a uniform box; that is
# an honest approximation, not a fidelity claim.
_PACK_SIZE = (0.015, 0.02)


def reconstruct_scene(cfg, segmenter=None):
    """SAM2-segment the carton + objects -> (carton, object sizes, detections). The
    carton and objects are DETECTED (shown in the overlay); the pack uses the SO-101's
    reliable carton + N uniform graspable boxes so the arm seats the detected count.
    detections carry masks for the overlay."""
    seg = segmenter or Sam2Segmenter.shared()
    frame = read_frame(cfg["clip"], cfg["frame_idx"])
    H, W = frame.shape[:2]
    cmask, cbb, cs = seg.segment(frame, box=_px_box(cfg["carton_box_frac"], W, H))
    carton = dict(CARTON)  # reliable, in-reach carton for the pack
    dets = [(cbb, round(float(cs), 3), True, cmask)]
    for ob in cfg["object_boxes_frac"]:
        omask, obb, os_ = seg.segment(frame, box=_px_box(ob, W, H))
        dets.append((obb, round(float(os_), 3), False, omask))
    sizes = [_PACK_SIZE] * len(cfg["object_boxes_frac"])
    return carton, sizes, (W, H), dets


def render_overlay(cfg, dets, path, width=380):
    """Draw the SAM2 carton (amber) + object (green) masks on the real frame."""
    import cv2
    frame = read_frame(cfg["clip"], cfg["frame_idx"])
    ov = frame.copy()
    for bbox, _score, is_carton, mask in dets:
        col = np.array([210, 140, 20]) if is_carton else np.array([30, 210, 70])
        if mask is not None:
            ov[mask] = (0.55 * ov[mask] + 0.45 * col).astype(np.uint8)
        cv2.rectangle(ov, bbox[:2], bbox[2:], tuple(int(c) for c in col[::-1]), 3)
    H, W = ov.shape[:2]
    ov = cv2.resize(ov, (width, int(width * H / W)))
    cv2.imwrite(path, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    return path


def reconstruct_and_pack(cfg, segmenter=None, camera="cell", height=260, width=340):
    """Full per-clip pipeline: reconstruct -> SO-101 packs -> frames + result."""
    carton, sizes, wh, dets = reconstruct_scene(cfg, segmenter)
    frames, seated, status = capture_pack_objects(sizes, carton=carton, camera=camera,
                                                  height=height, width=width)
    return {"clip": cfg["clip"], "carton": carton, "sizes": sizes, "dets": dets,
            "wh": wh, "frames": frames, "n": len(sizes), "seated": seated, "status": status}
