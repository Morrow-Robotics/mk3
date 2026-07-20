"""Build the physics showcase: render a real MuJoCo pack per SKU, with the FSM
timeline and a physical-placement check, ready for the dashboard to display.

Each entry carries a self-contained base64 mp4 of the parallel-jaw LeRobot arm
doing the task, so the page works served or as a single --shot file.
"""

from __future__ import annotations

import base64
import os
import tempfile

from .film import capture_arm_pack, capture_pack, encode_mp4


def _inside_carton(world) -> bool:
    px, py = world.product_xy()
    c = world.carton
    return bool(c["cx"] - c["hx"] <= px <= c["cx"] + c["hx"]
               and c["cy"] - c["hy"] <= py <= c["cy"] + c["hy"])


def _mp4_b64(frames, fps: int = 18) -> str:
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        encode_mp4(frames, path, fps)
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    finally:
        os.unlink(path)


def _entry(frames, result, world) -> dict:
    return {
        "mp4_b64": _mp4_b64(frames),
        "final_state": result.final_state,
        "success": result.success,
        "first_attempt": result.first_attempt_success,
        "recoveries": result.recoveries,
        "inside_carton": _inside_carton(world),
        "timeline": [{"edge": e.get("edge"), "outcome": e.get("outcome")}
                     for e in result.timeline],
    }


def build_showcase(kinds=("box", "cylinder", "pouch"), seed: int = 0, camera: str = "side") -> dict:
    out = {"kinds": {}}
    for kind in kinds:
        frames, result, world = capture_pack(kind, seed=seed, camera=camera)
        out["kinds"][kind] = _entry(frames, result, world)
    return out


def build_arm_showcase(kinds=("box",), seed: int = 0, camera: str = "cell") -> dict:
    """Render the REAL SO-101 5-DOF arm executing the same compiled skill —
    orientation-aware IK + friction grasp, not a floating mocap gripper."""
    out = {"kinds": {}}
    for kind in kinds:
        frames, result, world = capture_arm_pack(kind, seed=seed, camera=camera)
        out["kinds"][kind] = _entry(frames, result, world)
    return out


def _png_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# Operator-seeded config for the showcased clip (fractional click-boxes + scale).
WATCH_CLIP = dict(clip="videos/pexels_7581335.mp4", frame_idx=18,
                  carton_box_frac=(0.16, 0.30, 0.80, 0.92),
                  product_box_frac=(0.30, 0.40, 0.62, 0.74),
                  scale_m_per_px=0.00024, kind="box", height_m=0.06)


def build_watch_showcase(cfg: dict | None = None) -> dict | None:
    """Watch a REAL clip with SAM2 → overlay proof → SO-101 pack it produced.
    Returns None (panel omitted) when SAM2 weights or the clip are absent."""
    from .watch import capture_watch_pack, have_sam2, render_overlay, segment_scene
    c = cfg or WATCH_CLIP
    if not have_sam2() or not os.path.isfile(c["clip"]):
        return None
    scene = segment_scene(c["clip"], c["frame_idx"], c["carton_box_frac"], c["product_box_frac"])
    fd, opath = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        render_overlay(c["clip"], scene, opath)
        overlay_b64 = _png_b64(opath)
    finally:
        os.unlink(opath)
    frames, ann, size, carton, result, world = capture_watch_pack(
        scene, c["scale_m_per_px"], c["kind"], c["height_m"])
    # honest packing-ACTIVITY profile over the SAM2-detected carton region
    from .pattern import packing_profile, sparkline_png_b64
    W, H = scene.image_wh
    cbb = scene.carton_bbox_px
    profile = packing_profile(c["clip"], (cbb[0] / W, cbb[1] / H, cbb[2] / W, cbb[3] / H))
    return {
        "clip": scene.clip, "frame_idx": scene.frame_idx,
        "overlay_png_b64": overlay_b64, "pack_mp4_b64": _mp4_b64(frames),
        "carton_score": round(scene.carton_score, 3), "product_score": round(scene.product_score, 3),
        "final_state": result.final_state, "success": result.success,
        "inside_carton": _inside_carton(world),
        "product_half_m": [round(v, 3) for v in size], "kind": c["kind"],
        "profile": profile.as_dict(), "sparkline_png_b64": sparkline_png_b64(profile.motion),
    }
