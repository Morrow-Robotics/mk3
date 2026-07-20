"""Success conditions, checked by name so they serialize into a SkillProgram.

Each condition is a `{"name": ..., "params": {...}}` dict living on a
transition. Grasp is verified from *hardware* (`scene.holding`, driven by the
vacuum/force signal) rather than vision, because the gripper occludes the
product at exactly the moment you need to confirm the grasp. Placement and
clearance are verified from vision (footprint overlap with the carton opening).
"""

from __future__ import annotations

import numpy as np

from .scene import SceneState


def _bbox_overlap_frac(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of `inner` bbox area that lies inside `outer` bbox."""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    w = max(0.0, min(ix1, ox1) - max(ix0, ox0))
    h = max(0.0, min(iy1, oy1) - max(iy0, oy0))
    inter = w * h
    area = max(1e-9, (ix1 - ix0) * (iy1 - iy0))
    return float(inter / area)


def _xy_in_bbox(xy: np.ndarray, bbox: np.ndarray) -> bool:
    x0, y0, x1, y1 = bbox
    return bool(x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1)


def check(cond: dict, scene: SceneState, robot) -> bool:
    """Evaluate a serializable success condition against a fresh scene."""
    name = cond["name"]
    p = cond.get("params", {})

    if name == "approached":
        # Tool is aligned over the product (top-down), within xy tolerance.
        dxy = scene.gripper_pose[:2, 3] - scene.product_centroid[:2]
        return float(np.linalg.norm(dxy)) < p.get("xy_tol", 0.03)

    if name == "grasped":
        return scene.holding  # hardware verdict (vacuum/force), never vision

    if name == "lifted":
        margin = p.get("margin", 0.04)
        return scene.product_centroid[2] > p["table_height"] + margin

    if name == "over_carton":
        return _xy_in_bbox(scene.product_centroid, scene.carton_opening)

    if name == "placed":
        overlap = _bbox_overlap_frac(scene.product_footprint, scene.carton_opening)
        return overlap >= p.get("min_overlap", 0.6)

    if name == "released":
        return not scene.holding

    if name == "verified":
        overlap = _bbox_overlap_frac(scene.product_footprint, scene.carton_opening)
        below_rim = scene.product_centroid[2] <= p.get("rim_z", 1e9)
        return (overlap >= p.get("min_overlap", 0.6)) and (not scene.holding) and below_rim

    raise ValueError(f"unknown success condition: {name!r}")
