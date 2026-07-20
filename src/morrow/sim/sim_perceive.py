"""A simulated overhead RGB-D perceiver that renders a `SceneState` from a `World`.

It produces the same mask-and-surface representation the real perceiver will:
a footprint mask, a depth surface, a centroid, and *ambiguous* yaw candidates.
No rigid 6-DoF pose is assumed, because a pouch does not have one.

When the world stages distractor products, `observe` selects the target by
matching the active skill's descriptor (`target_descriptor`) — real selection
logic, not a cheat — and returns that product's scene. Occlusion adds geometric
perception noise (jittered centroid, shrunk footprint) on top.
"""

from __future__ import annotations

import time

import numpy as np

from ..geometry import wrap_angle
from ..scene import SceneState
from .world import World


AMBIGUITY_MARGIN = 0.5  # min score gap to trust the pick; below it, flag not guess


def _match_score(d: dict, descriptor: dict) -> float:
    tgt_size = np.array(descriptor.get("approx_size", [0.0, 0.0]))
    s = 1.0 if d["kind"] == descriptor.get("kind") else 0.0
    return s - 10.0 * float(np.linalg.norm(np.array(d["size"]) - tgt_size))


def select_target_ranked(detections: list[dict], descriptor: dict) -> tuple[int, float]:
    """Return (best index, margin over the runner-up). A small margin means two
    detections match the descriptor about equally — the pick is not trustworthy."""
    scores = [_match_score(d, descriptor) for d in detections]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    margin = (scores[order[0]] - scores[order[1]]) if len(scores) > 1 else 1e9
    return order[0], float(margin)


def select_target(detections: list[dict], descriptor: dict) -> int:
    """Pick the detection best matching the active SKU descriptor (kind + size)."""
    return select_target_ranked(detections, descriptor)[0]


class SimPerceiver:
    def __init__(self, world: World, resolution: int = 80, target_descriptor: dict | None = None):
        self.world = world
        self.res = resolution
        self.target_descriptor = target_descriptor

    def _render(self):
        w = self.world
        x0, x1, y0, y1 = w.workspace
        xs = np.linspace(x0, x1, self.res)
        ys = np.linspace(y0, y1, self.res)
        gx, gy = np.meshgrid(xs, ys)
        p = w.product
        c, s = np.cos(-p.yaw), np.sin(-p.yaw)
        lx = c * (gx - p.cx) - s * (gy - p.cy)
        ly = s * (gx - p.cx) + c * (gy - p.cy)
        mask = (np.abs(lx) <= p.hx) & (np.abs(ly) <= p.hy)
        depth = np.where(mask, w.product_top_z(), w.table_height).astype(np.float64)
        return mask, depth

    def _detect(self, product, is_target: bool) -> dict:
        fp = product.footprint()
        if is_target:
            centroid = self.world.product_centroid()  # dynamic (attached/resting)
        else:
            centroid = np.array([product.cx, product.cy, self.world.table_height + product.height])
        return {"kind": product.kind, "size": [fp[2] - fp[0], fp[3] - fp[1]],
                "footprint": fp, "centroid": centroid, "product": product}

    def observe(self) -> SceneState:
        w = self.world
        mask, depth = self._render()

        detections = [self._detect(w.product, True)] + [self._detect(d, False) for d in w.distractors]
        if self.target_descriptor is not None and w.distractors:
            idx, margin = select_target_ranked(detections, self.target_descriptor)
        else:
            idx, margin = 0, 1e9
        sel = detections[idx]
        prod = sel["product"]
        centroid = sel["centroid"].copy()
        footprint = sel["footprint"].copy()

        if prod.kind == "cylinder":
            yaws = [0.0]  # rotationally symmetric; any grasp yaw is fine
        else:
            yaws = [wrap_angle(prod.yaw), wrap_angle(prod.yaw + np.pi)]  # 180 ambiguity

        occluded = False
        if w.occlusion_prob > 0.0 and w.rng is not None and w.rng.random() < w.occlusion_prob:
            occluded = True
            centroid[:2] += w.rng.uniform(-0.025, 0.025, size=2)  # apparent-centre shift
            cx, cy = (footprint[0] + footprint[2]) / 2, (footprint[1] + footprint[3]) / 2
            footprint = np.array([cx + (footprint[0] - cx) * 0.8, cy + (footprint[1] - cy) * 0.8,
                                  cx + (footprint[2] - cx) * 0.8, cy + (footprint[3] - cy) * 0.8])

        confidence = 0.98
        if w.perception_dropout_prob > 0.0 and w.rng is not None:
            if w.rng.random() < w.perception_dropout_prob:
                confidence = 0.4  # a low-confidence frame; the gate should veto it

        return SceneState(
            product_mask=mask,
            product_depth=depth,
            product_centroid=centroid,
            product_footprint=footprint,
            product_yaw_candidates=yaws,
            carton_frame=w.carton.pose(),
            carton_opening=w.carton.opening(),
            gripper_pose=w.ee_pose.copy(),
            gripper_signal=w.signal(),
            holding=w.attached,
            perception_confidence=confidence,
            timestamp=time.time(),
            uncertainty={"selected_kind": sel["kind"], "n_candidates": len(detections),
                         "occluded": occluded, "selection_margin": round(margin, 4),
                         "ambiguous": bool(margin < AMBIGUITY_MARGIN)},
        )
