"""Deterministic candidate generation, fail-closed gates, then ranking.

For the current edge we generate a modest set of parameter variations, seeded
so the same (skill, scene, edge, round) always yields the same ordering — the
planner is reproducible; randomization lives in the evaluation setup, not here.
The nominal demonstrated parameters are always candidate zero, so the skill is
tried as demonstrated first and variation is only explored on retries.

Gates come BEFORE ranking: reject anything unreachable, off-object, or out of
the carton. These are feasibility checks, not industrial safety.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import bbox_overlap_area, xy_in_bbox
from .motion import grasp_point_xy, instantiate_edge, place_point_xy
from .scene import SceneState
from .skill import STATE_ORDER, SkillProgram, SkillState


@dataclass
class Candidate:
    id: str
    params: dict
    score: float = 0.0


def _seed(seed: int, edge, rnd: int) -> int:
    # Edge key is the from-state's position (unique per edge) — a STABLE hash.
    # (Python salts str/tuple hashing per process, which would break
    # cross-process reproducibility on yaw-dependent scenes.)
    edge_idx = STATE_ORDER.index(edge[0])
    return (int(seed) * 1_000_003 + edge_idx * 131 + rnd * 17) % (2**31 - 1)


def generate_candidates(skill: SkillProgram, scene: SceneState, edge, seed: int,
                        rnd: int = 0, n: int = 20) -> list[Candidate]:
    rng = np.random.RandomState(_seed(seed, edge, rnd))
    yaws = scene.product_yaw_candidates or [0.0]
    out: list[Candidate] = []
    for i in range(n):
        nominal = (i == 0 and rnd == 0)
        gn = np.zeros(2) if nominal else rng.uniform(-0.012, 0.012, size=2)
        pn = np.zeros(2) if nominal else rng.uniform(-0.015, 0.015, size=2)
        params = {
            "grasp_yaw": float(yaws[rng.randint(len(yaws))]),
            "grasp_offset_noise": gn,
            "lift_noise": 0.0 if nominal else float(rng.uniform(-0.01, 0.02)),
            "travel_noise": 0.0 if nominal else float(rng.uniform(-0.01, 0.02)),
            "place_offset_noise": pn,
            "place_z_noise": 0.0 if nominal else float(rng.uniform(-0.005, 0.005)),
        }
        out.append(Candidate(id=f"{seed}:{edge[1].value}:{rnd}:{i}", params=params))
    return out


def apply_feasibility(cands: list[Candidate], skill: SkillProgram, scene: SceneState,
                      robot, edge) -> list[Candidate]:
    if scene.perception_confidence < 0.55:
        return []
    survivors = []
    for c in cands:
        waypoints = instantiate_edge(skill, edge, scene, robot, c.params)
        if not all(robot.reachable(wp) for wp in waypoints):
            continue
        if edge[1] is SkillState.GRASPED:
            gxy = grasp_point_xy(skill, scene, c.params)
            if not xy_in_bbox(gxy, scene.product_footprint):
                continue  # grasp point must land on the object
        if edge[1] in (SkillState.OVER_CARTON, SkillState.PLACED):
            pxy = place_point_xy(skill, scene, c.params)
            if not xy_in_bbox(pxy, scene.carton_opening):
                continue  # place point must land in the carton
            if scene.occupied_regions and _would_collide(pxy, scene.product_footprint,
                                                          scene.occupied_regions):
                continue  # would land on an already-placed item
        survivors.append(c)
    return survivors


RANKER_WEIGHT = 3.0  # how strongly a learned grasp ranker can reorder candidates


def rank_candidates(cands: list[Candidate], skill: SkillProgram, scene: SceneState,
                    edge, ranker=None) -> list[Candidate]:
    ref_yaw = scene.product_yaw_candidates[0] if scene.product_yaw_candidates else 0.0
    for c in cands:
        noise = float(np.linalg.norm(c.params["grasp_offset_noise"]) +
                      np.linalg.norm(c.params["place_offset_noise"]))
        score = -2.5 * noise  # demo_similarity: closest to demonstrated first
        if edge[1] is SkillState.GRASPED:
            gxy = grasp_point_xy(skill, scene, c.params)
            score += -1.0 * float(np.linalg.norm(gxy - scene.product_centroid[:2]))  # grasp_quality
            if ranker is not None:  # opt-in: add the learned seal-probability term
                from .ranker import blend_score
                score += RANKER_WEIGHT * blend_score(
                    ranker, c.params["grasp_yaw"], ref_yaw, c.params["grasp_offset_noise"])
        if edge[1] in (SkillState.OVER_CARTON, SkillState.PLACED):
            cc = scene.carton_frame[:2, 3]
            pxy = place_point_xy(skill, scene, c.params)
            score += -0.5 * float(np.linalg.norm(pxy - cc))  # clearance to carton center
        c.score = score
    return sorted(cands, key=lambda c: c.score, reverse=True)


def _would_collide(place_xy, product_footprint, occupied, clearance: float = 0.005) -> bool:
    """Would the product's footprint, placed at place_xy, overlap an occupied region?"""
    hx = (product_footprint[2] - product_footprint[0]) / 2 + clearance
    hy = (product_footprint[3] - product_footprint[1]) / 2 + clearance
    pred = (place_xy[0] - hx, place_xy[1] - hy, place_xy[0] + hx, place_xy[1] + hy)
    return any(bbox_overlap_area(pred, o) > 0.0 for o in occupied)
