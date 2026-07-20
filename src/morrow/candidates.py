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

from dataclasses import dataclass, field

import numpy as np

from .motion import grasp_point_xy, instantiate_edge, place_point_xy
from .scene import SceneState
from .skill import SkillProgram, SkillState


@dataclass
class Candidate:
    id: str
    params: dict
    score: float = 0.0
    meta: dict = field(default_factory=dict)


def _seed(seed: int, edge, rnd: int) -> int:
    a, b = edge
    return (int(seed) * 1_000_003 + hash((a.value, b.value)) % 9973 * 131 + rnd * 17) % (2**31 - 1)


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
            if not _in_bbox(gxy, scene.product_footprint):
                continue  # grasp point must land on the object
        if edge[1] in (SkillState.OVER_CARTON, SkillState.PLACED):
            pxy = place_point_xy(skill, scene, c.params)
            if not _in_bbox(pxy, scene.carton_opening):
                continue  # place point must land in the carton
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


def _in_bbox(xy, bbox) -> bool:
    x0, y0, x1, y1 = bbox
    return bool(x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1)
