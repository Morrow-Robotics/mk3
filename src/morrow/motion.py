"""Instantiate one FSM edge into concrete base-frame waypoints.

The skill stores motion as frame-relative parameters; this turns them into
world poses against the *current* scene and the *current* tool pose. Because
every edge is re-instantiated from a fresh scene, recovery and retries always
act on where things actually are — never a stale snapshot.
"""

from __future__ import annotations

import numpy as np

from .geometry import Transform, frame, translation, yaw_of
from .scene import SceneState
from .skill import SkillProgram, SkillState


def _rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def grasp_point_xy(skill: SkillProgram, scene: SceneState, params: dict) -> np.ndarray:
    """World xy of the intended grasp, from the demonstrated object-frame offset."""
    rel = skill.transition(SkillState.APPROACHED, SkillState.GRASPED).rel
    nom = np.array(rel["grasp_offset"])
    off = nom + np.array(params.get("grasp_offset_noise", [0.0, 0.0]))
    world_off = _rot2(params["grasp_yaw"]) @ off
    return scene.product_centroid[:2] + world_off


def place_point_xy(skill: SkillProgram, scene: SceneState, params: dict) -> np.ndarray:
    """World xy of the intended place, from the demonstrated carton-frame offset."""
    rel = skill.transition(SkillState.LIFTED, SkillState.OVER_CARTON).rel
    nom = np.array(rel["place_offset"])
    off = nom + np.array(params.get("place_offset_noise", [0.0, 0.0]))
    cf = scene.carton_frame
    world_off = _rot2(yaw_of(cf)) @ off
    return translation(cf)[:2] + world_off


def instantiate_edge(skill: SkillProgram, edge, scene: SceneState, robot,
                     params: dict) -> list[Transform]:
    a, b = edge
    tr = skill.transition(a, b)
    yaw = params["grasp_yaw"]
    table = skill.metadata["table_height"]
    top = scene.product_top_z
    cf = scene.carton_frame
    carton_floor = float(translation(cf)[2])
    ee = translation(robot.get_ee_pose())

    if b is SkillState.APPROACHED:
        xy = grasp_point_xy(skill, scene, params)
        return [frame((xy[0], xy[1], top + tr.rel["approach_height"]), yaw)]

    if b is SkillState.GRASPED:
        xy = grasp_point_xy(skill, scene, params)
        return [frame((xy[0], xy[1], top + tr.rel["grasp_z"]), yaw)]

    if b is SkillState.LIFTED:
        z = table + tr.rel["lift_height"] + params.get("lift_noise", 0.0)
        return [frame((ee[0], ee[1], z), yaw)]

    if b is SkillState.OVER_CARTON:
        xy = place_point_xy(skill, scene, params)
        z = carton_floor + tr.rel["travel_z"] + params.get("travel_noise", 0.0)
        return [frame((xy[0], xy[1], z), yaw)]

    if b is SkillState.PLACED:
        xy = place_point_xy(skill, scene, params)
        z = carton_floor + tr.rel["place_z"] + params.get("place_z_noise", 0.0)
        return [frame((xy[0], xy[1], z), yaw)]

    if b is SkillState.RELEASED:
        return [robot.get_ee_pose()]  # release is actuation, not motion

    if b is SkillState.VERIFIED:
        z = carton_floor + tr.rel["withdraw_z"]
        return [frame((ee[0], ee[1], z), yaw)]

    raise ValueError(f"cannot instantiate edge to {b}")
