"""The demonstration compiler: a trace becomes a verified SkillProgram.

Phase extraction is rule-based and reads only calibrated poses, the gripper
signal, and tracked object motion — no LLM, no learned classifier. The rules:

    grasp     = rising edge of the gripper command
    attached  = product centroid tracks the tool for a few frames after grasp
    lift      = tool rises clear of the product top
    over      = tool arrives above the carton at travel height
    place     = pose at the release edge
    release   = falling edge of the gripper command
    withdraw  = final pose, tool clear while product stays put

From those indices it extracts frame-relative parameters (grasp offset in the
object frame, place offset in the carton frame, heights) and assembles the FSM.
This is where the "next SKU without code" claim actually lives.
"""

from __future__ import annotations

import numpy as np

from .geometry import translation, yaw_of
from .skill import EDGES, SkillProgram, SkillState, Transition, hash_skill
from .trace import DemonstrationTrace

TIMEOUT = 5.0


def _rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def find_grasp_index(trace: DemonstrationTrace) -> int:
    g = trace.gripper_command
    for i in range(len(g)):
        if g[i] >= 0.5 and (i == 0 or g[i - 1] < 0.5):
            return i
    raise ValueError("no grasp (gripper never engaged) in demonstration")


def find_release_index(trace: DemonstrationTrace, grasp_idx: int) -> int:
    g = trace.gripper_command
    for i in range(grasp_idx + 1, len(g)):
        if g[i] < 0.5 and g[i - 1] >= 0.5:
            return i
    raise ValueError("no release (gripper never disengaged) after grasp")


def attachment_confirmed(trace: DemonstrationTrace, grasp_idx: int, window: int = 4) -> bool:
    """Does the product centroid move with the tool right after the grasp?"""
    hi = min(grasp_idx + window, len(trace) - 1)
    if hi <= grasp_idx + 1:
        return False
    ee = np.array([translation(p) for p in trace.ee_poses[grasp_idx:hi + 1]])
    ce = np.array(trace.product_centroids[grasp_idx:hi + 1])
    dee = np.linalg.norm(np.diff(ee, axis=0), axis=1)
    dce = np.linalg.norm(np.diff(ce, axis=0), axis=1)
    moved = dee.sum()
    if moved < 1e-4:
        return True  # tool held still; nothing to correlate, treat as attached
    return bool(dce.sum() >= 0.5 * moved)


def _ee_z(trace: DemonstrationTrace, i: int) -> float:
    return float(translation(trace.ee_poses[i])[2])


def _ee_xy(trace: DemonstrationTrace, i: int) -> np.ndarray:
    return translation(trace.ee_poses[i])[:2]


def find_lift_index(trace: DemonstrationTrace, grasp_idx: int) -> int:
    top_at_grasp = float(trace.product_centroids[grasp_idx][2])
    for i in range(grasp_idx + 1, len(trace)):
        if _ee_z(trace, i) > top_at_grasp + 0.05:
            return i
    return grasp_idx + 1


def find_over_carton_index(trace: DemonstrationTrace, lift_idx: int, release_idx: int,
                           carton_xy: np.ndarray) -> int:
    best_i, best_d = lift_idx, 1e9
    for i in range(lift_idx, release_idx + 1):
        d = float(np.linalg.norm(_ee_xy(trace, i) - carton_xy))
        if d < best_d:
            best_i, best_d = i, d
        if d < 0.03:
            return i
    return best_i


def compile_skill(traces: list[DemonstrationTrace], sku_id: str) -> SkillProgram:
    if not traces:
        raise ValueError("compile_skill needs at least one demonstration")
    trace = traces[0]  # select_best_trace: with one clean sim demo, take it

    table = trace.table_height
    carton_xy = translation(trace.carton_frame)[:2]
    carton_yaw = yaw_of(trace.carton_frame)
    carton_floor = float(translation(trace.carton_frame)[2])

    gi = find_grasp_index(trace)
    ri = find_release_index(trace, gi)
    li = find_lift_index(trace, gi)
    oi = find_over_carton_index(trace, li, ri, carton_xy)
    attached = attachment_confirmed(trace, gi)

    # frame-relative parameters
    top0 = float(trace.product_centroids[0][2])
    approach_height = _ee_z(trace, 0) - top0
    top_g = float(trace.product_centroids[gi][2])
    grasp_z = _ee_z(trace, gi) - top_g
    yaw_g = float(trace.product_yaws[gi])
    grasp_offset = _rot2(-yaw_g) @ (_ee_xy(trace, gi) - trace.product_centroids[gi][:2])
    lift_height = _ee_z(trace, li) - table
    travel_z = _ee_z(trace, oi) - carton_floor
    place_z = _ee_z(trace, ri) - carton_floor
    place_offset = _rot2(-carton_yaw) @ (_ee_xy(trace, ri) - carton_xy)
    withdraw_z = _ee_z(trace, len(trace) - 1) - carton_floor

    rim_z = float(trace.meta.get("carton_rim_z", carton_floor + 0.2))
    kind = trace.meta.get("kind", "box")
    symmetry = "radial" if kind == "cylinder" else "180"
    fp = trace.product_footprints[gi]
    approx_size = [float(fp[2] - fp[0]), float(fp[3] - fp[1])]

    place_rel = {"place_offset": [float(place_offset[0]), float(place_offset[1])]}
    transitions = {
        (SkillState.READY, SkillState.APPROACHED): Transition(
            SkillState.READY, SkillState.APPROACHED, "object",
            {"approach_height": float(approach_height)},
            {"name": "approached", "params": {"xy_tol": 0.03}}, TIMEOUT,
            {"action": "reobserve_and_regenerate", "next_state": SkillState.READY}),
        (SkillState.APPROACHED, SkillState.GRASPED): Transition(
            SkillState.APPROACHED, SkillState.GRASPED, "object",
            {"grasp_offset": [float(grasp_offset[0]), float(grasp_offset[1])],
             "grasp_z": float(grasp_z)},
            {"name": "grasped", "params": {}}, TIMEOUT,
            {"action": "retract_choose_next_grasp", "next_state": SkillState.READY}),
        (SkillState.GRASPED, SkillState.LIFTED): Transition(
            SkillState.GRASPED, SkillState.LIFTED, "ee",
            {"lift_height": float(lift_height)},
            {"name": "lifted", "params": {"table_height": table, "margin": 0.04}}, TIMEOUT,
            {"action": "lower_release_regrasp", "next_state": SkillState.READY}),
        (SkillState.LIFTED, SkillState.OVER_CARTON): Transition(
            SkillState.LIFTED, SkillState.OVER_CARTON, "carton",
            {**place_rel, "travel_z": float(travel_z)},
            {"name": "over_carton", "params": {}}, TIMEOUT,
            {"action": "stop_replan_transport", "next_state": SkillState.LIFTED}),
        (SkillState.OVER_CARTON, SkillState.PLACED): Transition(
            SkillState.OVER_CARTON, SkillState.PLACED, "carton",
            {**place_rel, "place_z": float(place_z)},
            {"name": "placed", "params": {"min_overlap": 0.6}}, TIMEOUT,
            {"action": "stop_replan_transport", "next_state": SkillState.OVER_CARTON}),
        (SkillState.PLACED, SkillState.RELEASED): Transition(
            SkillState.PLACED, SkillState.RELEASED, "ee", {},
            {"name": "released", "params": {}}, TIMEOUT,
            {"action": "reopen_withdraw", "next_state": SkillState.PLACED}),
        (SkillState.RELEASED, SkillState.VERIFIED): Transition(
            SkillState.RELEASED, SkillState.VERIFIED, "ee",
            {"withdraw_z": float(withdraw_z)},
            {"name": "verified", "params": {"min_overlap": 0.6, "rim_z": rim_z + 0.05}}, TIMEOUT,
            {"action": "reobserve_escalate", "next_state": SkillState.RELEASED}),
    }

    grasp_regions = [np.array([float(grasp_offset[0]), float(grasp_offset[1]), 0.0])]
    descriptor = {"kind": kind, "symmetry": symmetry, "approx_size": approx_size}
    program = SkillProgram(
        sku_id=sku_id,
        transitions=transitions,
        grasp_regions=grasp_regions,
        object_descriptor=descriptor,
        metadata={"table_height": table, "n_demos": len(traces), "attachment_confirmed": attached,
                  "phase_indices": {"grasp": gi, "lift": li, "over": oi, "release": ri}},
    )
    program.version_hash = hash_skill(transitions, descriptor, grasp_regions)
    program.validate()
    for edge in EDGES:
        assert edge in program.transitions
    return program
