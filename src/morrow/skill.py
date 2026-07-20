"""The verified skill program: a finite state machine, not a list of phases.

Why an FSM and not a phase list: if a candidate fails after the object is
already grasped or already in the carton, the system must never restart from
READY and, say, approach an object that is already in the gripper. Every
recovery is defined *per transition* and always runs from the robot's actual
current state. The states are strictly ordered:

    READY -> APPROACHED -> GRASPED -> LIFTED -> OVER_CARTON -> PLACED
          -> RELEASED -> VERIFIED

Each transition carries a frame-relative motion parameterization (extracted
from the demonstration), a serializable success condition, a timeout, and a
specific recovery. Nothing here holds Python callables or raw poses, so a
SkillProgram serializes cleanly and hashes reproducibly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SkillState(Enum):
    READY = "READY"
    APPROACHED = "APPROACHED"
    GRASPED = "GRASPED"
    LIFTED = "LIFTED"
    OVER_CARTON = "OVER_CARTON"
    PLACED = "PLACED"
    RELEASED = "RELEASED"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


# The one happy path. Terminal states (VERIFIED, FAILED) are not in it.
STATE_ORDER: tuple[SkillState, ...] = (
    SkillState.READY,
    SkillState.APPROACHED,
    SkillState.GRASPED,
    SkillState.LIFTED,
    SkillState.OVER_CARTON,
    SkillState.PLACED,
    SkillState.RELEASED,
    SkillState.VERIFIED,
)

EDGES: tuple[tuple[SkillState, SkillState], ...] = tuple(
    (STATE_ORDER[i], STATE_ORDER[i + 1]) for i in range(len(STATE_ORDER) - 1)
)


def next_state(current: SkillState) -> SkillState | None:
    """The successor of `current` on the happy path, or None if terminal."""
    for i, s in enumerate(STATE_ORDER[:-1]):
        if s is current:
            return STATE_ORDER[i + 1]
    return None


@dataclass
class Transition:
    """One edge of the FSM. `frame` selects the reference the motion is
    expressed in at runtime: "object" (approach/grasp), "carton" (transport/
    place), or "ee" (lift/release/withdraw, relative to current pose)."""

    from_state: SkillState
    to_state: SkillState
    frame: str  # "object" | "carton" | "ee"
    rel: dict  # nominal frame-relative parameters extracted from the demo
    success: dict  # {"name": str, "params": {...}} -> see conditions.py
    timeout: float
    recovery: dict  # {"action": str, "next_state": SkillState}


@dataclass
class SkillProgram:
    sku_id: str
    transitions: dict[tuple[SkillState, SkillState], Transition]
    grasp_regions: list[np.ndarray]  # candidate grasp points in object frame
    object_descriptor: dict  # approx_size, kind (rigid/cylinder/pouch), symmetry
    version_hash: str = ""
    metadata: dict = field(default_factory=dict)

    def transition(self, frm: SkillState, to: SkillState) -> Transition:
        try:
            return self.transitions[(frm, to)]
        except KeyError:
            raise KeyError(f"skill {self.sku_id!r} has no transition {frm.value}->{to.value}")

    def validate(self) -> None:
        """Fail loud if the FSM is not a complete READY..VERIFIED chain."""
        for edge in EDGES:
            if edge not in self.transitions:
                a, b = edge
                raise ValueError(f"skill {self.sku_id!r} missing transition {a.value}->{b.value}")


def _canonical(obj):
    """Recursively make a structure JSON-serializable and order-stable."""
    if isinstance(obj, SkillState):
        return obj.value
    if isinstance(obj, np.ndarray):
        return [round(float(v), 6) for v in obj.reshape(-1)]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 6)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def hash_skill(transitions: dict, descriptor: dict, grasp_regions: list) -> str:
    payload = {
        "transitions": {
            f"{a.value}->{b.value}": _canonical(
                {"frame": t.frame, "rel": t.rel, "success": t.success,
                 "timeout": t.timeout, "recovery": t.recovery}
            )
            for (a, b), t in transitions.items()
        },
        "descriptor": _canonical(descriptor),
        "grasp_regions": _canonical(grasp_regions),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]
