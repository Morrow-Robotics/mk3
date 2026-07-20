"""SkillProgram <-> JSON. The payoff for keeping skills free of callables.

A skill holds only named conditions, frame-relative numbers, and recovery
names, so it round-trips to JSON exactly and the reloaded skill recomputes the
same `version_hash`. That is what makes a skill a versioned, shippable artifact
rather than a live Python object you have to rebuild from a demo every time.
"""

from __future__ import annotations

import json

import numpy as np

from .skill import SkillProgram, SkillState, Transition, hash_skill


def _transition_to_dict(t: Transition) -> dict:
    return {
        "from": t.from_state.value,
        "to": t.to_state.value,
        "frame": t.frame,
        "rel": t.rel,
        "success": t.success,
        "timeout": t.timeout,
        "recovery": {"action": t.recovery["action"], "next_state": t.recovery["next_state"].value},
    }


def _transition_from_dict(d: dict) -> Transition:
    return Transition(
        from_state=SkillState(d["from"]),
        to_state=SkillState(d["to"]),
        frame=d["frame"],
        rel=d["rel"],
        success=d["success"],
        timeout=d["timeout"],
        recovery={"action": d["recovery"]["action"], "next_state": SkillState(d["recovery"]["next_state"])},
    )


def skill_to_dict(skill: SkillProgram) -> dict:
    return {
        "sku_id": skill.sku_id,
        "version_hash": skill.version_hash,
        "object_descriptor": skill.object_descriptor,
        "metadata": skill.metadata,
        "grasp_regions": [[float(v) for v in g] for g in skill.grasp_regions],
        "transitions": {
            f"{a.value}->{b.value}": _transition_to_dict(t)
            for (a, b), t in skill.transitions.items()
        },
    }


def skill_from_dict(d: dict) -> SkillProgram:
    transitions = {}
    for key, td in d["transitions"].items():
        a, b = key.split("->")
        transitions[(SkillState(a), SkillState(b))] = _transition_from_dict(td)
    skill = SkillProgram(
        sku_id=d["sku_id"],
        transitions=transitions,
        grasp_regions=[np.array(g, dtype=float) for g in d["grasp_regions"]],
        object_descriptor=d["object_descriptor"],
        version_hash=d.get("version_hash", ""),
        metadata=d.get("metadata", {}),
    )
    skill.validate()
    return skill


def skill_to_json(skill: SkillProgram, indent: int = 2) -> str:
    return json.dumps(skill_to_dict(skill), indent=indent, sort_keys=True)


def skill_from_json(text: str) -> SkillProgram:
    return skill_from_dict(json.loads(text))


def save_skill(skill: SkillProgram, path: str) -> None:
    with open(path, "w") as f:
        f.write(skill_to_json(skill))


def load_skill(path: str) -> SkillProgram:
    with open(path) as f:
        return skill_from_json(f.read())


def recompute_hash(skill: SkillProgram) -> str:
    """Recompute the content hash from the skill's own fields (for verification)."""
    return hash_skill(skill.transitions, skill.object_descriptor, skill.grasp_regions)
