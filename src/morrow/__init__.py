"""morrow (mk3) — demonstration -> verified skill program for robotic packing.

Public surface, smallest useful set. The pipeline is:

    record demo  ->  compile_skill  ->  run_skill (verified FSM)  ->  metrics

Hardware and perception are boundaries (`Robot`, `Perceiver`); a simulation
backend implements them today, a physical bench later. Nothing above the
boundary knows which it is talking to.
"""

from .compile import compile_skill
from .execute import RunResult, run_skill
from .perceive import Perceiver
from .robot import Robot
from .scene import SceneState
from .serialize import load_skill, save_skill, skill_from_json, skill_to_json
from .skill import EDGES, STATE_ORDER, SkillProgram, SkillState
from .trace import DemonstrationTrace

__all__ = [
    "DemonstrationTrace",
    "SceneState",
    "SkillProgram",
    "SkillState",
    "STATE_ORDER",
    "EDGES",
    "compile_skill",
    "run_skill",
    "RunResult",
    "Robot",
    "Perceiver",
    "skill_to_json",
    "skill_from_json",
    "save_skill",
    "load_skill",
]
