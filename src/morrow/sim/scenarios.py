"""Build tabletop worlds and onboard skills from them.

Three product classes: a rigid box and a cylinder (the primary success cases,
per the frozen scope) and a flexible pouch (the stress case, with a smaller
reliable seal region so grasp offset actually matters). Randomization moves the
product and carton so the skill has to transfer, not replay.
"""

from __future__ import annotations

import numpy as np

from ..compile import compile_skill
from .record import record_demo
from .sim_perceive import SimPerceiver
from .sim_robot import SimRobot
from .world import Carton, Product, World


def _carton() -> Carton:
    return Carton(cx=0.20, cy=0.0, yaw=0.0, hx=0.12, hy=0.12, wall_z=0.12, floor_z=0.0)


def make_product(kind: str, cx: float, cy: float, yaw: float) -> Product:
    if kind == "box":
        return Product("box", 0.035, 0.035, 0.05, grasp_radius=0.03, cx=cx, cy=cy, yaw=yaw)
    if kind == "cylinder":
        return Product("cylinder", 0.03, 0.03, 0.08, grasp_radius=0.028, cx=cx, cy=cy, yaw=yaw)
    if kind == "pouch":
        return Product("pouch", 0.055, 0.03, 0.02, grasp_radius=0.02, cx=cx, cy=cy, yaw=yaw)
    raise ValueError(f"unknown product kind: {kind!r}")


def make_world(kind: str, cx: float = -0.15, cy: float = 0.0, yaw: float = 0.0,
               force_fail_seals: int = 0, slip_prob: float = 0.0, rng=None) -> World:
    return World(make_product(kind, cx, cy, yaw), _carton(),
                 force_fail_seals=force_fail_seals, slip_prob=slip_prob, rng=rng)


def forced_failure_world(kind: str = "box") -> World:
    """A world whose first full round of grasps misses, to show recovery live."""
    return make_world(kind, force_fail_seals=7)


def randomize(kind: str, rng: np.random.RandomState, jitter_carton: bool = True,
              slip_prob: float = 0.12) -> World:
    cx = rng.uniform(-0.25, -0.05)
    cy = rng.uniform(-0.15, 0.15)
    yaw = rng.uniform(-np.pi, np.pi)
    world = make_world(kind, cx, cy, yaw, slip_prob=slip_prob, rng=rng)
    if jitter_carton:
        world.carton.cx = 0.20 + rng.uniform(-0.04, 0.04)
        world.carton.cy = 0.0 + rng.uniform(-0.04, 0.04)
    return world


def onboard(kind: str, sku_id: str, n_demos: int = 2):
    """Record demos on a canonical world and compile a SkillProgram."""
    traces = []
    for _ in range(n_demos):
        world = make_world(kind)
        robot = SimRobot(world)
        perceiver = SimPerceiver(world)
        traces.append(record_demo(world, robot, perceiver))
    return compile_skill(traces, sku_id)
