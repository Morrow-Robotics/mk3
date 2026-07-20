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


def default_carton() -> Carton:
    return Carton(cx=0.20, cy=0.0, yaw=0.0, hx=0.12, hy=0.12, wall_z=0.12, floor_z=0.0)


def pack_carton() -> Carton:
    """A larger case for multi-item sequences (a real order sizes its own case)."""
    return Carton(cx=0.20, cy=0.0, yaw=0.0, hx=0.15, hy=0.15, wall_z=0.12, floor_z=0.0)


_carton = default_carton  # internal alias


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
              slip_prob: float = 0.12, carton_yaw: bool = False,
              perception_dropout: float = 0.0, occlusion: float = 0.0) -> World:
    cx = rng.uniform(-0.25, -0.05)
    cy = rng.uniform(-0.15, 0.15)
    yaw = rng.uniform(-np.pi, np.pi)
    world = make_world(kind, cx, cy, yaw, slip_prob=slip_prob, rng=rng)
    world.perception_dropout_prob = perception_dropout
    world.occlusion_prob = occlusion
    if jitter_carton:
        world.carton.cx = 0.20 + rng.uniform(-0.04, 0.04)
        world.carton.cy = 0.0 + rng.uniform(-0.04, 0.04)
    if carton_yaw:
        world.carton.yaw = rng.uniform(-0.4, 0.4)
    return world


def staged(kind: str, rng: np.random.RandomState, distractor_kinds=None) -> World:
    """The target SKU plus distractor products staged elsewhere on the table.
    The perceiver must select the target by descriptor and ignore the rest."""
    world = randomize(kind, rng)
    others = distractor_kinds or [k for k in ("box", "cylinder", "pouch") if k != kind]
    spots = [(0.0, -0.28), (0.05, 0.28), (-0.05, -0.32)]
    for k, (dx, dy) in zip(others, spots):
        world.distractors.append(make_product(k, dx, dy, rng.uniform(-np.pi, np.pi)))
    return world


def staged_ambiguous(kind: str, rng: np.random.RandomState) -> World:
    """Two near-identical same-kind products — selection cannot disambiguate,
    so the cell should flag rather than guess which one to pack."""
    world = randomize(kind, rng)
    world.distractors.append(make_product(kind, 0.0, 0.28, rng.uniform(-np.pi, np.pi)))
    return world


def stress(kind: str, rng: np.random.RandomState) -> World:
    """A harder world: rotated carton, low-confidence frames, and occlusion noise."""
    return randomize(kind, rng, carton_yaw=True, perception_dropout=0.2, occlusion=0.15)


def structured(kind: str, rng: np.random.RandomState, seal_yaw_pref: float = 0.0) -> World:
    """A SKU whose seal reliability depends on grasp yaw — the signal the analytic
    score can't see and the learned ranker discovers from the log."""
    world = randomize(kind, rng)
    world.product.seal_yaw_pref = seal_yaw_pref
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


def onboard_timed(kind: str, sku_id: str, n_demos: int = 2):
    """Onboard and stash wall-clock changeover time in metadata (not hashed).

    The number the pitch cares about: minutes from demo to a running skill. In
    sim it is milliseconds; on the bench the recorder path populates the same
    field from a real teleop demo. Kept out of the content hash so determinism
    and serialization round-trips are unaffected."""
    import time
    t0 = time.perf_counter()
    skill = onboard(kind, sku_id, n_demos)
    skill.metadata = dict(skill.metadata, onboarding_seconds=round(time.perf_counter() - t0, 4))
    return skill
