"""Legibility helpers: where a stress batch gets stuck, made countable.

`failure_breakdown` runs a stress batch and tallies `failure_reason` (or
"completed") so the dashboard can show *which* edge the rare human-flags happen
on, rather than just a single intervention percentage.
"""

from __future__ import annotations

import numpy as np

from ..execute import run_skill
from ..sim.scenarios import onboard, stress
from ..sim.sim_perceive import SimPerceiver
from ..sim.sim_robot import SimRobot


def failure_breakdown(kind: str = "box", n: int = 100, seed_base: int = 1000) -> dict:
    skill = onboard(kind, kind)
    counts: dict = {}
    for i in range(n):
        w = stress(kind, np.random.RandomState(seed_base + i))
        r = run_skill(skill, SimRobot(w), SimPerceiver(w), seed=i)
        key = r.failure_reason if r.failure_reason else "completed"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
