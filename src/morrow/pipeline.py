"""The frozen investor sequence, assembled as one serializable record.

    record demo -> compiled state graph -> run from randomized poses
    -> force a grasp failure -> show recovery -> onboard a SECOND sku (no code)
    -> run it -> benchmark numbers

Both the CLI and the dashboard render exactly this object, so what an investor
sees on screen is the same thing the test suite asserts on.
"""

from __future__ import annotations

import numpy as np

from .eval.benchmark import run_benchmark
from .execute import RunResult, run_skill
from .skill import EDGES, SkillProgram
from .sim.scenarios import forced_failure_world, onboard, randomize
from .sim.sim_perceive import SimPerceiver
from .sim.sim_robot import SimRobot


def skill_graph(skill: SkillProgram) -> dict:
    """The compiled FSM as a serializable graph for display."""
    edges = []
    for a, b in EDGES:
        t = skill.transition(a, b)
        edges.append({
            "from": a.value, "to": b.value, "frame": t.frame,
            "success": t.success["name"], "recovery": t.recovery["action"],
        })
    return {"sku_id": skill.sku_id, "hash": skill.version_hash,
            "descriptor": skill.object_descriptor,
            "phase_indices": skill.metadata.get("phase_indices", {}), "edges": edges}


def _run_record(r: RunResult) -> dict:
    return {"success": r.success, "final_state": r.final_state,
            "first_attempt": r.first_attempt_success, "retries": r.retries,
            "recoveries": r.recoveries, "flagged": r.flagged, "timeline": r.timeline}


def investor_sequence(primary: str = "box", second: str = "cylinder",
                      n_show: int = 5, benchmark_n: int = 100) -> dict:
    skill = onboard(primary, primary)

    runs = []
    for i in range(n_show):
        w = randomize(primary, np.random.RandomState(500 + i))
        runs.append(_run_record(run_skill(skill, SimRobot(w), SimPerceiver(w), seed=i)))

    fw = forced_failure_world(primary)
    forced = _run_record(run_skill(skill, SimRobot(fw), SimPerceiver(fw), seed=99))

    skill2 = onboard(second, second)  # a new SKU, no code written
    w2 = randomize(second, np.random.RandomState(777))
    second_run = _run_record(run_skill(skill2, SimRobot(w2), SimPerceiver(w2), seed=7))

    return {
        "primary_graph": skill_graph(skill),
        "runs": runs,
        "forced_failure": forced,
        "second_graph": skill_graph(skill2),
        "second_run": second_run,
        "benchmark": run_benchmark(n=benchmark_n),
    }
