"""The frozen investor sequence, assembled as one serializable record.

    record demo -> compiled state graph -> run from randomized poses
    -> force a grasp failure -> show recovery -> onboard a SECOND sku (no code)
    -> run it -> benchmark numbers

Both the CLI and the dashboard render exactly this object, so what an investor
sees on screen is the same thing the test suite asserts on.
"""

from __future__ import annotations

import numpy as np

from .eval.analysis import failure_breakdown
from .eval.benchmark import run_benchmark
from .eval.ranker_eval import compare_ranker, ranker_demo_pick
from .execute import RunResult, run_skill
from .journal import EpisodeLog, summarize_log
from .sequence import demo_pack_sequence
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

    # data flywheel: log every run's grasp attempts, then summarize
    journal = EpisodeLog()
    for i in range(30):
        w = randomize(primary, np.random.RandomState(900 + i))
        run_skill(skill, SimRobot(w), SimPerceiver(w), seed=i, journal=journal)
    flywheel = summarize_log(journal.records)
    ranker = compare_ranker(primary, n=50, n_train=50, seal_yaw_pref=0.0)
    ranker["example"] = ranker_demo_pick(primary, seed=6003, n_train=50, seal_yaw_pref=0.0)

    seq = demo_pack_sequence(seed=0)
    sequence = {"packed": seq.packed, "total": seq.total, "success": seq.success,
                "items": [{"sku": x.sku, "kind": x.kind, "slot": list(x.slot),
                           "final_state": x.final_state} for x in seq.results]}

    return {
        "primary_graph": skill_graph(skill),
        "runs": runs,
        "forced_failure": forced,
        "second_graph": skill_graph(skill2),
        "second_run": second_run,
        "benchmark": run_benchmark(n=benchmark_n),
        "benchmark_stress": run_benchmark(n=benchmark_n, stress_mode=True),
        "stuck_breakdown": failure_breakdown(primary, n=benchmark_n),
        "flywheel": flywheel,
        "ranker": ranker,
        "sequence": sequence,
    }
