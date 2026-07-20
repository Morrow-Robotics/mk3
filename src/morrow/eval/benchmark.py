"""The frozen randomized benchmark.

Deterministic given `seed_base`: same seeds -> same worlds -> same numbers, so a
result is reproducible and a regression is visible. Each product class is
onboarded once, then run across N randomized poses (product + carton moved,
full yaw, suction slip noise). The open-loop replay baseline is run on the same
worlds for the changeover contrast.
"""

from __future__ import annotations

import numpy as np

from ..execute import run_skill
from ..sim.record import record_demo
from ..sim.scenarios import make_world, onboard, randomize, stress
from ..sim.sim_perceive import SimPerceiver
from ..sim.sim_robot import SimRobot
from .baseline import run_fixed_replay
from .metrics import summarize

KINDS = ("box", "cylinder", "pouch")


def run_benchmark(n: int = 100, seed_base: int = 1000, kinds=KINDS, n_demos: int = 2,
                  stress_mode: bool = False, journal=None) -> dict:
    build = stress if stress_mode else randomize
    report = {"n": n, "seed_base": seed_base, "stress": stress_mode, "kinds": {}}
    for kind in kinds:
        skill = onboard(kind, kind, n_demos=n_demos)
        w0 = make_world(kind)
        trace = record_demo(w0, SimRobot(w0), SimPerceiver(w0))

        results, baseline_ok = [], 0
        for i in range(n):
            w = build(kind, np.random.RandomState(seed_base + i))
            results.append(run_skill(skill, SimRobot(w), SimPerceiver(w), seed=i, journal=journal))
            wb = build(kind, np.random.RandomState(seed_base + i))
            baseline_ok += int(run_fixed_replay(trace, skill, wb))

        report["kinds"][kind] = {
            "skill_hash": skill.version_hash,
            "onboarding": {"n_demos": n_demos, "code_changes": 0},
            "morrow": summarize(results).as_dict(),
            "baseline_open_loop_success_rate": baseline_ok / n,
        }
    return report


def format_report(report: dict) -> str:
    mode = "stress" if report.get("stress") else "standard"
    lines = [f"benchmark ({mode})  n={report['n']}  seed_base={report['seed_base']}", ""]
    for kind, r in report["kinds"].items():
        m = r["morrow"]
        lines.append(f"[{kind}]  skill {r['skill_hash']}  onboarded from "
                     f"{r['onboarding']['n_demos']} demos, "
                     f"{r['onboarding']['code_changes']} code changes")
        lines.append(f"    morrow   final {m['final_success_rate']*100:5.1f}%   "
                     f"first-attempt {m['first_attempt_rate']*100:5.1f}%   "
                     f"human-intervention {m['human_intervention_rate']*100:4.1f}%")
        lines.append(f"    replay   final {r['baseline_open_loop_success_rate']*100:5.1f}%   "
                     f"(open-loop, taught once — fails when the product moves)")
        lines.append("")
    return "\n".join(lines)
