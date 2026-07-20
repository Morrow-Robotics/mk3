"""Prove the learned ranker earns its place: train on a log, then A/B it.

On a `structured` SKU (seal reliability depends on grasp yaw) the analytic score
picks the yaw arbitrarily. We log analytic-only runs, fit a `GraspRanker`, then
compare first-attempt success with and without it on held-out worlds. Both halves
are seeded, so the comparison is reproducible.
"""

from __future__ import annotations

import numpy as np

from ..execute import run_skill
from ..journal import EpisodeLog
from ..ranker import GraspRanker
from ..sim.scenarios import onboard, structured
from ..sim.sim_perceive import SimPerceiver
from ..sim.sim_robot import SimRobot


def compare_ranker(kind: str = "box", n: int = 60, n_train: int = 60,
                   seal_yaw_pref: float = 0.0) -> dict:
    skill = onboard(kind, kind)

    # 1) gather a log with the analytic ranker only, on structured worlds
    log = EpisodeLog()
    for i in range(n_train):
        w = structured(kind, np.random.RandomState(5000 + i), seal_yaw_pref)
        run_skill(skill, SimRobot(w), SimPerceiver(w), seed=i, journal=log)
    ranker = GraspRanker.fit(log.records)

    # 2) held-out A/B on the same worlds, analytic vs analytic+ranker
    def first_attempt_rate(use_ranker: bool) -> float:
        fa = 0
        for i in range(n):
            w = structured(kind, np.random.RandomState(6000 + i), seal_yaw_pref)
            r = run_skill(skill, SimRobot(w), SimPerceiver(w), seed=i,
                          ranker=ranker if use_ranker else None)
            fa += r.first_attempt_success
        return fa / n

    return {
        "kind": kind,
        "n": n,
        "n_train": n_train,
        "seal_yaw_pref": seal_yaw_pref,
        "analytic_first_attempt": first_attempt_rate(False),
        "ranker_first_attempt": first_attempt_rate(True),
    }


def ranker_demo_pick(kind: str = "box", seed: int = 6001, n_train: int = 60,
                     seal_yaw_pref: float = 0.0) -> dict:
    """One structured scene, analytic vs ranker: which grasp yaw each picked and
    whether it sealed. Makes the flywheel legible on a single run."""
    skill = onboard(kind, kind)
    log = EpisodeLog()
    for i in range(n_train):
        w = structured(kind, np.random.RandomState(5000 + i), seal_yaw_pref)
        run_skill(skill, SimRobot(w), SimPerceiver(w), seed=i, journal=log)
    ranker = GraspRanker.fit(log.records)

    def first_grasp(use_ranker: bool) -> dict:
        w = structured(kind, np.random.RandomState(seed), seal_yaw_pref)
        r = run_skill(skill, SimRobot(w), SimPerceiver(w), seed=seed,
                      ranker=ranker if use_ranker else None)
        g = r.grasp_attempts[0] if r.grasp_attempts else {}
        return {"grasp_yaw_rel": g.get("grasp_yaw_rel"), "sealed": g.get("sealed"),
                "first_attempt": r.first_attempt_success}

    return {"seed": seed, "analytic": first_grasp(False), "ranker": first_grasp(True)}
