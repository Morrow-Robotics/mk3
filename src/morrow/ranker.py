"""A learned grasp-success ranker — the payoff of the data flywheel.

Trained on the episode log's grasp attempts (`journal.py`): features of a
proposed grasp -> probability it seals. It is a deliberately tiny, deterministic
logistic regression (numpy only, zero-initialised, fixed steps/lr, no
randomness) so the same log always yields the same model — a skill+ranker run
stays reproducible.

Its job is to notice what the analytic score structurally ignores. In this sim
that is grasp yaw: for a 180-symmetric product the two geometrically identical
grasps seal differently, and only experience reveals which. On the bench the
same mechanism learns which regions of a real deformable product seal.

The analytic path is always the default; the ranker is opt-in and only *adds* a
term. It is never trusted to override the fail-closed feasibility gates.
"""

from __future__ import annotations

import numpy as np

from .geometry import wrap_angle


def grasp_features(grasp_yaw_rel: float, offset_noise) -> np.ndarray:
    """[bias, cos(yaw_rel), sin(yaw_rel), |offset|] — enough to encode a yaw
    preference of any phase plus an offset-magnitude effect."""
    mag = float(np.hypot(offset_noise[0], offset_noise[1]))
    return np.array([1.0, np.cos(grasp_yaw_rel), np.sin(grasp_yaw_rel), mag])


class GraspRanker:
    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def prob(self, grasp_yaw_rel: float, offset_noise) -> float:
        x = grasp_features(grasp_yaw_rel, offset_noise)
        return float(1.0 / (1.0 + np.exp(-float(self.weights @ x))))

    @classmethod
    def fit(cls, records: list[dict], steps: int = 600, lr: float = 0.5) -> "GraspRanker":
        X, y = [], []
        for r in records:
            for g in r.get("grasp_attempts", []):
                if "grasp_yaw_rel" not in g:
                    continue
                X.append(grasp_features(g["grasp_yaw_rel"], g["grasp_offset_noise"]))
                y.append(1.0 if g.get("sealed") else 0.0)
        if not X or len(set(y)) < 2:
            return cls(np.zeros(4))  # no signal -> prob 0.5 everywhere -> no effect
        X = np.array(X)
        y = np.array(y)
        w = np.zeros(X.shape[1])
        n = len(y)
        for _ in range(steps):  # deterministic full-batch gradient descent
            p = 1.0 / (1.0 + np.exp(-X @ w))
            w -= lr * (X.T @ (p - y)) / n
        return cls(w)


def blend_score(ranker: GraspRanker, grasp_yaw: float, product_yaw_ref: float,
                offset_noise) -> float:
    """The additive term a ranker contributes to a grasp candidate's score,
    centred so an uninformative ranker (prob 0.5) contributes nothing."""
    rel = wrap_angle(grasp_yaw - product_yaw_ref)
    return ranker.prob(rel, offset_noise) - 0.5
