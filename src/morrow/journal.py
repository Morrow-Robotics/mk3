"""The episode log — the data flywheel, made concrete and deterministic.

Every run can append one structured record: what skill (by hash) ran on what
seed, how it ended, and — crucially — the parameters and sealed/miss outcome of
every grasp attempt. Those grasp records are the training set a learned
grasp-success ranker would consume later; until then they are just an honest
audit trail.

Records carry NO wall-clock time (that would break reproducibility). A run on a
given (skill, seed) always logs the same record.
"""

from __future__ import annotations

import json


class EpisodeLog:
    """Append-only episode log, optionally persisted to a JSONL file."""

    def __init__(self, path: str | None = None):
        self.path = path
        self.records: list[dict] = []

    def append(self, record: dict) -> None:
        self.records.append(record)
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    def __len__(self) -> int:
        return len(self.records)


def record_from_result(result, skill, seed: int) -> dict:
    """Build a deterministic episode record from a RunResult (duck-typed)."""
    return {
        "sku_id": result.sku_id,
        "skill_hash": skill.version_hash,
        "seed": int(seed),
        "success": bool(result.success),
        "final_state": result.final_state,
        "first_attempt": bool(result.first_attempt_success),
        "retries": int(result.retries),
        "recoveries": int(result.recoveries),
        "attempts": int(result.attempts),
        "flagged": bool(result.flagged),
        "grasp_attempts": list(result.grasp_attempts),
    }


def read_log(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summarize_log(records: list[dict]) -> dict:
    """Aggregate a set of episode records — the shape a flywheel dashboard shows."""
    n = len(records)
    grasps = [g for r in records for g in r.get("grasp_attempts", [])]
    sealed = sum(1 for g in grasps if g.get("sealed"))
    return {
        "episodes": n,
        "successes": sum(1 for r in records if r.get("success")),
        "grasp_attempts": len(grasps),
        "grasp_seal_rate": (sealed / len(grasps)) if grasps else 0.0,
        "mean_retries": (sum(r.get("retries", 0) for r in records) / n) if n else 0.0,
    }
