"""Aggregate a batch of runs into the numbers an investor and an operator ask for.

first_attempt_rate  — verified with no retries and no recoveries (clean)
final_success_rate  — verified at all, including autonomous recovery
human_intervention_rate — fraction that parked and flagged for a person
mean_retries / mean_recoveries — how much autonomous work the cell did
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..execute import RunResult


@dataclass
class Metrics:
    n: int
    final_success_rate: float
    first_attempt_rate: float
    human_intervention_rate: float
    mean_retries: float
    mean_recoveries: float

    def as_dict(self) -> dict:
        return asdict(self)


def summarize(results: list[RunResult]) -> Metrics:
    n = len(results)
    if n == 0:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return Metrics(
        n=n,
        final_success_rate=sum(r.success for r in results) / n,
        first_attempt_rate=sum(r.first_attempt_success for r in results) / n,
        human_intervention_rate=sum(r.flagged for r in results) / n,
        mean_retries=sum(r.retries for r in results) / n,
        mean_recoveries=sum(r.recoveries for r in results) / n,
    )
