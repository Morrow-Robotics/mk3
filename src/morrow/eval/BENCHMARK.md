# The frozen benchmark

`morrow eval` / `morrow.eval.run_benchmark(n=100, seed_base=1000)`.

## What it measures

For each product class (box, cylinder, pouch) the harness onboards a skill from
`n_demos` demonstrations, then runs it across `n` randomized worlds and, on the
same worlds, runs the open-loop replay baseline.

Randomization per trial (seeded by `seed_base + i`, so fully reproducible):
- product xy in the staging region, **full yaw** (−π, π]
- carton xy jittered ±4 cm
- 12% suction-slip probability on otherwise-good seals

## Metrics

- **final_success_rate** — reached `VERIFIED`, including autonomous recovery.
- **first_attempt_rate** — verified with **no** retries and **no** recoveries.
- **human_intervention_rate** — fraction that parked and flagged a person.
- **mean_retries / mean_recoveries** — autonomous work done per run.
- **baseline_open_loop_success_rate** — the taught-once replay, for contrast.

## Reproducibility

`run_benchmark(n) == run_benchmark(n)` is a test (`tests/test_eval.py`). Same
seeds → same worlds → same numbers. A regression is visible as a diff, not a
vibe. Candidate ordering is deterministic; the only randomness is the seeded
world/slip generation.

## Honesty

These are **simulator** numbers. They demonstrate the mechanism — perception →
hardware-verified grasp → per-transition recovery — not industrial reliability.
The analytic world has no wrinkled-film physics, no perception error beyond the
confidence gate, and no cycle-time constraint. The forced-failure world and slip
noise exist precisely so recovery is exercised rather than skipped. Do not quote
these as production numbers; quote them as "the mechanism works, here is the
harness that will measure the real cell the same way."
