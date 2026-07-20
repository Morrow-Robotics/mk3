# Physical bench acceptance — Real Bench v0 (Phase 4)

This is the acceptance test for the **physical** SO-101, and it is deliberately
**separate from the analytic `morrow eval` 100/88/0 numbers**. Those are sim
regression evidence. The gates below are the only evidence that mk3 survives
reality, and none of them can be produced without hardware. Nothing in this file
is claimed to have passed yet — it is the protocol to run once the arm is on the
bench.

## Rig

- LeRobot SO-101 **follower**, rigidly mounted, top-down; SO-101 **leader** for
  teleoperation. Standard **parallel-jaw** end-effector (not suction).
- Fixed **overhead camera**, calibrated table plane; camera-to-robot calibration
  `T_base_camera` recorded and version-pinned.
- Emergency stop wired and tested **before** any autonomous run
  (`SO101BenchRobot.emergency_stop`).
- Products: **three representative rectangular boxes** + one carton. Defer
  cylinders, pouches, and arbitrary consumer-video understanding.

## Procedure

1. **Teleoperate** one packing example → `record_live()` produces a real
   `DemonstrationTrace` → `compile.py` produces a `SkillProgram` (no code change).
2. **Move + rotate** the product and carton within the reach envelope; run the
   compiled skill closed-loop.
3. **Force a miss** (offset the grasp) and confirm per-transition recovery fires
   from the current state — not a full restart.
4. **Benchmark** across an explicit pose grid; log **every** attempt and every
   failure **by transition** to the journal. Only train the grasp ranker *after*
   real failures exist.
5. **Onboard two unseen box SKUs**; time each; confirm **no code changes** between
   SKUs.

## Pose grid (measure the real envelope, don't assume it)

The MuJoCo model predicts ~88% box-pack inside `x∈[0.24,0.28] m, |y|≤0.06 m`,
falling off past `x≥0.30`. Measure the physical envelope the same way so day one
produces a falsifiable sim-to-real comparison:

- x ∈ {0.22, 0.24, 0.26, 0.28, 0.30} m
- y ∈ {-0.06, -0.03, 0.0, 0.03, 0.06} m
- yaw ∈ {0°, 30°, 60°, 90°}
- ≥ 100 attempts total; each attempt logs pose, outcome, and failing transition.

## Fundraising gates (all required)

| Gate | Threshold |
|---|---|
| First-attempt success | ≥ 80% |
| Success within two retries | ≥ 95% |
| Forced misses recovered | ≥ 10 |
| Unseen box SKUs onboarded | 2, each < 15 min, **no code changes** |
| Per-attempt logs | real journal for every attempt |
| Workspace / failure map | honest, measured (not the sim's) |
| Design partner | ≥ 1 manufacturer providing product / video / pilot |

## Bring-up checklist — the five embodiment facts

Each `physics/arm.py` fact has a physical analogue to re-calibrate on the real arm:

1. **Gripper ctrl direction** — the model's ctrl is inverted (high = open). Verify
   the physical follower's convention before any close-on-product move.
2. **Tool-site offset** — the ~2 cm `gripperframe`→pinch offset is calibrated in
   sim; re-measure it on hardware (finger geometry differs).
3. **IK on a scratch state only** — plan on a copy, command the follower with joint
   targets; never let the planner teleport the commanded state mid-grasp.
4. **Abstract gripper intent** — record demos as open/closed intent, decoupled from
   the raw servo value.
5. **Workspace limits** — replace the sim envelope with the *measured* one from the
   pose grid; `reachable()` must fail-closed outside it.
