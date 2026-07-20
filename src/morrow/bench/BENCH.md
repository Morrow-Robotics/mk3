# Bench bring-up

How to take the sim-proven stack onto a physical cell. The software above the
boundary (`compile`, `execute`, `candidates`, `motion`, `conditions`, `eval`)
does **not** change. You implement two adapters and one calibration.

## The swap

```python
# sim
from morrow.sim import SimRobot, SimPerceiver
robot, perceiver = SimRobot(world), SimPerceiver(world)

# bench — same run_skill, same everything downstream
from morrow.bench import BenchRobot, BenchPerceiver
robot = BenchRobot(arm=arm, vacuum=vac, calibration=calib)
perceiver = BenchPerceiver(camera=cam, segmenter=sam, calibration=calib)
```

## Hardware (minimum)

- **Arm**: top-down capable (LeRobot-class for the demo; industrial later).
- **End-effector**: suction cup(s). Budget real time to iterate cup size,
  durometer, and flow rate against your actual film — this eats schedules.
- **Vacuum pressure sensor** on one analog input. This is the grasp verdict.
  ~$50 and it does more for reliability than any amount of ranking.
- **Overhead RGB-D camera**, rigidly mounted, seeing the staging area + carton.

## Order of bring-up (matches Week 1 of the plan)

1. **Camera-to-robot calibration** → `T_base_camera`, intrinsics. Everything
   geometric depends on this; do it first and store a `calibration_id`.
2. **`BenchRobot`**: `get_ee_pose`, `follow`, `engage`/`release`,
   `safe_retract`, `park_and_flag`, `reachable` (IK + joint limits),
   `gripper_signal` (read the sensor). `holding` is then just the threshold.
3. **`BenchPerceiver`**: implement the `_capture / _segment /
   _deproject_centroid / _footprint / _yaw_candidates / _carton / _confidence`
   helpers against **recorded** frames first, then compose them in `observe`.
4. Record 1–3 real demonstrations → `DemonstrationTrace` (the recorder is the
   one sim-specific piece; write a bench recorder that fills the same contract),
   `compile_skill`, then `run_skill`.

## Invariants to preserve (don't "fix" these away)

- **Grasp is verified by the vacuum sensor, never by vision** (the tool
  occludes the product at grasp time).
- **Re-perceive before every attempt**; recovery runs from the actual current
  state.
- **Fail-closed feasibility gates are not safety.** Add a real safety layer
  (E-stop, force limits, guarding) before anything runs near a person.
- Keep skills serializable (`morrow.serialize`) so a hash pins exactly what ran.

## What "done" looks like on the bench

The same investor sequence as sim, on real product: record → show the compiled
state graph → move product/carton → run → force a miss → watch it recover →
onboard a second SKU with no code → read the metrics off `morrow eval`-style
aggregation of real runs.
