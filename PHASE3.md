# morrow — physics + CV + the real LeRobot arm (Phase 3)

`README.md` covers the analytic core: a demonstration compiler that turns one
short demo into a **verified state-machine skill** (`READY → APPROACHED → GRASPED
→ LIFTED → OVER_CARTON → PLACED → RELEASED → VERIFIED`), driven by `run_skill`
over a `Robot`/`Perceiver` boundary. Same FSM, same compiler, same conditions.

This document covers the layer built on top of it: **customer video → physics-
accurate MuJoCo sim on the actual LeRobot SO-101 arm**. It is honest about what is
real and what a monocular consumer clip cannot give you.

> Optional dependency. `pip install -e '.[physics]'` (MuJoCo). The SAM2 pieces
> also need real weights — set `MORROW_SAM2_CKPT` to a `sam2.1` checkpoint. Every
> physics/CV test is `importorskip`-guarded, so a bare install still runs green.

## The boundary is the whole trick

`run_skill` doesn't know what's underneath it. The exact same compiled skill runs
against three embodiments that all implement the `Robot`/`Perceiver` protocols:

| Embodiment | Module | Grasp verdict | Notes |
|---|---|---|---|
| Analytic sim | `morrow.sim` | seal geometry | numpy only, the frozen benchmark |
| Floating parallel-jaw | `physics/world.py` (`MjWorld`) | two-finger **contact** | MuJoCo, fast, mocap-welded gripper |
| **Real SO-101 arm** | `physics/arm.py` (`ArmWorld`) | two-finger **contact** | MuJoCo Menagerie, 5-DOF + IK |

No suction anywhere — the first arms are the standard LeRobot parallel jaw, so
grasp is verified from hardware contact between the product and *both* fingers,
never from vision.

## The real arm — `physics/arm.py`

`ArmWorld` / `ArmRobot` / `ArmPerceiver` wrap the actual **SO-101** (MuJoCo
Menagerie `robotstudio_so101`). An orientation-aware damped-least-squares IK
drives the position actuators tool-down to put the *pinch point between the
fingers* on each Cartesian waypoint; the jaw grasps by friction; `run_skill`
carries the FSM to `VERIFIED` with the product physically in the carton.

Five embodiment facts that are load-bearing (all verified against the model):

- The gripper ctrl is **inverted** — `ctrl≈1.75` opens ~12 cm, `ctrl≈-0.17` closes.
- The tool site sits ~2 cm off the fixed jaw, so the grasp happens at a small
  site-frame offset that is **calibrated once** at construction, not hard-coded.
- IK runs on a **scratch** `MjData` and only returns joint targets; the live arm
  moves only by driving actuators — otherwise a single IK jump teleports the arm
  out from under a grasped box (this was the bug that made every lift "drop" it
  despite a measured 72 N grip).
- The demo records **abstract** gripper intent (0 open / 1 closed), decoupled from
  the inverted raw ctrl, so the compiler's phase extraction lands the right frames.
- The arm is **small**: box-pack is ~88 % in the core envelope `x∈[0.24,0.28] m,
  |y|≤0.06 m`, degrading at the reach edge (`x≥0.30` mostly fails — it can't stay
  tool-down that far out). `in_workspace` surfaces this; it is real, not hidden.

Honest limit: the **cylinder is best-effort only**. A round object gives the
parallel jaw curved line-contact that slips during the lift — a real parallel-jaw
limitation, not a sim artefact. The showcase and tests use the box.

## Watch a real clip — `physics/watch.py`

The full **watch → do** path, built to the honest boundary:

1. **SAM2** (real weights, MPS) segments the carton and an operator-seeded product
   on a frame of a real clip → precise pixel bboxes. Single-frame segmentation is
   reliable; fully-unattended video tracking **drifts** under hand occlusion
   (measured ~43/90 frames, carton centre wandering >250 px), so segmentation is
   operator-*seeded*, semi-automatic — not magic.
2. A monocular clip has **no metric scale**, so the operator supplies metres-per-
   pixel + product kind/height (same honesty as `annotate`). SAM2 gives the pixel
   geometry, nothing invents depth.
3. The watched workflow is reproduced by the **real SO-101** in its validated reach
   envelope — the same task, executed and verified in physics, not a pixel-perfect
   metric replay (impossible from one uncalibrated camera). The watched product is
   clamped into the arm's graspable/reachable range so the pack is always valid.

`have_sam2()` gates everything, exactly like the MuJoCo optional dependency.

## Packing activity — `physics/pattern.py`

Counting *discrete items* placed in a carton is **not** reliably recoverable from
occluded consumer clips — the motion signal is a continuous stream of hand
activity and a peak count is threshold-sensitive (measured 9/7/4 peaks at
confidence 0.34–0.49 on the three clips). So we don't claim it. What is genuine
(pure cv2 frame-diff, deterministic): a **packing-activity profile** over the
SAM2-detected carton region — `active_fraction`, a *candidate* place-event count,
a `confidence` in [0,1] (`LOW`/`MEDIUM`/`HIGH` from peak separation), event times,
and a sparkline. The count is an explicit low-confidence, **operator-confirmable**
estimate, never an assertion.

## Running it

```bash
pip install -e '.[physics]'                 # MuJoCo backend
export MORROW_SAM2_CKPT=/path/to/sam2.1_hiera_tiny.pt   # for the watch pipeline

morrow cell                                 # dashboard on http://127.0.0.1:8001
morrow cell --shot dash.html                # render the dashboard to a file
morrow watch                                # SAM2 → SO-101 pack on the showcase clip
morrow watch --clip videos/my.mp4 --frame 20 --scale 0.00024 --kind box --shot out
morrow annotate examples/annotation_box.json   # a marked-up clip → physics pack
```

Drop customer clips into `./videos/` (git-ignored). The dashboard shows, left to
right: the clips, the **real SO-101** packing (teal hero), the watch→SAM2→pack
panel with the activity sparkline, the fast floating-gripper packs, and an
**interactive marker** — grab any clip frame, drag a rough box, hit *SAM2 refine*,
and pack it on the real arm.

## Layout

```
src/morrow/physics/
  world.py  mj_robot.py  mj_perceive.py  record.py   # floating parallel-jaw cell
  arm.py                                             # the REAL SO-101 (IK + friction grasp)
  watch.py                                           # SAM2 segment → annotation → SO-101 pack
  pattern.py                                         # cv2 packing-activity profile
  film.py  showcase.py  webview.py                   # render + the `morrow cell` dashboard
  annotate.py                                        # operator-marked clip → physics skill
tests/  test_physics.py  test_watch.py  test_pattern.py   # all optional-dep guarded
```

## What this is, and isn't

This is the software half — it makes the pitch legible and the physics honest.
The load-bearing next step is **real hardware bring-up + the customer calls**,
which is not a code task. The `Robot`/`Perceiver` protocols and the SO-101 IK are
the bridge: when a physical arm arrives, it implements the same boundary and the
same `run_skill` drives it.
