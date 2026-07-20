# Style Guide

Same taste as mk2: three codebases, one idea each. Precedence when they
conflict: **correct → simple → small → fast**.

- **tinygrad** — radical smallness. Every line is a liability; the best diff is
  a deletion. Abstractions must pay rent.
- **openpilot** — pragmatic robustness. This code commands a physical arm.
  Prefer the boring, obvious thing that works. Fail loud at the boundary.
- **PyTorch** — a humane API. The surface a person touches is obvious and hard
  to misuse, even if the internals are gnarly.

## mk3-specific conventions

These fell out of the architecture and are worth keeping.

- **Boundaries, not backends-in-the-core.** Hardware and perception are
  `Protocol`s (`robot.py`, `perceive.py`). The compiler, executor, and
  evaluation never import `sim`. If you need a new capability from the robot,
  add it to the boundary and implement it in *every* backend, not just sim.

- **Grasp is hardware, placement is vision.** Never verify a grasp from a
  camera — the tool occludes the product. `conditions.py` reads `scene.holding`
  (the vacuum/force verdict) for `grasped`. Keep it that way.

- **Serializable skills.** A `SkillProgram` holds no callables and no raw poses
  — only named conditions, frame-relative parameters, and recovery names. It
  must round-trip to JSON and hash reproducibly (`hash_skill`). If you're
  tempted to store a lambda on a transition, you're about to break versioning.

- **Re-perceive before you act.** Recovery reasons about the world as it is, not
  a stale snapshot. Every candidate attempt calls `perceiver.observe()` before
  it instantiates motion. Do not cache a scene across an attempt.

- **Determinism is a feature.** Candidate generation is seeded by
  `(seed, edge, round)`; the nominal demonstrated parameters are always
  candidate zero. Randomness lives in the *evaluation* setup, never loose inside
  the planner. The benchmark must satisfy `run_benchmark(n) == run_benchmark(n)`.

- **The simulator is honest.** It never reports a number it can't defend as a
  mechanism demonstration. Slip noise and forced failures exist so recovery is
  actually exercised — don't tune them away to make a slide look better.

- **One concept per file.** If a file needs section headers to navigate, it's
  two files. Public API is explicit in `__init__.py`; internals are not
  re-exported.

## The review test

1. Could this be smaller? (tinygrad)
2. Will it survive contact with a real arm — bad perception, a missed seal,
   a moved product? (openpilot)
3. Is the call site something you'd hand a stranger? (PyTorch)
