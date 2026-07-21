"""Multi-object pack-out on the SO-101 model — the arm packs several objects from
a reconstructed scene into one carton, as a sequence that ports to real hardware.

The SO-101 model is a small arm: a single grasp is ~88% reliable and its placement
is imprecise (~2 cm), so a naive multi-slot sequence drops boxes. This packer adds
the SAME two recoveries the FSM uses, which is what makes it reliable AND
hardware-honest:

  * grasp verify + retry — after closing, `grasped()` must be true; else reopen and
    re-approach the box's current pose.
  * place verify + recover — after releasing, the box must be seated in its slot;
    else re-pick it from wherever it ended up and place again.

Every motion is real IK + a contact grasp, so `SO101BenchRobot` runs the identical
sequence on the physical arm. Objects are rectangular boxes (what the parallel jaw
grips); their sizes/positions come from the reconstructed scene.
"""

from __future__ import annotations

import numpy as np

import mujoco

from .arm import ARM_JOINTS, CLOSE, OPEN, WIDE, ArmWorld, _load, build_arm_scene

_RGBA = ["0.20 0.75 0.35 1", "0.90 0.55 0.15 1", "0.25 0.55 0.90 1",
         "0.85 0.30 0.55 1", "0.60 0.60 0.20 1"]

# Carton sized so imprecise-but-real placements still land inside, kept within the
# SO-101's reliable reach (shallow in y).
CARTON = dict(cx=0.267, cy=0.045, hx=0.055, hy=0.05, wall=0.03, thick=0.006)


def _box_body(name, x, y, hx, hz, rgba):
    return (f'<body name="{name}" pos="{x} {y} {hz}"><freejoint/>'
            f'<geom name="{name}" type="box" size="{hx} {hx} {hz}" mass="0.05" '
            f'friction="1 .05 .01" solref="0.01 1" contype="2" conaffinity="3" '
            f'condim="3" rgba="{rgba}"/></body>')


class MultiArmWorld(ArmWorld):
    """SO-101 + one carton + N objects at isolated in-feed spots. Reuses every
    ArmWorld primitive; `select(i)` points the grasp queries at object i."""

    def __init__(self, sizes, carton=None):
        self.n = len(sizes)
        self.sizes = sizes                       # list of (hx, hz) half-extents
        self.kind = "box"
        self.hx = self.hy = sizes[0][0]
        self.hz = sizes[0][1]
        self.carton = carton or dict(CARTON)
        self.picks = self._pick_grid(self.n)
        hx0, hz0 = sizes[0]
        base = build_arm_scene("box", hx0, hx0, hz0, self.picks[0][0], self.picks[0][1],
                               0.0, self.carton)
        base = base.replace('name="product"', 'name="box_0"').replace(
            'conaffinity="1" condim="3"', 'conaffinity="3" condim="3"')
        extra = "".join(
            _box_body(f"box_{i}", self.picks[i][0], self.picks[i][1], sizes[i][0], sizes[i][1],
                      _RGBA[i % len(_RGBA)])
            for i in range(1, self.n))
        self.model = _load(base.replace("</worldbody>", extra + "</worldbody>"))
        self.data = mujoco.MjData(self.model)
        self._dk = mujoco.MjData(self.model)
        m = self.model
        self.site = m.site("gripperframe").id
        self.gid = m.actuator("gripper").id
        self.aid = [m.actuator(nm).id for nm in ARM_JOINTS]
        self.qadr = [m.joint(nm).qposadr[0] for nm in ARM_JOINTS]
        self.dadr = [m.joint(nm).dofadr[0] for nm in ARM_JOINTS]
        self.grng = [m.jnt_range[m.joint(nm).id] for nm in ARM_JOINTS]
        self.gq = m.joint("gripper").qposadr[0]
        self.g_open, self.g_closed = 1.75, -0.17
        self._fixed, self._moving = self._jaw_geoms()
        self.boxes = [m.body(f"box_{i}").id for i in range(self.n)]
        self.box_geoms = [m.geom(f"box_{i}").id for i in range(self.n)]
        self.flagged = False
        mujoco.mj_forward(m, self.data)
        self.pid, self.g_prod = self.boxes[0], self.box_geoms[0]
        self.pinch_off = np.zeros(3)
        self.pinch_off = self._calibrate_pinch()
        self.drive(self.ik((self.picks[0][0], self.picks[0][1], self.SAFE_Z)), OPEN, 300)

    def _pick_grid(self, n):
        cols, rows = [0.25, 0.285], [-0.06, -0.035]
        return [(c, r) for r in rows for c in cols][:n]

    def select(self, i):
        self.pid, self.g_prod = self.boxes[i], self.box_geoms[i]

    def box_xy(self, i):
        return [float(v) for v in self.data.xpos[self.boxes[i]][:2]]

    def box_pos(self, i):
        return self.data.xpos[self.boxes[i]].copy()

    def in_carton(self, i):
        c, p = self.carton, self.data.xpos[self.boxes[i]]
        return bool(c["cx"] - c["hx"] <= p[0] <= c["cx"] + c["hx"]
                    and c["cy"] - c["hy"] <= p[1] <= c["cy"] + c["hy"] and p[2] < 0.06)

    def slots(self):
        # a compact cluster near the carton centre (the reliable place zone)
        c = self.carton
        xs = [c["cx"] - 0.02, c["cx"] + 0.02]
        ys = [c["cy"] - 0.012, c["cy"] + 0.012]
        return [(x, y) for y in ys for x in xs][:self.n]

    def _qnow(self):
        return [float(self.data.qpos[a]) for a in self.qadr]

    def render(self, camera="cell", height=260, width=340):
        if getattr(self, "_renderer", None) is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()


def _move(w, target, grip, dw, on_frame=None, every=3):
    start = w.pinch_pos().copy()
    target = np.asarray(target, dtype=float)
    steps = max(4, int(np.linalg.norm(target - start) / 0.02))
    for k in range(1, steps + 1):
        w.drive(w.ik(start + (target - start) * (k / steps), down_weight=dw), grip, 55)
        if on_frame and k % every == 0:
            on_frame()


def _grasp(w, i, on_frame, tries=3):
    """Grasp object i from its current pose; verify + retry. Returns grasped bool."""
    hz = w.sizes[i][1]
    for _ in range(tries):
        w.select(i)
        bx, by = w.box_xy(i)
        _move(w, (bx, by, hz + 0.06), OPEN, 0.5, on_frame)
        _move(w, (bx, by, hz), OPEN, 0.5, on_frame)
        w.drive(w._qnow(), CLOSE, 700)
        if w.grasped():
            return True
        w.drive(w._qnow(), OPEN, 150)
        _move(w, (bx, by, hz + 0.06), OPEN, 0.5, on_frame)
    return False


def _place(w, i, slot, on_frame):
    """Carry the grasped object to `slot` and release (tool-down for accuracy)."""
    hz = w.sizes[i][1]
    bx, by = w.box_xy(i)
    _move(w, (bx, by, 0.09), CLOSE, 0.5, on_frame)        # lift
    sx, sy = slot
    _move(w, (sx, sy, 0.09), CLOSE, 0.5, on_frame)        # traverse tool-down
    _move(w, (sx, sy, hz + 0.015), CLOSE, 0.5, on_frame)  # lower into carton
    w.drive(w._qnow(), WIDE, 300)                         # release
    _move(w, (sx, sy, 0.09), WIDE, 0.5, on_frame)         # withdraw


def _drive_pack(w, on_frame=None, place_tries=3):
    slots = w.slots()
    status = [False] * w.n
    for i in range(w.n):
        for _ in range(place_tries):
            if not _grasp(w, i, on_frame):
                _move(w, (*w.box_xy(i), 0.09), OPEN, 0.5, on_frame)  # can't grasp; skip
                break
            _place(w, i, slots[i], on_frame)
            if w.in_carton(i):
                status[i] = True
                break
            # placement miss: box slipped out — recover and re-pick it next loop
    return status


def pack_objects(sizes, carton=None, on_frame=None, place_tries=3):
    """Pack the objects (list of (hx,hz) half-extents) into the carton with the
    SO-101 model. grasp verify+retry and place verify+recover. Returns
    (world, seated, per_object) where per_object[i] is True iff object i seated."""
    w = MultiArmWorld(sizes, carton=carton)
    status = _drive_pack(w, on_frame, place_tries)
    return w, sum(status), status


def capture_pack_objects(sizes, carton=None, camera="cell", height=260, width=340):
    """Render the SO-101 model packing the objects into frames for the dashboard."""
    w = MultiArmWorld(sizes, carton=carton)
    w.render(camera, height, width)  # warm
    frames = []
    status = _drive_pack(w, on_frame=lambda: frames.append(w.render(camera, height, width)))
    return frames, sum(status), status
