"""Multi-item pack-out with the floating parallel-jaw gripper (MuJoCo physics).

The honest visual answer to "the clips show many objects, the arm places one":
several rectangular boxes fed from an in-feed row, each picked and placed into its
own slot in one carton, accumulating in a grid — in real contact physics.

Why the FLOATING gripper and not the SO-101 model: the SO-101 model is a small arm
whose reliable placement zone is tiny, so a multi-slot grid drops boxes (measured
~2–3 of N seat). The floating parallel jaw is positioned by mocap — exact, no reach
limit — so it seats the whole grid reliably. Same Robot boundary, same contact
grasp verdict; the trade is embodiment realism (floating vs a real arm) for
placement reach. The SO-101 model stays the single-box, IK-driven embodiment.

Honest ceiling (unchanged): these are **rectangular boxes**, not the picture
frames / books / varied materials a real clip shows. Diverse-material grasping and
multi-object video detection are Phase-4 hardware + end-effector work.
"""

from __future__ import annotations

import numpy as np

import mujoco

from .scene import CARTON, build_mjcf

_RGBA = ["0.10 0.45 0.90 1", "0.90 0.55 0.10 1", "0.16 0.64 0.64 1", "0.85 0.30 0.55 1"]
OPEN, CLOSE = 0.0, 0.035   # finger ctrl: 0 open, 0.035 closed
HB = 0.03                  # box half-extent (6 cm boxes)


def _pick_row(n):
    xs = np.linspace(-0.12, 0.12, n)
    return [(float(x), -0.20) for x in xs]


def _slots(n, c):
    # a 2xk grid inside the (large) carton, near its centre
    xs, ys = [c["cx"] - 0.04, c["cx"] + 0.04], [c["cy"] - 0.05, c["cy"] + 0.05]
    grid = [(x, y) for y in ys for x in xs]
    return grid[:n]


def build_multi_mjcf(n, picks, carton):
    base = build_mjcf("box", HB, HB, HB, picks[0][0], picks[0][1], 0.0, carton)
    base = base.replace('name="product"', 'name="box_0"')
    extra = "".join(
        f'<body name="box_{i}" pos="{picks[i][0]} {picks[i][1]} {HB}"><freejoint/>'
        f'<geom name="box_{i}" type="box" size="{HB} {HB} {HB}" mass="0.05" '
        f'friction="1.2 0.05 0.001" rgba="{_RGBA[i % len(_RGBA)]}"/></body>'
        for i in range(1, n))
    return base.replace("</worldbody>", extra + "</worldbody>")


class MultiCell:
    """Floating-gripper cell with n boxes + one carton; mocap-driven pick/place."""

    def __init__(self, n=4, carton=None):
        self.n = n
        self.carton = carton or dict(CARTON)
        self.picks = _pick_row(n)
        self.model = mujoco.MjModel.from_xml_string(build_multi_mjcf(n, self.picks, self.carton))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.lfg, self.rfg = m.geom("lfg").id, m.geom("rfg").id
        self.boxes = [m.body(f"box_{i}").id for i in range(n)]
        self.box_geoms = [m.geom(f"box_{i}").id for i in range(n)]
        self._renderer = None
        self.data.ctrl[:] = OPEN
        self.set_ee(self.picks[0][0], self.picks[0][1], 0.30)
        self.step(300)

    def set_ee(self, x, y, z):
        self.data.mocap_pos[0] = [x, y, z]
        self.data.mocap_quat[0] = [1, 0, 0, 0]

    def step(self, k):
        for _ in range(k):
            mujoco.mj_step(self.model, self.data)

    def grasped(self, gi):
        left = right = False
        for i in range(self.data.ncon):
            s = {int(self.data.contact[i].geom1), int(self.data.contact[i].geom2)}
            if gi in s and self.lfg in s:
                left = True
            if gi in s and self.rfg in s:
                right = True
        return left and right

    def seated(self):
        c = self.carton
        return sum(1 for b in self.boxes
                   if c["cx"] - c["hx"] <= self.data.xpos[b][0] <= c["cx"] + c["hx"]
                   and c["cy"] - c["hy"] <= self.data.xpos[b][1] <= c["cy"] + c["hy"]
                   and self.data.xpos[b][2] < 0.06)

    def render(self, camera="side", height=260, width=340):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()


def _moveto(cell, x, y, z, on_frame=None, steps=70, every=6):
    cur = cell.data.mocap_pos[0].copy()
    tgt = np.array([x, y, z])
    for k in range(1, steps + 1):
        cell.set_ee(*(cur + (tgt - cur) * (k / steps)))
        cell.step(3)
        if on_frame and k % every == 0:
            on_frame()


def _drive_pack(cell, on_frame=None):
    slots = _slots(cell.n, cell.carton)
    gpz = HB + 0.036            # palm height that straddles a box
    grasped = 0
    for i in range(cell.n):
        px, py = cell.picks[i]
        cell.data.ctrl[:] = OPEN
        _moveto(cell, px, py, gpz + 0.10, on_frame)     # over the in-feed box
        _moveto(cell, px, py, gpz, on_frame)            # down onto it
        cell.data.ctrl[:] = CLOSE
        cell.step(160)
        if on_frame:
            on_frame()
        grasped += int(cell.grasped(cell.box_geoms[i]))
        _moveto(cell, px, py, 0.24, on_frame)           # lift clear
        sx, sy = slots[i]
        _moveto(cell, sx, sy, 0.24, on_frame)           # traverse to slot
        _moveto(cell, sx, sy, gpz + 0.02, on_frame)     # lower into carton
        cell.data.ctrl[:] = OPEN
        cell.step(120)                                  # release
        if on_frame:
            on_frame()
        _moveto(cell, sx, sy, 0.24, on_frame)           # withdraw
    return grasped, cell.seated()


def pack_boxes(n=4):
    """Pack n boxes into the carton grid. Returns (cell, grasped, seated)."""
    cell = MultiCell(n=n)
    grasped, seated = _drive_pack(cell)
    return cell, grasped, seated


def capture_multi_pack(n=4, camera="side", height=260, width=340):
    """Render the floating-gripper multi-item pack to frames for the dashboard."""
    cell = MultiCell(n=n)
    cell.render(camera, height, width)  # warm the renderer
    frames = []
    grasped, seated = _drive_pack(cell, on_frame=lambda: frames.append(cell.render(camera, height, width)))
    return frames, grasped, seated
