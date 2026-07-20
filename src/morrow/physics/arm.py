"""arm.py — the REAL LeRobot SO-101 arm as the Robot boundary.

Phase 3 replaces the floating mocap gripper with the actual 5-DOF SO-101
(MuJoCo Menagerie, `robotstudio_so101`). The same `run_skill` FSM drives it:
the skill's Cartesian waypoints are the *pinch point* between the fingers, and
an orientation-aware damped-least-squares IK drives the position actuators
tool-down to put the pinch there. The parallel jaw grasps by friction and the
grasp verdict is contact between the product and BOTH jaws — no suction, no flag.

Two hard-won facts about this embodiment, both verified against the model:

  * The gripper ctrl is INVERTED: ctrl≈1.75 opens the jaws to ~12 cm, ctrl≈-0.17
    closes them. `OPEN`/`CLOSE` below encode that.
  * The tool site (`gripperframe`) is offset ~2 cm from the fixed jaw, so the
    grasp actually happens at a small site-frame offset that we CALIBRATE once
    at construction (`_calibrate_pinch`) rather than hard-code.

IK runs on a scratch MjData and only ever returns joint targets; the live arm is
moved solely by driving the position actuators, so physics carries a grasped
product instead of the arm teleporting out from under it.

Reach: the SO-101 is a small desk arm. Measured box-pack reliability (frame-
relative skill from one demo at (0.27,-0.06), then run across a grid): ~88% in the
CORE envelope x in [0.24, 0.28] m, |y| <= 0.06 m, degrading toward the reach edge
(x >= 0.30 mostly fails — the arm can't stay tool-down that far out). So the whole
cell (product + carton) is placed in that core, NOT at the analytic sim's
coordinates, and `in_workspace` surfaces the limit rather than hiding it. The
guarded test uses the demo pose; the edge falloff is a real hardware property.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

import mujoco

from ..geometry import Transform, frame, translation, wrap_angle, yaw_of
from ..scene import SceneState
from ..sim.world import rect_aabb

_MENAGERIE = os.path.expanduser(
    "~/.cache/robot_descriptions/mujoco_menagerie/robotstudio_so101")

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
OPEN = 0.7      # ctrl to hold the jaws open enough to straddle a ~4 cm product
CLOSE = -0.17   # ctrl to squeeze (position target the product blocks)
WIDE = 1.4      # ctrl to release wide

# A carton sized and placed inside the SO-101's reach (all metres). The usable
# top-down grasp envelope is narrow (|y| <= ~0.07 at grasp depth), so product and
# carton straddle y=0 rather than sitting far out.
ARM_CARTON = dict(cx=0.27, cy=0.06, hx=0.05, hy=0.05, wall=0.04, thick=0.006)

# Product half-extents the standard jaw can actually straddle in this workspace.
# HONEST reliability: the box packs reliably (flat faces → firm two-face pinch).
# The cylinder is BEST-EFFORT only — a round object gives the parallel jaw curved
# line-contact that slips during the lift (measured ~0/5 at the demo pose), a real
# parallel-jaw limitation, not a sim artefact. So the showcase/tests use the box;
# a reliable cylinder pick would need a different jaw or a caging grasp (unbuilt).
ARM_SIZES = {
    "box": (0.02, 0.02, 0.03),
    "cylinder": (0.02, 0.02, 0.03),
}


def _carton_walls_xml(c: dict) -> str:
    cx, cy, hx, hy = c["cx"], c["cy"], c["hx"], c["hy"]
    t, wz = c["thick"], c["wall"]
    rgba = "0.82 0.70 0.55 1"

    def wall(name, x, y, sx, sy):
        return (f'<geom name="{name}" type="box" pos="{x} {y} {wz/2}" '
                f'size="{sx} {sy} {wz/2}" rgba="{rgba}" contype="1" conaffinity="1"/>')

    return "\n      ".join([
        wall("cw_n", cx, cy + hy + t, hx + t, t),
        wall("cw_s", cx, cy - hy - t, hx + t, t),
        wall("cw_e", cx + hx + t, cy, t, hy),
        wall("cw_w", cx - hx - t, cy, t, hy),
    ])


def _product_xml(kind: str, hx: float, hy: float, hz: float, px: float, py: float, pyaw: float) -> str:
    q = f"{np.cos(pyaw/2)} 0 0 {np.sin(pyaw/2)}"
    fr = 'friction="1 .03 .003" solref="0.01 1" contype="2" conaffinity="1" condim="3"'
    if kind == "cylinder":
        g = f'<geom type="cylinder" name="product" size="{hx} {hz}" {fr} rgba="0.90 0.35 0.15 1"/>'
    else:
        g = f'<geom type="box" name="product" size="{hx} {hy} {hz}" {fr} rgba="0.20 0.75 0.35 1"/>'
    return (f'<body name="product" pos="{px} {py} {hz}" quat="{q}">\n'
            f'      <freejoint/>\n      {g}\n    </body>')


def build_arm_scene(kind: str, hx: float, hy: float, hz: float,
                    px: float, py: float, pyaw: float, carton: dict) -> str:
    return f"""<mujoco model="so101_cell">
  <include file="so101.xml"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="150" elevation="-22"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.12 0.16 0.22" rgb2="0 0 0" width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.18 0.22 0.28"
      rgb2="0.10 0.13 0.18" markrgb="0.55 0.6 0.7" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="6 6" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0.2 0 1.2" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="grid"/>
    <camera name="cell" pos="0.30 -0.55 0.42" xyaxes="1 0 0 0 0.5 0.9"/>
    <camera name="top" pos="0.27 0 0.7" xyaxes="1 0 0 0 1 0"/>
    {_carton_walls_xml(carton)}
    {_product_xml(kind, hx, hy, hz, px, py, pyaw)}
  </worldbody>
</mujoco>"""


def _load(xml: str) -> mujoco.MjModel:
    """Load a scene that `<include>`s so101.xml, resolving meshes via its dir."""
    fd, path = tempfile.mkstemp(suffix=".xml", dir=_MENAGERIE)
    try:
        os.write(fd, xml.encode()); os.close(fd)
        return mujoco.MjModel.from_xml_path(path)
    finally:
        os.remove(path)


def _quat_yaw(q) -> float:
    qw, qx, qy, qz = q
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


class ArmWorld:
    """A MuJoCo cell built around the real SO-101, plus the queries and the
    IK/drive primitives the Robot boundary needs. Coordinates are the arm's."""

    SAFE_Z = 0.10  # the small arm can only stay tool-down up to ~0.10 m
    WORKSPACE = (0.21, 0.32, -0.10, 0.10)  # honest top-down reach envelope

    def __init__(self, kind: str = "box", px: float = 0.27, py: float = -0.06,
                 pyaw: float = 0.0, carton: dict | None = None, size=None):
        if kind not in ARM_SIZES:
            raise ValueError(f"arm cell supports {list(ARM_SIZES)}, not {kind!r}")
        self.kind = kind
        self.hx, self.hy, self.hz = size if size is not None else ARM_SIZES[kind]
        self.carton = carton or ARM_CARTON
        self.model = _load(build_arm_scene(kind, self.hx, self.hy, self.hz, px, py, pyaw, self.carton))
        self.data = mujoco.MjData(self.model)
        self._dk = mujoco.MjData(self.model)  # scratch state for kinematic IK
        m = self.model
        self.site = m.site("gripperframe").id
        self.pid = m.body("product").id
        self.g_prod = m.geom("product").id
        self.gid = m.actuator("gripper").id
        self.aid = [m.actuator(n).id for n in ARM_JOINTS]
        self.qadr = [m.joint(n).qposadr[0] for n in ARM_JOINTS]
        self.dadr = [m.joint(n).dofadr[0] for n in ARM_JOINTS]
        self.grng = [m.jnt_range[m.joint(n).id] for n in ARM_JOINTS]
        self.gq = m.joint("gripper").qposadr[0]
        self.g_open, self.g_closed = 1.75, -0.17
        self._fixed, self._moving = self._jaw_geoms()
        self.flagged = False
        mujoco.mj_forward(m, self.data)
        self.pinch_off = np.zeros(3)  # provisional until calibrated (IK targets the site)
        self.pinch_off = self._calibrate_pinch()
        # home: hover tool-down over the product so the first follow starts clean
        self.drive(self.ik((px, py, self.SAFE_Z)), OPEN, 300)

    # --- geometry bookkeeping -------------------------------------------
    def _jaw_geoms(self):
        m = self.model
        fixed, moving = set(), set()
        for gi in range(m.ngeom):
            b = m.body(m.geom_bodyid[gi]).name
            nm = m.geom(gi).name or ""
            if b == "gripper" and nm.startswith("fixed_jaw"):
                fixed.add(gi)
            elif b == "moving_jaw_so101_v1":
                moving.add(gi)
        return fixed, moving

    def _calibrate_pinch(self) -> np.ndarray:
        """Site-frame offset from the tool site to the jaw-tip midpoint at the
        pre-grasp opening — where a straddled product actually sits."""
        self.drive(self.ik((0.27, 0.0, 0.08)), OPEN, 300)
        R = self.data.site_xmat[self.site].reshape(3, 3)
        ft = self.data.geom_xpos[self.model.geom("fixed_jaw_sph_tip2").id]
        mt = self.data.geom_xpos[self.model.geom("moving_jaw_sph_tip2").id]
        return R.T @ ((ft + mt) / 2 - self.data.site_xpos[self.site])

    # --- stepping / actuation -------------------------------------------
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def ik(self, target_xyz, iters: int = 400, down_weight: float = 0.5) -> list:
        """DLS IK on the scratch state; returns arm joint targets. `down_weight`
        trades position accuracy against pointing the tool straight down: full
        weight for the grasp (must be tool-down), low weight while carrying (the
        small arm can only reach lift height if the wrist is allowed to tilt).
        Never mutates the live arm — the caller drives to the result."""
        dk = self._dk
        dk.qpos[:] = self.data.qpos
        dk.qvel[:] = 0.0
        tp = np.asarray(target_xyz, dtype=float)
        down = np.array([0.0, 0.0, -1.0])
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        for _ in range(iters):
            mujoco.mj_forward(self.model, dk)
            R = dk.site_xmat[self.site].reshape(3, 3)
            pp = dk.site_xpos[self.site] + R @ self.pinch_off
            mujoco.mj_jacSite(self.model, dk, jp, jr, self.site)
            J = np.vstack([jp[:, self.dadr], jr[:, self.dadr] * down_weight])
            e = np.concatenate([tp - pp, np.cross(R[:, 0], down) * down_weight])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            for i, a in enumerate(self.qadr):
                dk.qpos[a] = np.clip(dk.qpos[a] + dq[i] * 0.6, *self.grng[i])
        return [float(dk.qpos[a]) for a in self.qadr]

    def drive(self, q, grip: float, n: int) -> tuple:
        for i, a in enumerate(self.aid):
            self.data.ctrl[a] = q[i]
        self.data.ctrl[self.gid] = grip
        self.step(n)
        return tuple(q)

    def hold_grip(self, grip: float, n: int) -> None:
        """Keep the current arm pose, change only the jaws (grasp / release)."""
        self.drive([float(self.data.qpos[a]) for a in self.qadr], grip, n)

    # --- queries (arm coordinates) --------------------------------------
    def pinch_pos(self) -> np.ndarray:
        R = self.data.site_xmat[self.site].reshape(3, 3)
        return self.data.site_xpos[self.site] + R @ self.pinch_off

    def ee_yaw(self) -> float:
        return _quat_yaw(self.data.xquat[self.model.body("gripper").id])

    def jaw_gap(self) -> float:
        # 1.0 fully open, ~0 closed.
        return float((self.data.qpos[self.gq] - self.g_closed) / (self.g_open - self.g_closed))

    def grasped(self) -> bool:
        """Product pinched between a fixed AND a moving jaw geom (contact verdict)."""
        f = mv = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            has = self.g_prod in (g1, g2)
            if not has:
                continue
            other = g2 if g1 == self.g_prod else g1
            if other in self._fixed:
                f = True
            if other in self._moving:
                mv = True
        return f and mv

    def product_xy(self) -> np.ndarray:
        return self.data.xpos[self.pid][:2].copy()

    def product_top_z(self) -> float:
        return float(self.data.xpos[self.pid][2]) + self.hz

    def product_yaw(self) -> float:
        return _quat_yaw(self.data.xquat[self.pid])

    def product_footprint(self) -> np.ndarray:
        xy = self.product_xy()
        return rect_aabb(float(xy[0]), float(xy[1]), self.hx, self.hy, self.product_yaw())

    def in_workspace(self, xy) -> bool:
        x0, x1, y0, y1 = self.WORKSPACE
        return bool(x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1)


# --------------------------------------------------------------------------
# Robot boundary over the SO-101.
# --------------------------------------------------------------------------
CART_STEP_LEN = 0.02   # metres of pinch travel per interpolated IK step
DRIVE_PER_STEP = 55    # physics steps per interpolated step
CAPTURE_EVERY = 2      # interpolated steps between render captures


class ArmRobot:
    has_vacuum = False

    def __init__(self, world: ArmWorld, on_frame=None):
        self.world = world
        self.on_frame = on_frame
        self._grip = OPEN  # jaws start open; engage/release change this

    # --- Robot protocol -------------------------------------------------
    def get_ee_pose(self) -> Transform:
        return frame(self.world.pinch_pos(), self.world.ee_yaw())

    def reachable(self, pose: Transform) -> bool:
        xyz = translation(pose)
        z = float(xyz[2])
        return self.world.in_workspace(xyz[:2]) and -0.02 <= z <= self.world.SAFE_Z + 0.03

    def gripper_signal(self) -> float:
        return self.world.jaw_gap()

    def holding(self) -> bool:
        return self.world.grasped()

    def follow(self, waypoints: list[Transform]) -> None:
        for wp in waypoints:
            self._move_to(translation(wp), yaw_of(wp))

    def engage(self) -> None:
        self._grip = CLOSE
        self.world.hold_grip(CLOSE, 700)   # squeeze until the product blocks the jaws
        self._capture()

    def release(self) -> None:
        self._grip = WIDE
        self.world.hold_grip(WIDE, 300)
        self._capture()

    def safe_retract(self) -> None:
        p = self.world.pinch_pos()
        self._move_to((p[0], p[1], self.world.SAFE_Z), self.world.ee_yaw())

    def park_and_flag(self) -> None:
        self.release()
        self._move_to((0.24, -0.12, self.world.SAFE_Z), 0.0)
        self.world.flagged = True

    # --- interpolated Cartesian move via IK + drive ---------------------
    def _move_to(self, target_xyz, yaw: float) -> None:
        w = self.world
        # Jaws open -> approaching/grasping: must be tool-down. Jaws closed/wide
        # -> carrying: relax orientation so the short arm can reach lift height.
        dw = 0.5 if self._grip == OPEN else 0.12
        start = w.pinch_pos().copy()
        target = np.asarray(target_xyz, dtype=float)
        steps = max(4, int(np.linalg.norm(target - start) / CART_STEP_LEN))
        for k in range(1, steps + 1):
            pt = start + (target - start) * (k / steps)
            w.drive(w.ik(pt, down_weight=dw), self._grip, DRIVE_PER_STEP)
            if self.on_frame is not None and k % CAPTURE_EVERY == 0:
                self._capture()

    def _capture(self) -> None:
        if self.on_frame is not None:
            self.on_frame()


class ArmPerceiver:
    def __init__(self, world: ArmWorld):
        self.world = world
        self._renderer = None

    def observe(self) -> SceneState:
        import time
        w = self.world
        xy = w.product_xy()
        yaw = w.product_yaw()
        centroid = np.array([xy[0], xy[1], w.product_top_z()])
        yaws = [0.0] if w.kind == "cylinder" else [wrap_angle(yaw), wrap_angle(yaw + np.pi)]
        c = w.carton
        return SceneState(
            product_mask=np.zeros((2, 2), dtype=bool),
            product_depth=np.zeros((2, 2)),
            product_centroid=centroid,
            product_footprint=w.product_footprint(),
            product_yaw_candidates=yaws,
            carton_frame=frame((c["cx"], c["cy"], 0.0), 0.0),
            carton_opening=rect_aabb(c["cx"], c["cy"], c["hx"], c["hy"], 0.0),
            gripper_pose=frame(w.pinch_pos(), w.ee_yaw()),
            gripper_signal=w.jaw_gap(),
            holding=w.grasped(),
            perception_confidence=0.98,
            timestamp=time.time(),
        )

    def render(self, camera: str = "cell", height: int = 240, width: int = 320) -> np.ndarray:
        if self._renderer is None or self._renderer.height != height or self._renderer.width != width:
            self._renderer = mujoco.Renderer(self.world.model, height=height, width=width)
        self._renderer.update_scene(self.world.data, camera=camera)
        return self._renderer.render()


# --------------------------------------------------------------------------
# Demonstration on the real arm -> compiled skill.
# --------------------------------------------------------------------------
def record_arm_demo(world: ArmWorld):
    """Script an ideal pick-place on the REAL SO-101 and record the physics
    consequences (the pinch trajectory), so `compile_skill` is unchanged."""
    from ..geometry import frame as _frame
    from ..trace import DemonstrationTrace
    w = world
    px, py = (float(v) for v in w.product_xy())
    c = w.carton
    gz = w.hz                       # pinch at the product's vertical centre
    over = (px, py, gz + 0.06)      # tool-down reachable (~0.09)
    lifted = (px, py, 0.10)         # product bottom clears the carton wall
    travel = (c["cx"], c["cy"], 0.10)
    place = (c["cx"], c["cy"], gz + 0.02)
    withdrawn = (c["cx"], c["cy"], w.SAFE_Z)

    ts, poses, gcmd, gsig, masks, cents, foots, yaws = [], [], [], [], [], [], [], []
    t = [0.0]

    # `cmd` is the ABSTRACT gripper intent the compiler reads (0 open, 1 closed);
    # `grip` is the raw SO-101 ctrl driven into physics (OPEN/CLOSE/WIDE). They are
    # decoupled because the SO-101 ctrl is inverted (high = open).
    def snap(cmd: float) -> None:
        ts.append(t[0]); t[0] += 0.1
        poses.append(_frame(w.pinch_pos(), w.ee_yaw()))
        gcmd.append(cmd); gsig.append(w.jaw_gap())
        masks.append(np.zeros((2, 2), dtype=bool))
        cents.append(np.array([*w.product_xy(), w.product_top_z()]))
        foots.append(w.product_footprint())
        yaws.append(w.product_yaw())

    def seg(target, grip, cmd, sub=5) -> None:
        dw = 0.5 if grip == OPEN else 0.12  # tool-down to grasp, relaxed while carrying
        start = w.pinch_pos().copy()
        target = np.asarray(target, dtype=float)
        for s in range(1, sub + 1):
            w.drive(w.ik(start + (target - start) * (s / sub), down_weight=dw), grip, DRIVE_PER_STEP)
            snap(cmd)

    snap(0.0)
    seg(over, OPEN, 0.0, 3)
    seg((px, py, gz), OPEN, 0.0, 5)
    w.hold_grip(CLOSE, 700); snap(1.0)        # close on the product (friction grasp)
    seg(lifted, CLOSE, 1.0, 4)
    seg(travel, CLOSE, 1.0, 5)
    seg(place, CLOSE, 1.0, 4)
    w.hold_grip(WIDE, 300); snap(0.0)         # release
    seg(withdrawn, WIDE, 0.0, 3)

    return DemonstrationTrace(
        timestamps=np.array(ts),
        ee_poses=poses,
        gripper_command=np.array(gcmd),
        gripper_signal=np.array(gsig),
        product_masks=masks,
        product_centroids=cents,
        product_footprints=foots,
        product_yaws=np.array(yaws),
        carton_frame=_frame((c["cx"], c["cy"], 0.0), 0.0),
        table_height=0.0,
        camera_calibration_id="so101-cam-0",
        software_version="morrow-mk3-so101-0.0.1",
        meta={"kind": w.kind, "carton_rim_z": c["wall"], "embodiment": "so101"},
    )


def onboard_arm(kind: str, sku_id: str, n_demos: int = 1):
    from ..compile import compile_skill
    traces = [record_arm_demo(ArmWorld(kind)) for _ in range(n_demos)]
    return compile_skill(traces, sku_id)
