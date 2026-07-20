"""MJCF for a packing cell: a top-down parallel-jaw gripper, product, and carton.

Real contact physics via MuJoCo. The gripper is a palm welded to a mocap body —
setting the mocap pose drags the gripper in Cartesian space, which is exactly
what the SkillProgram emits (Cartesian EE waypoints). Two fingers on slide
joints close under position actuators; the object is held by friction, not by an
`attached` flag. This is the standard LeRobot parallel jaw — no suction.

When a real SO-101 MJCF is dropped in, the same Cartesian waypoints drive it via
IK; nothing above the Robot boundary changes.
"""

from __future__ import annotations

CARTON = dict(cx=0.20, cy=0.0, hx=0.09, hy=0.09, wall=0.05, thick=0.006)


def _carton_walls(c: dict) -> str:
    cx, cy, hx, hy, w, t = c["cx"], c["cy"], c["hx"], c["hy"], c["wall"], c["thick"]
    z = w / 2
    parts = [
        f'<geom type="box" pos="{cx} {cy - hy - t} {z}" size="{hx + t} {t} {w/2}" rgba="0.82 0.70 0.52 1"/>',
        f'<geom type="box" pos="{cx} {cy + hy + t} {z}" size="{hx + t} {t} {w/2}" rgba="0.82 0.70 0.52 1"/>',
        f'<geom type="box" pos="{cx - hx - t} {cy} {z}" size="{t} {hy} {w/2}" rgba="0.82 0.70 0.52 1"/>',
        f'<geom type="box" pos="{cx + hx + t} {cy} {z}" size="{t} {hy} {w/2}" rgba="0.82 0.70 0.52 1"/>',
    ]
    return "\n      ".join(parts)


def _product_geom(kind: str, hx: float, hy: float, hz: float) -> str:
    if kind == "cylinder":
        return f'<geom name="product" type="cylinder" size="{hx} {hz}" mass="0.05" ' \
               'friction="1.2 0.05 0.001" rgba="0.16 0.64 0.64 1"/>'
    rgba = "0.10 0.45 0.90 1" if kind == "box" else "0.90 0.55 0.10 1"
    return f'<geom name="product" type="box" size="{hx} {hy} {hz}" mass="0.05" ' \
           f'friction="1.2 0.05 0.001" rgba="{rgba}"/>'


def build_mjcf(kind: str = "box", hx: float = 0.03, hy: float = 0.03, hz: float = 0.03,
               px: float = -0.15, py: float = 0.0, pyaw: float = 0.0,
               carton: dict | None = None) -> str:
    c = carton or CARTON
    import numpy as np
    qw, qz = float(np.cos(pyaw / 2)), float(np.sin(pyaw / 2))
    return f"""<mujoco model="morrow_cell">
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <default>
    <geom solimp="0.97 0.995 0.001" solref="0.004 1"/>
  </default>
  <worldbody>
    <light pos="0 0 1.2" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="table" type="plane" size="0 0 0.05" pos="0 0 0" rgba="0.93 0.93 0.94 1"/>
    <camera name="top" pos="0.02 0 0.7" xyaxes="1 0 0 0 1 0"/>
    <camera name="side" pos="0.02 -0.7 0.35" xyaxes="1 0 0 0 0.5 0.9"/>
    {_carton_walls(c)}
    <body name="product" pos="{px} {py} {hz}" quat="{qw} 0 0 {qz}">
      <freejoint/>
      {_product_geom(kind, hx, hy, hz)}
    </body>
    <body name="target" mocap="true" pos="{px} {py} 0.30"/>
    <body name="palm" pos="{px} {py} 0.30">
      <freejoint/>
      <geom type="box" size="0.02 0.045 0.012" mass="0.15" rgba="0.3 0.3 0.32 1"/>
      <body name="left_finger" pos="-0.045 0 -0.035">
        <joint name="lf" type="slide" axis="1 0 0" range="0 0.035"/>
        <geom name="lfg" type="box" size="0.006 0.02 0.03" mass="0.02" friction="1.5 0.1 0.001" rgba="0.2 0.2 0.22 1"/>
      </body>
      <body name="right_finger" pos="0.045 0 -0.035">
        <joint name="rf" type="slide" axis="-1 0 0" range="0 0.035"/>
        <geom name="rfg" type="box" size="0.006 0.02 0.03" mass="0.02" friction="1.5 0.1 0.001" rgba="0.2 0.2 0.22 1"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <weld name="mount" body1="palm" body2="target" solref="0.01 1" solimp="0.95 0.99 0.001"/>
  </equality>
  <actuator>
    <position name="lf_act" joint="lf" kp="80" ctrlrange="0 0.035" forcerange="-40 40"/>
    <position name="rf_act" joint="rf" kp="80" ctrlrange="0 0.035" forcerange="-40 40"/>
  </actuator>
</mujoco>"""
