"""Record a demonstration in simulation.

On the bench this is a human teleoperating the leader arm. Here it is a scripted
ideal pack, executed on the *real* SimRobot so the recorded gripper signal,
attachment, and rising centroid are genuine consequences of the motion — not
hand-written answers. The compiler then has to recover the phases from this
trace exactly as it would from a noisy human demo.
"""

from __future__ import annotations

import numpy as np

from ..geometry import Transform, frame, translation
from ..trace import DemonstrationTrace
from .sim_robot import SimRobot
from .sim_perceive import SimPerceiver
from .world import World

APPROACH_H = 0.08  # tool height above product top at approach
LIFT_Z = 0.22  # absolute tool z when lifted clear
SOFTWARE_VERSION = "morrow-mk3-sim-0.0.1"


def _interp(a: Transform, b: Transform, steps: int) -> list[Transform]:
    """Linear interpolation of translation + yaw between two top-down poses."""
    from ..geometry import yaw_of, wrap_angle

    pa, pb = translation(a), translation(b)
    ya, yb = yaw_of(a), yaw_of(b)
    dyaw = wrap_angle(yb - ya)
    out = []
    for k in range(1, steps + 1):
        f = k / steps
        pos = pa + f * (pb - pa)
        out.append(frame(pos, ya + f * dyaw))
    return out


def record_demo(world: World, robot: SimRobot, perceiver: SimPerceiver,
                steps_per_segment: int = 4, dt: float = 0.1) -> DemonstrationTrace:
    p = world.product
    c = world.carton
    top = world.product_top_z()
    y = p.yaw

    over = frame((p.cx, p.cy, top + APPROACH_H), y)
    grasp = frame((p.cx, p.cy, top), y)
    lifted = frame((p.cx, p.cy, LIFT_Z), y)
    travel = frame((c.cx, c.cy, c.wall_z + 0.10), y)
    place = frame((c.cx, c.cy, c.floor_z + p.height), y)
    withdrawn = frame((c.cx, c.cy, world.SAFE_Z), y)

    ts, poses, gcmd, gsig = [], [], [], []
    masks, cents, foots, yaws = [], [], [], []
    t = [0.0]

    def snap(cmd: float) -> None:
        s = perceiver.observe()
        ts.append(t[0])
        poses.append(robot.get_ee_pose())
        gcmd.append(cmd)
        gsig.append(robot.gripper_signal())
        masks.append(s.product_mask.copy())
        cents.append(s.product_centroid.copy())
        foots.append(s.product_footprint.copy())
        yaws.append(p.yaw)
        t[0] += dt

    # start pose
    robot.follow([over])
    snap(0.0)
    # approach -> grasp
    for wp in _interp(over, grasp, steps_per_segment):
        robot.follow([wp]); snap(0.0)
    # seal
    robot.engage(); snap(1.0)
    # lift
    for wp in _interp(grasp, lifted, steps_per_segment):
        robot.follow([wp]); snap(1.0)
    # transport
    for wp in _interp(lifted, travel, steps_per_segment):
        robot.follow([wp]); snap(1.0)
    # descend to place
    for wp in _interp(travel, place, steps_per_segment):
        robot.follow([wp]); snap(1.0)
    # release
    robot.release(); snap(0.0)
    # withdraw
    for wp in _interp(place, withdrawn, steps_per_segment):
        robot.follow([wp]); snap(0.0)

    return DemonstrationTrace(
        timestamps=np.array(ts),
        ee_poses=poses,
        gripper_command=np.array(gcmd),
        gripper_signal=np.array(gsig),
        product_masks=masks,
        product_centroids=cents,
        product_footprints=foots,
        product_yaws=np.array(yaws),
        carton_frame=c.pose(),
        table_height=world.table_height,
        camera_calibration_id="sim-cam-0",
        software_version=SOFTWARE_VERSION,
        meta={"kind": p.kind, "carton_rim_z": c.wall_z},
    )
