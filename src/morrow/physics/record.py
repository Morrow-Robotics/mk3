"""Record a demonstration inside MuJoCo, on the same parallel-jaw embodiment we
execute on — so the compiled grasp height matches the jaws and `compile_skill`
stays unchanged. The scripted ideal pick-place drives the real physics (the box
is actually grasped by friction); every recorded field is a physics consequence.
"""

from __future__ import annotations

import numpy as np

from ..compile import compile_skill
from ..geometry import frame, wrap_angle
from ..trace import DemonstrationTrace
from .world import MjWorld

_MASK = np.zeros((2, 2), dtype=bool)
SOFTWARE_VERSION = "morrow-mk3-mujoco-0.0.1"


def record_mj_demo(world: MjWorld, sub_steps: int = 22) -> DemonstrationTrace:
    w = world
    px, py = (float(v) for v in w.product_xy())
    gpz = w.hz + 0.036  # palm height that makes the fingers straddle the product
    gyaw = w.product_yaw()
    c = w.carton
    over = (px, py, gpz + 0.09)
    grasp = (px, py, gpz)
    lifted = (px, py, 0.22)
    travel = (c["cx"], c["cy"], 0.22)
    place = (c["cx"], c["cy"], gpz)
    withdrawn = (c["cx"], c["cy"], w.SAFE_Z)

    ts, poses, gcmd, gsig, masks, cents, foots, yaws = [], [], [], [], [], [], [], []
    t = [0.0]

    def snap(cmd: float) -> None:
        ts.append(t[0]); t[0] += 0.1
        poses.append(frame(w.ee_pos(), w.ee_yaw()))
        gcmd.append(cmd); gsig.append(w.jaw_gap())
        masks.append(_MASK.copy())
        cents.append(np.array([*w.product_xy(), w.product_top_z()]))
        foots.append(w.product_footprint())
        yaws.append(w.product_yaw())

    def seg(target, yaw, cmd, snaps=3) -> None:
        start = w.data.mocap_pos[0].copy()
        yaw0 = w.ee_yaw()
        dyaw = wrap_angle(yaw - yaw0)
        for s in range(1, snaps + 1):
            f = s / snaps
            w.set_ee(start + (np.array(target) - start) * f, yaw0 + dyaw * f)
            w.step(sub_steps)
            snap(cmd)

    snap(0.0)
    seg(over, gyaw, 0.0, 2)
    seg(grasp, gyaw, 0.0, 3)
    w.set_jaws(0.035); w.step(140); snap(1.0)          # close on the product
    seg(lifted, gyaw, 1.0, 3)
    seg(travel, 0.0, 1.0, 3)
    seg(place, 0.0, 1.0, 3)
    w.set_jaws(0.0); w.step(90); snap(0.0)             # release
    seg(withdrawn, 0.0, 0.0, 2)

    return DemonstrationTrace(
        timestamps=np.array(ts),
        ee_poses=poses,
        gripper_command=np.array(gcmd),
        gripper_signal=np.array(gsig),
        product_masks=masks,
        product_centroids=cents,
        product_footprints=foots,
        product_yaws=np.array(yaws),
        carton_frame=frame((c["cx"], c["cy"], 0.0), 0.0),
        table_height=0.0,
        camera_calibration_id="mujoco-cam-0",
        software_version=SOFTWARE_VERSION,
        meta={"kind": w.kind, "carton_rim_z": c["wall"]},
    )


def onboard_mj(kind: str, sku_id: str, n_demos: int = 1):
    traces = [record_mj_demo(MjWorld(kind)) for _ in range(n_demos)]
    return compile_skill(traces, sku_id)
