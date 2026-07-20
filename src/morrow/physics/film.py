"""Render a physics pack to frames / an mp4 — the "sim of the arm doing the task".

Runs the FSM against MuJoCo while capturing rendered frames, then encodes them
with ffmpeg. This is what the dashboard shows next to each customer video.
"""

from __future__ import annotations

import subprocess

import numpy as np

from ..execute import run_skill
from .mj_perceive import MjPerceiver
from .mj_robot import MjRobot
from .record import onboard_mj
from .world import MjWorld


def capture_pack(kind: str = "box", seed: int = 0, px: float = -0.15, py: float = 0.0,
                 pyaw: float = 0.0, camera: str = "top", height: int = 240, width: int = 320):
    skill = onboard_mj(kind, kind)
    world = MjWorld(kind, px=px, py=py, pyaw=pyaw)
    per = MjPerceiver(world)
    per.render(camera, height, width)  # warm the renderer
    frames: list = []
    robot = MjRobot(world, on_frame=lambda: frames.append(per.render(camera, height, width)))
    result = run_skill(skill, robot, per, seed=seed)
    return frames, result, world


def capture_arm_pack(kind: str = "box", seed: int = 0, px: float = 0.27, py: float = -0.06,
                     pyaw: float = 0.0, camera: str = "cell", height: int = 260, width: int = 340):
    """Render the REAL SO-101 arm executing the compiled skill end-to-end."""
    from .arm import ArmPerceiver, ArmRobot, ArmWorld, onboard_arm
    skill = onboard_arm(kind, f"so101-{kind}")
    world = ArmWorld(kind, px=px, py=py, pyaw=pyaw)
    per = ArmPerceiver(world)
    per.render(camera, height, width)  # warm the renderer
    frames: list = []
    robot = ArmRobot(world, on_frame=lambda: frames.append(per.render(camera, height, width)))
    result = run_skill(skill, robot, per, seed=seed)
    return frames, result, world


def encode_mp4(frames: list, path: str, fps: int = 20) -> str:
    if not frames:
        raise ValueError("no frames to encode")
    h, w, _ = frames[0].shape
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-loglevel", "error", path],
        stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    return path
