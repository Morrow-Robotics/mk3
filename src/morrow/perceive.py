"""The perception boundary.

A `Perceiver` returns a fresh `SceneState` on demand. The executor re-perceives
before every candidate attempt and after every transition, so recovery always
reasons about the world as it actually is, not a stale snapshot.

In simulation this renders from the world model. On the bench it wraps an
overhead RGB-D camera plus a segmentation model (SAM-class) — that adapter is
deliberately not here yet; it is the one piece that needs the physical rig.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .scene import SceneState


@runtime_checkable
class Perceiver(Protocol):
    def observe(self) -> SceneState:
        """Capture and interpret the current scene."""
