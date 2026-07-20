"""pattern.py — honest packing-activity extraction from a real clip.

What this does NOT do: reliably count discrete items placed in a carton. That was
tried and measured — on cluttered consumer clips the motion signal inside the
carton region is a continuous stream of hand activity, and a peak count is
threshold-sensitive and does not correspond to a verifiable item count (9/7/4
"peaks" at confidence 0.34–0.49 on our three clips). Reporting that as "N items
packed" would be a fabrication.

What it DOES do, honestly: within the (SAM2-detected, operator-confirmed) carton
opening region it measures a per-frame motion signal — real classical CV, cv2
frame-differencing, deterministic. From it we report:

  * `active_fraction`  — how much of the clip shows packing activity in the carton
  * `n_events`         — a CANDIDATE reach/place-event count (peaks in the signal)
  * `confidence`       — how well-separated those peaks are from the noise floor,
                         in [0,1]; on consumer clips this is LOW by construction
  * `event_times_s`    — where the candidate events fall

The count is explicitly a low-confidence ESTIMATE meant to be operator-confirmed,
never an assertion. `confidence_label` says LOW / MEDIUM / HIGH so the dashboard
and any downstream pack can fall back to a human when it's shaky.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PackingProfile:
    n_events: int              # candidate reach/place events (operator-confirmable)
    confidence: float          # [0,1] peak separation from the noise floor
    confidence_label: str      # LOW / MEDIUM / HIGH
    active_fraction: float     # fraction of the clip active in the carton region
    duration_s: float
    event_times_s: list
    motion: list               # the normalized signal (for a sparkline)
    note: str

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["motion"] = [round(float(v), 3) for v in self.motion]
        return d


def carton_activity(clip: str, bbox_frac, stride: int = 2, cell=(160, 120)):
    """Per-(sampled)-frame mean frame-difference inside the carton region.
    Returns (timestamps_s, motion_signal, fps). Deterministic."""
    import cv2
    cap = cv2.VideoCapture(clip)
    W, H, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 25.0
    x0, y0 = int(bbox_frac[0] * W), int(bbox_frac[1] * H)
    x1, y1 = int(bbox_frac[2] * W), int(bbox_frac[3] * H)
    prev = None
    ts, sig, i = [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0 and y1 > y0 and x1 > x0:
            g = cv2.cvtColor(cv2.resize(fr[y0:y1, x0:x1], cell), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                sig.append(float(np.mean(np.abs(g - prev))))
                ts.append(i / fps)
            prev = g
        i += 1
    cap.release()
    return np.array(ts), np.array(sig), fps


def detect_events(ts: np.ndarray, motion: np.ndarray, thr_frac: float = 0.45,
                  min_gap_s: float = 0.3):
    """Peaks in the normalized motion signal → candidate events + a confidence.
    Confidence = mean peak prominence above the median, penalized when peaks are
    so dense they can't be told apart (continuous motion → low confidence)."""
    if len(motion) < 3:
        return [], 0.0
    s = (motion - motion.min()) / (np.ptp(motion) + 1e-9)
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.1
    min_gap = max(1, int(round(min_gap_s / max(dt, 1e-3))))
    peaks = []
    for i in range(1, len(s) - 1):
        if s[i] > thr_frac and s[i] >= s[i - 1] and s[i] >= s[i + 1]:
            if not peaks or i - peaks[-1] >= min_gap:
                peaks.append(i)
    if not peaks:
        return [], 0.0
    prominence = float(np.mean([s[p] for p in peaks]) - np.median(s))
    # density penalty: if peaks fill most slots, they're not separable events.
    density = len(peaks) / max(1, len(s) / min_gap)
    separation = float(np.clip(1.0 - density, 0.0, 1.0))
    confidence = float(np.clip(prominence * 2.0 * (0.5 + 0.5 * separation), 0.0, 1.0))
    events = [{"t": float(ts[p]), "strength": round(float(s[p]), 3)} for p in peaks]
    return events, round(confidence, 2)


def _label(conf: float) -> str:
    return "HIGH" if conf >= 0.66 else ("MEDIUM" if conf >= 0.4 else "LOW")


def packing_profile(clip: str, carton_bbox_frac, stride: int = 2) -> PackingProfile:
    ts, motion, fps = carton_activity(clip, carton_bbox_frac, stride=stride)
    if len(motion) == 0:
        return PackingProfile(0, 0.0, "LOW", 0.0, 0.0, [], [], "no frames read")
    baseline = float(np.median(motion))
    active_fraction = float(np.mean(motion > baseline * 1.3))
    events, confidence = detect_events(ts, motion)
    norm = ((motion - motion.min()) / (np.ptp(motion) + 1e-9)).tolist()
    return PackingProfile(
        n_events=len(events),
        confidence=confidence,
        confidence_label=_label(confidence),
        active_fraction=round(active_fraction, 2),
        duration_s=round(float(ts[-1]), 1),
        event_times_s=[round(e["t"], 1) for e in events],
        motion=norm,
        note=("carton-region activity (cv2 frame-diff). The event count is a "
              "LOW-confidence estimate on occluded consumer clips — operator-confirmable, "
              "not a verified item count."),
    )


def sparkline_png_b64(motion, events_idx=None, width: int = 340, height: int = 60) -> str:
    """Render the motion signal as a small sparkline PNG (base64) — proof artefact."""
    import base64
    import os
    import tempfile

    import cv2
    img = np.full((height, width, 3), 22, np.uint8)
    m = np.asarray(motion, dtype=np.float32)
    if len(m) >= 2:
        xs = np.linspace(0, width - 1, len(m)).astype(int)
        ys = (height - 4 - m * (height - 10)).astype(int)
        for i in range(len(m) - 1):
            cv2.line(img, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), (170, 200, 90), 1)
        for e in (events_idx or []):
            if 0 <= e < len(m):
                cv2.circle(img, (int(xs[e]), int(ys[e])), 3, (60, 160, 255), -1)
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(path, img)
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    finally:
        os.unlink(path)
