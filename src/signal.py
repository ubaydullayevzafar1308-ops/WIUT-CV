"""Traffic-light phase of the avenue_near approach from the signal heads visible in the frame.

Two heads face the camera and switch together (checked on all samples):
the pedestrian head (red man top, green man bottom) is the primary source, the
3-lamp vehicle head on the median confirms it and fills in when the pedestrian
head is unreadable. At the end of green the vehicle phase trails the
pedestrian one by a fixed clearance (3 s green + 3 s amber, measured).

Lamps are read by chroma, compared inside one head, never against absolute
brightness: lit lamps reach ~57 levels of chroma in direct sun and ~145 at dusk,
while an unlit green lamp can show ~36 at dusk from the blue ambient light.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.scene import Scene
from src.video import read_frames

RED, RED_AMBER, AMBER, GREEN, UNKNOWN = "red", "red_amber", "amber", "green", "unknown"
PHASES = (RED, RED_AMBER, AMBER, GREEN, UNKNOWN)
PED_HEAD, VEH_HEAD = "ped_avenue_near", "veh_median"
LAMP_PERCENTILE = 95        # a lamp's score is a high percentile of its chroma: robust to ROI slack


@dataclass
class SignalTimeline:
    """Signal readings of one video at ``sample_fps``.

    ``ped``/``veh`` are the per-sample readings of each head, ``phase`` the
    fused and smoothed phase (one of PHASES).
    """

    t: np.ndarray
    ped: np.ndarray
    veh: np.ndarray
    phase: np.ndarray

    def phase_at(self, t_sec: float | np.ndarray) -> np.ndarray:
        """Phase of the latest sample at or before ``t_sec`` (``unknown`` before the first sample)."""
        idx = np.searchsorted(self.t, t_sec, side="right") - 1
        return np.where(idx >= 0, self.phase[np.clip(idx, 0, None)], UNKNOWN)


def _crop(frame: np.ndarray, roi: list[float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    return frame[int(y1 * h):int(np.ceil(y2 * h)), int(x1 * w):int(np.ceil(x2 * w))]


def lamp_scores(crop: np.ndarray, kinds: tuple[str, ...]) -> np.ndarray:
    """Chroma score of each lamp; the head is split into equal horizontal bands, top to bottom.

    red: R - max(G, B); amber: min(R, G) - B; green (cyan LEDs): (G + B) / 2 - R.
    """
    f = crop.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    chroma = {"red": r - np.maximum(g, b), "amber": np.minimum(r, g) - b, "green": (g + b) / 2 - r}
    bands = np.array_split(np.arange(crop.shape[0]), len(kinds))
    return np.array([np.percentile(chroma[k][band], LAMP_PERCENTILE) for k, band in zip(kinds, bands)])


def pedestrian_state(crop: np.ndarray, params: dict[str, Any]) -> str:
    """red / green / unknown from the pedestrian head (red man top, green man bottom)."""
    if crop.size == 0:
        return UNKNOWN
    red, green = lamp_scores(crop, ("red", "green"))
    if max(red, green) < params["min_lit"]:
        return UNKNOWN
    contrast = (red - green) / (abs(red) + abs(green) + 1e-6)
    if contrast > params["ped_margin"]:
        return RED
    if contrast < -params["ped_margin"]:
        return GREEN
    return UNKNOWN


def vehicle_state(crop: np.ndarray, params: dict[str, Any]) -> str:
    """red / red_amber / amber / green / unknown from the 3-lamp vehicle head."""
    if crop.size == 0:
        return UNKNOWN
    scores = lamp_scores(crop, ("red", "amber", "green"))
    top = scores.max()
    if top < params["min_lit"]:
        return UNKNOWN
    red, amber, green = scores >= max(params["min_lit"], params["veh_lit_fraction"] * top)
    if green and not (red or amber):
        return GREEN
    if red and amber:
        return RED_AMBER
    if red:
        return RED
    if amber:
        return AMBER
    return UNKNOWN


def fuse(ped: str, veh: str) -> str:
    """Pedestrian head decides red/green; the vehicle head refines red (amber, red+amber) and fills gaps."""
    if ped == RED and veh in (RED_AMBER, AMBER):
        return veh
    if ped in (RED, GREEN):
        return ped
    return veh


def vehicle_clearance(t: np.ndarray, phase: np.ndarray, veh: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Vehicle green and amber after the green man goes red.

    On every sample the vehicle phase trails the pedestrian one: after the
    green man goes red, vehicles keep (flashing) green for
    ``vehicle_green_after_ped_sec`` and amber for ``vehicle_amber_sec``, then
    red. Vehicle-head readings take precedence where readable; in direct sun
    flashing green and amber often read as unknown, and the timing fills them
    in. Green never follows amber, and a vehicle red ends the clearance early.
    """
    green_sec, amber_sec = params["vehicle_green_after_ped_sec"], params["vehicle_amber_sec"]
    out = phase.copy()
    onsets = np.flatnonzero((phase[:-1] == GREEN) & (phase[1:] == RED)) + 1
    for i in onsets:
        j, amber_started = i, False
        while j < len(t) and t[j] - t[i] < green_sec + amber_sec and veh[j] != RED and phase[j] == RED:
            amber_started |= veh[j] == AMBER or (veh[j] != GREEN and t[j] - t[i] >= green_sec)
            out[j] = AMBER if amber_started else GREEN
            j += 1
    return out


def smooth(t: np.ndarray, states: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Majority vote over a sliding window, then carry the last phase through short unknown gaps."""
    half = params["smooth_window_sec"] / 2
    codes = np.array([PHASES.index(s) for s in states])
    voted = codes.copy()
    for i, ti in enumerate(t):
        window = codes[(t >= ti - half) & (t <= ti + half)]
        known = window[window != PHASES.index(UNKNOWN)]
        if len(known):
            counts = np.bincount(known, minlength=len(PHASES))
            voted[i] = counts.argmax() if counts.max() * 2 > len(window) else codes[i]
    out = np.array([PHASES[c] for c in voted], dtype=object)
    last, last_t = UNKNOWN, -np.inf
    for i, ti in enumerate(t):
        if out[i] != UNKNOWN:
            last, last_t = out[i], ti
        elif ti - last_t <= params["max_hold_sec"]:
            out[i] = last
    return out


def read_timeline(video_path: str, scene: Scene, params: dict[str, Any]) -> SignalTimeline:
    """Read both heads at ``sample_fps`` and fuse them into a smoothed phase timeline.

    ``scene`` must already be aligned to this video (``scene_for_video``);
    ``params`` is the full parameter dict.
    """
    sp, vp = params["signal"], params["video"]
    rois = {light["id"]: light["roi"] for light in scene.traffic_lights}
    frames = read_frames(
        video_path,
        stride=vp["stride"],
        target_width=sp["frame_width"],
        threads=vp["decoder_threads"],
        skip_nonref=vp["skip_nonref"],
        interpolation=vp["interpolation"],
        skip_check_frames=vp["skip_check_frames"],
    )
    t, ped, veh = [], [], []
    next_t = 0.0
    for _, t_sec, bgr in frames:
        if t_sec < next_t:
            continue
        next_t = t_sec + 1.0 / sp["sample_fps"]
        t.append(t_sec)
        ped.append(pedestrian_state(_crop(bgr, rois[PED_HEAD]), sp))
        veh.append(vehicle_state(_crop(bgr, rois[VEH_HEAD]), sp))
    t_arr, ped_arr, veh_arr = np.array(t), np.array(ped, dtype=object), np.array(veh, dtype=object)
    fused = np.array([fuse(p, v) for p, v in zip(ped_arr, veh_arr)], dtype=object)
    phase = vehicle_clearance(t_arr, smooth(t_arr, fused, sp), veh_arr, sp)
    return SignalTimeline(t=t_arr, ped=ped_arr, veh=veh_arr, phase=phase)
