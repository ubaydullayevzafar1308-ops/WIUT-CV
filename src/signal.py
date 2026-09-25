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

RED, RED_AMBER, AMBER, GREEN, UNKNOWN = "red", "red_amber", "amber", "green", "unknown"
PHASES = (RED, RED_AMBER, AMBER, GREEN, UNKNOWN)
PED_HEAD, VEH_HEAD = "ped_avenue_near", "veh_median"
LAMP_PERCENTILE = 95        # a lamp's score is a high percentile of its chroma: robust to ROI slack


@dataclass
class SignalTimeline:
    """Signal readings of one video at ``sample_fps``, taken from the frames the tracker decodes.

    ``ped``/``veh`` are the per-sample readings of each head (``veh`` debounced,
    see ``debounce``), ``phase`` the
    fused and smoothed vehicle phase of avenue_near (one of PHASES), ``walk`` the
    same before the vehicle clearance: green exactly while the green man of the
    pedestrian head facing the camera is lit. That head is green while the avenue
    traffic moves: it is the signal of the pedestrians crossing the side street.
    """

    t: np.ndarray
    ped: np.ndarray
    veh: np.ndarray
    phase: np.ndarray
    walk: np.ndarray

    def phase_at(self, t_sec: float | np.ndarray) -> np.ndarray:
        """Phase of the latest sample at or before ``t_sec`` (``unknown`` before the first sample)."""
        idx = np.searchsorted(self.t, t_sec, side="right") - 1
        return np.where(idx >= 0, self.phase[np.clip(idx, 0, None)], UNKNOWN)

    def walk_at(self, t_sec: float | np.ndarray) -> np.ndarray:
        """Pedestrian (side street) phase of the latest sample at or before ``t_sec``."""
        idx = np.searchsorted(self.t, t_sec, side="right") - 1
        return np.where(idx >= 0, self.walk[np.clip(idx, 0, None)], UNKNOWN)


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


def debounce(t: np.ndarray, states: np.ndarray, hold_sec: float) -> np.ndarray:
    """Accept a new reading only if it holds for ``hold_sec``; shorter blips take the current state.

    Unknown readings stay unknown and do not interrupt a run (in direct sun single
    samples misread a lamp: a lone "red" amid green readings would end the vehicle
    clearance early).
    """
    out = np.array(states, dtype=object)
    known = np.flatnonzero(out != UNKNOWN)
    current = None
    for n, i in enumerate(known):
        state = states[i]
        if current is None or state == current:
            current = state
            continue
        k = n
        while k + 1 < len(known) and states[known[k + 1]] == state:
            k += 1
        if t[known[k]] - t[i] >= hold_sec:
            current = state
        else:
            out[i] = current
    return out


def read_heads(frame: np.ndarray, rois: dict[str, list[float]], params: dict[str, Any]) -> tuple[str, str]:
    """(pedestrian, vehicle) readings of one BGR frame of any size; ``params`` is the ``signal`` section."""
    return pedestrian_state(_crop(frame, rois[PED_HEAD]), params), vehicle_state(_crop(frame, rois[VEH_HEAD]), params)


def timeline_from_samples(t: np.ndarray, ped: np.ndarray, veh: np.ndarray, params: dict[str, Any]) -> SignalTimeline:
    """Fuse and smooth per-sample head readings (collected by the tracking loop) into a phase timeline."""
    ped = np.asarray(ped, dtype=object)
    veh = debounce(np.asarray(t), np.asarray(veh, dtype=object), params["veh_hold_sec"])
    fused = np.array([fuse(p, v) for p, v in zip(ped, veh)], dtype=object)
    walk = smooth(t, fused, params)
    phase = vehicle_clearance(t, walk, veh, params)
    return SignalTimeline(t=np.asarray(t), ped=ped, veh=veh, phase=phase, walk=walk)
