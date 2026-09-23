"""Signal-head reading, fusion, smoothing and the vehicle clearance after the green man goes red."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import ROOT, load_params
from src.scene import scene_for_video
from src.signal import (AMBER, GREEN, RED, RED_AMBER, UNKNOWN, fuse, pedestrian_state, read_timeline, smooth,
                        vehicle_clearance, vehicle_state)

PARAMS = load_params()
SP = PARAMS["signal"]

# BGR of a lit lamp and of the unlit housing, as measured: direct sun is washed out, dusk is saturated
LIGHTING = {
    "sun": {"red": (60, 55, 120), "amber": (50, 110, 120), "green": (120, 110, 70), "off": (60, 62, 64)},
    "dusk": {"red": (40, 40, 230), "amber": (20, 150, 230), "green": (200, 210, 60), "off": (70, 50, 30)},
}


def head(lit: list[str | None], lighting: str, lamp_px: int = 10, width: int = 12) -> np.ndarray:
    """A signal head crop: one band per lamp, top to bottom; None = unlit."""
    colors = LIGHTING[lighting]
    bands = [np.full((lamp_px, width, 3), colors[kind] if kind else colors["off"], np.uint8) for kind in lit]
    return np.vstack(bands)


@pytest.mark.parametrize("lighting", ["sun", "dusk"])
def test_pedestrian_head(lighting):
    assert pedestrian_state(head(["red", None], lighting), SP) == RED
    assert pedestrian_state(head([None, "green"], lighting), SP) == GREEN
    assert pedestrian_state(np.zeros((0, 0, 3), np.uint8), SP) == UNKNOWN


def test_unlit_head():
    assert pedestrian_state(head([None, None], "sun"), SP) == UNKNOWN
    # at dusk the bluish housing has green chroma, so a dark head reads green (only happens in the off
    # half of the flashing green man, where green is right); it must never read red
    assert pedestrian_state(head([None, None], "dusk"), SP) != RED


@pytest.mark.parametrize("lighting", ["sun", "dusk"])
@pytest.mark.parametrize(("lamps", "expected"), [
    (["red", None, None], RED),
    (["red", "amber", None], RED_AMBER),
    ([None, "amber", None], AMBER),
    ([None, None, "green"], GREEN),
])
def test_vehicle_head(lighting, lamps, expected):
    assert vehicle_state(head(lamps, lighting), SP) == expected


def test_fuse_prefers_pedestrian_head():
    assert fuse(GREEN, RED) == GREEN
    assert fuse(RED, RED_AMBER) == RED_AMBER
    assert fuse(UNKNOWN, GREEN) == GREEN
    assert fuse(UNKNOWN, UNKNOWN) == UNKNOWN


def test_smooth_removes_blips_and_bridges_short_gaps():
    t = np.arange(0, 10, 0.5)
    states = np.array([RED] * 6 + [GREEN] + [RED] * 5 + [UNKNOWN] * 4 + [RED] * 4, dtype=object)
    out = smooth(t, states, SP)
    assert set(out) == {RED}


def test_vehicle_clearance_after_green_man():
    t = np.arange(0, 12, 0.5)
    phase = np.array([GREEN] * 4 + [RED] * 20, dtype=object)          # green man goes red at t = 2 s
    veh = np.array([GREEN] * 4 + [UNKNOWN, GREEN] * 3 + [UNKNOWN] * 6 + [RED] * 8, dtype=object)
    out = vehicle_clearance(t, phase, veh, SP)
    green_end = 2 + SP["vehicle_green_after_ped_sec"]
    assert all(out[(t >= 2) & (t < green_end)] == GREEN)
    assert all(out[(t >= green_end) & (t < green_end + SP["vehicle_amber_sec"])] == AMBER)
    assert all(out[t >= green_end + SP["vehicle_amber_sec"]] == RED)


def test_clearance_never_returns_from_amber_to_green():
    t = np.arange(0, 8, 0.5)
    phase = np.array([GREEN] * 2 + [RED] * 14, dtype=object)
    veh = np.array([GREEN] * 2 + [AMBER, GREEN, AMBER] + [UNKNOWN] * 11, dtype=object)
    out = vehicle_clearance(t, phase, veh, SP)
    after = list(out[2:])
    assert GREEN not in after[after.index(AMBER):]


def test_timeline_on_dusk_sample():
    path = ROOT / "samples" / "C3905.MP4"
    if not path.exists():
        pytest.skip("C3905.MP4 not available (videos are not in git)")
    timeline = read_timeline(str(path), scene_for_video(str(path), PARAMS), PARAMS)
    assert np.mean(timeline.phase == UNKNOWN) < 0.02
    # phases read off the video by eye: red, green from ~34 s, amber ~72-75 s, red, green from ~115 s
    for t_sec, expected in [(15, RED), (50, GREEN), (73.5, AMBER), (95, RED), (122, GREEN)]:
        assert timeline.phase_at(t_sec) == expected, t_sec
