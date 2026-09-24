"""Scene geometry from configs/scene.json: zones, lines and lane directions in normalised coordinates."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely import contains_xy
from shapely import affinity
from shapely.geometry import LineString, Polygon

from src.config import ROOT, load_params
from src.registration import register_video, transform_points

log = logging.getLogger(__name__)

SCENE_PATH = ROOT / "configs" / "scene.json"


@dataclass
class Lane:
    """A carriageway section and its allowed directions of travel.

    ``directions`` is a ``(k, 2)`` array of unit vectors in pixel-aspect space;
    two-way sections list both directions.
    """

    id: str
    polygon: Polygon
    directions: np.ndarray


@dataclass
class Scene:
    """Zones and lines of the camera view, drawn on the reference frame.

    Polygons and lines are shapely objects in normalised [0, 1] coordinates.
    """

    carriageway: Polygon
    intersection: Polygon
    lanes: list[Lane]
    crossings: dict[str, Polygon]
    islands: dict[str, Polygon]
    stop_lines: dict[str, LineString]
    solid_lines: dict[str, LineString]
    stopping_zones: dict[str, Polygon] = field(default_factory=dict)
    traffic_lights: list[dict] = field(default_factory=list)

    def warped(self, warp: np.ndarray) -> Scene:
        """The scene mapped into a video's coordinates with a 2x3 affine warp (see src/registration.py)."""

        def poly(p: Polygon) -> Polygon:
            return Polygon(transform_points(warp, np.asarray(p.exterior.coords)))

        def line(ln: LineString) -> LineString:
            return LineString(transform_points(warp, np.asarray(ln.coords)))

        return Scene(
            carriageway=poly(self.carriageway),
            intersection=poly(self.intersection),
            lanes=[Lane(ln.id, poly(ln.polygon), ln.directions) for ln in self.lanes],
            crossings={k: poly(v) for k, v in self.crossings.items()},
            islands={k: poly(v) for k, v in self.islands.items()},
            stop_lines={k: line(v) for k, v in self.stop_lines.items()},
            solid_lines={k: line(v) for k, v in self.solid_lines.items()},
            stopping_zones={k: poly(v) for k, v in self.stopping_zones.items()},
            traffic_lights=[
                {**tl, "roi": transform_points(warp, np.asarray(tl["roi"]).reshape(2, 2)).ravel().tolist()}
                for tl in self.traffic_lights
            ],
        )

    def on_carriageway(self, xy: np.ndarray, inset: float = 0.0, aspect: float = 1.0) -> np.ndarray:
        """Boolean mask: which ``(N, 2)`` points lie on the carriageway and not on an island.

        With ``inset`` > 0 a point must be at least that far inside the carriageway
        edge (in frame widths; ``aspect`` = height / width makes the margin the
        same in every direction): used for pedestrians, whose anchor sits on the
        kerb while they wait on the pavement.
        """
        area = self.carriageway if inset <= 0 else shrink(self.carriageway, inset, aspect)
        inside = contains_xy(area, xy[:, 0], xy[:, 1])
        for island in self.islands.values():
            inside &= ~contains_xy(island, xy[:, 0], xy[:, 1])
        return inside

    def in_stopping_zone(self, xy: np.ndarray) -> np.ndarray:
        """Boolean mask: which ``(N, 2)`` points lie where vehicles routinely stop (bus stop, kerbside parking)."""
        mask = np.zeros(len(xy), dtype=bool)
        for zone in self.stopping_zones.values():
            mask |= contains_xy(zone, xy[:, 0], xy[:, 1])
        return mask

    def in_crossing(self, xy: np.ndarray) -> np.ndarray:
        """Boolean mask: which ``(N, 2)`` points lie on any pedestrian crossing."""
        mask = np.zeros(len(xy), dtype=bool)
        for crossing in self.crossings.values():
            mask |= contains_xy(crossing, xy[:, 0], xy[:, 1])
        return mask


def shrink(polygon: Polygon, margin: float, aspect: float) -> Polygon:
    """Polygon shrunk by ``margin`` frame widths, measured with y scaled by ``aspect``."""
    scaled = affinity.scale(polygon, xfact=1.0, yfact=aspect, origin=(0, 0))
    return affinity.scale(scaled.buffer(-margin), xfact=1.0, yfact=1.0 / aspect, origin=(0, 0))


def load_scene(path: Path = SCENE_PATH) -> Scene:
    """Load scene.json; comment keys (starting with ``_``) are ignored."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    def named(items: list[dict], key: str, kind) -> dict:
        return {item["id"]: kind(item[key]) for item in items}

    lanes = []
    for item in data["lanes"]:
        directions = np.asarray(item["directions"], dtype=np.float64)
        lanes.append(Lane(item["id"], Polygon(item["polygon"]), directions / np.linalg.norm(directions, axis=1, keepdims=True)))
    return Scene(
        carriageway=Polygon(data["carriageway"]),
        intersection=Polygon(data["intersection"]),
        lanes=lanes,
        crossings=named(data["crossings"], "polygon", Polygon),
        islands=named(data["islands"], "polygon", Polygon),
        stop_lines=named(data["stop_lines"], "line", LineString),
        solid_lines=named(data["solid_lines"], "line", LineString),
        stopping_zones=named(data.get("stopping_zones", []), "polygon", Polygon),
        traffic_lights=data["traffic_lights"],
    )


def scene_for_video(video_path: str, params: dict | None = None) -> Scene:
    """The scene aligned to this video; unshifted zones (with a warning) if the alignment is rejected."""
    params = params or load_params()
    scene = load_scene()
    reg = register_video(video_path, params["registration"])
    name = Path(video_path).name
    if not reg.ok:
        log.warning("%s: scene registration rejected (%s); using zones without shift", name, reg.reason)
        return scene
    shift = transform_points(reg.warp, np.array([[0.5, 0.5]]))[0] - 0.5
    log.info("%s: scene registered (cc=%.2f, centre shift dx=%+.3f dy=%+.3f)", name, reg.cc, shift[0], shift[1])
    return scene.warped(reg.warp)
