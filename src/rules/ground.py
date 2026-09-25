"""Scene zones on the road plane (``risk.ground_plane``), in metres: margins that mean the same everywhere."""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Polygon

from src.risk import ground_plane
from src.rules.base import VideoContext


def to_ground(ctx: VideoContext, polygon: Polygon) -> Polygon:
    """A scene polygon on the road plane, in metres (the mapping keeps straight lines straight)."""
    g = ctx.params["risk"]["ground"]
    coords = np.asarray(polygon.exterior.coords)
    return Polygon(ground_plane(coords[:, 0], coords[:, 1], g, ctx.aspect) * g["box_width_m"])


def ground_points(ctx: VideoContext, rows: np.ndarray) -> np.ndarray:
    """Shapely points of the rows' anchors on the road plane, in metres."""
    f = ctx.features
    g = ctx.params["risk"]["ground"]
    pos = ground_plane(f["x"][rows], f["y"][rows], g, ctx.aspect) * g["box_width_m"]
    return shapely.points(pos[:, 0], pos[:, 1])


def crossing_distance_m(ctx: VideoContext, rows: np.ndarray) -> np.ndarray:
    """Distance of the rows' anchors to the nearest pedestrian crossing on the road plane (0 on one), metres."""
    crossings = shapely.union_all([to_ground(ctx, c) for c in ctx.scene.crossings.values()])
    return shapely.distance(crossings, ground_points(ctx, rows))


def road_depth_m(ctx: VideoContext, rows: np.ndarray) -> np.ndarray:
    """How far inside the carriageway (islands cut out) the rows' anchors are on the road plane, metres; 0 outside."""
    road = to_ground(ctx, ctx.scene.carriageway)
    for island in ctx.scene.islands.values():
        road = road.difference(to_ground(ctx, island))
    points = ground_points(ctx, rows)
    return np.where(shapely.contains(road, points), shapely.distance(road.boundary, points), 0.0)
