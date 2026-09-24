"""Per-class event rules; each module covers a group of classes and implements ``base.Rule``."""
from __future__ import annotations

from src.postprocess import Segment
from src.rules.base import Rule, VideoContext
from src.rules.jaywalking import Jaywalking
from src.rules.stopping import Congestion, StoppedVehicle
from src.rules.wrong_way import WrongWay

RULES: list[Rule] = [StoppedVehicle(), Congestion(), WrongWay(), Jaywalking()]


def apply_rules(ctx: VideoContext) -> list[Segment]:
    """Raw segments of every rule, before post-processing."""
    return [segment for rule in RULES for segment in rule.apply(ctx)]
