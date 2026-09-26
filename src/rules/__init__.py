"""Per-class event rules; each module covers a group of classes and implements ``base.Rule``."""
from __future__ import annotations

from src.postprocess import Segment
from src.rules.base import Rule, VideoContext
from src.rules.collisions import Accident, NearMiss
from src.rules.jaywalking import Jaywalking
from src.rules.lanes import IllegalTurn, SolidLineCrossing
from src.rules.signal_rules import RedLight, StopLineViolation
from src.rules.stopping import Congestion, StoppedVehicle
from src.rules.wrong_way import WrongWay
from src.rules.yielding import FailureToYield

RULES: list[Rule] = [StoppedVehicle(), Congestion(), WrongWay(), Jaywalking(), RedLight(), StopLineViolation(),
                     FailureToYield(), Accident(), NearMiss(), IllegalTurn(), SolidLineCrossing()]


def apply_rules(ctx: VideoContext) -> list[Segment]:
    """Raw segments of every rule, before post-processing (none without a single tracked object)."""
    if not len(ctx.features):
        return []
    return [segment for rule in RULES for segment in rule.apply(ctx)]
