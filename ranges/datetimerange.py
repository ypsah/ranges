"""
DatetimeRange provides a range-like interface (and more) over intervals of datetimes

Like python's built-in range(), it has a start, a stop, and a step attribute, like range() it implements the Sequence
interface.
Unlike python's built-in range(), DatetimeRange also implements the Set interface. The implementation has limitations:
not every pair of DatetimeRange can be intersected, or unioned, but membership testing does runs in O(1) time
complexity.
"""

from __future__ import annotations

from collections.abc import Sequence, Set
from datetime import datetime, timedelta
from enum import auto, Enum, unique
from functools import cached_property, total_ordering


@unique
class _RangeOrdering(Enum):
    """
    Helper class to describe whether a DatetimeRange is ascending or descending
    """
    ASCENDING = auto()
    DESCENDING = auto()


@total_ordering
class DatetimeRange(Sequence[datetime], Set[datetime]):
    """
    A range-like interface over intervals of datetimes that implements the Set interface on a best-effort basis
    """

    def __init__(self, start: datetime, stop: datetime, step: timedelta, *args, **kwargs):
        if step == timedelta():
            raise ValueError("step must be non-null")

        super().__init__(*args, **kwargs)
        self._start = start
        self._stop = stop
        self._step = step
        self._len = max(0, (stop - start) // step)

    @property
    def start(self) -> datetime:
        """
        Left boundary (inclusive) of the range
        """
        return self._start

    @property
    def stop(self) -> datetime:
        """
        Right boundary (exclusive) of the range
        """
        return self._stop

    @property
    def step(self) -> timedelta:
        """
        Increment between two consecutive elements of the range
        """
        return self._step

    @cached_property
    def _ordering(self) -> _RangeOrdering:
        """
        Internal helper to keep track of whether the range is ascending or descending
        """
        return _RangeOrdering.ASCENDING if self.step > timedelta() else _RangeOrdering.DESCENDING

    def __len__(self) -> int:
        return self._len

    def __hash__(self) -> int:
        match len(self):
            case 0:
                return 0
            case 1:
                return hash(self.start)
        return hash(self.start) ^ hash(self[-1]) ^ hash(self.step)

    def __eq__(self, other: DatetimeRange) -> bool:
        if not isinstance(other, DatetimeRange):
            return NotImplemented

        if len(self) != len(other):
            return False

        match len(self):
            case 0:
                return True
            case 1:
                return self.start == other.start
        return self.start == other.start and self[-1] == other[-1]

    def __getitem__(self, i: int) -> datetime:
        if abs(i) >= len(self):
            raise IndexError("index out of range")
        if i < 0:
            return self[len(self) - i]
        return self.start + i * self.step

    def __contains__(self, dt: datetime) -> bool:
        if self._ordering is _RangeOrdering.ASCENDING:
            if not self.start <= dt < self.stop:
                return False
        else:
            if not self.start >= dt > self.stop:
                return False
        return not (dt - self.start).total_seconds() % self.step.total_seconds()

    def __and__(self, other: DatetimeRange) -> DatetimeRange:
        if not isinstance(other, DatetimeRange):
            return NotImplemented

        if self._ordering is not other._ordering:
            raise ValueError("Cannot intersect two DatetimeRanges with different orderings")
        steps = (self.step, other.step)
        if max(steps, key=abs) % min(steps, key=abs):
            raise ValueError("Cannot intersect two DatetimeRanges whose steps are not multiples of one another")

        starts = (self.start, other.start)
        stops = (self.stop, other.stop)
        if self._ordering is _RangeOrdering.ASCENDING:
            return DatetimeRange(max(starts), min(stops), max(steps))
        return DatetimeRange(min(starts), max(stops), min(steps))

    def __or__(self, other: DatetimeRange) -> DatetimeRange:
        if not isinstance(other, DatetimeRange):
            return NotImplemented

        steps = (self.step, other.step)
        if max(steps, key=abs) % min(steps, key=abs):
            raise ValueError("Cannot merge two DatetimeRanges whose steps are not at least multiples of one another")
        if self.step != other.step:
            if self.start != other.start or self.stop != other.stop:
                raise ValueError("Cannot merge two DatetimeRanges whose steps are different and boundaries don't match")
            return DatetimeRange(self.start, self.stop, min(steps, key=abs))
        starts = (self.start, other.start)
        stops = (self.stop, other.stop)
        if min(stops) < max(starts) if self._ordering is _RangeOrdering.ASCENDING else min(starts) < max(stops):
            raise ValueError("Cannot merge two non-overlapping and non-contiguous DatetimeRanges")

        if self._ordering is _RangeOrdering.ASCENDING:
            return DatetimeRange(min(starts), max(stops), self.step)
        return DatetimeRange(max(starts), min(stops), self.step)

    def isdisjoint(self, other: DatetimeRange) -> bool:
        return not self & other

    def __le__(self, other: DatetimeRange) -> bool:
        if not isinstance(other, DatetimeRange):
            return NotImplemented

        try:
            return (self | other) == other
        except ValueError:
            return False

    __lt__ = None
    __gt__ = None
    __ge__ = None

    def __sub__(self, other: DatetimeRange) -> DatetimeRange:
        if not isinstance(other, DatetimeRange):
            return NotImplemented

        other = other & self
        if not other:
            return self

        if self.step == other.step:
            if self.start == other.start:
                return DatetimeRange(other.stop, self.stop, self.step)
            if self.stop == other.stop:
                return DatetimeRange(self.start, other.start, self.step)
            raise ValueError("Cannot create a sparse DatetimeRange")

        if self.start == other[-1]:
            return DatetimeRange(self.start + self.step, self.stop, self.step)
        if self[-1] == other.start:
            return DatetimeRange(self.start, self[-1], self.step)

        if self.step * 2 != other.step:
            raise ValueError("Cannot substract two DatetimeRanges with different steps that overlap on more than one datetime unless the subtrahend's step is twice that of the minuend's")

        if self.start == other.start and self[-1] == other[-1]:
            return DatetimeRange(self.start + self.step, self[-1], other.step)
        if self.start + self.step == other.start and self[-1] == other[-1] + self.step:
            return DatetimeRange(self.start, self.stop, other.step)
        raise ValueError("Cannot substract two DatetimeRanges with different steps when their boundaries are not sufficiently aligned")

    def __xor__(self, other: DatetimeRange) -> DatetimeRange:
        return self - other | other - self
