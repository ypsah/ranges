from __future__ import annotations

from collections.abc import Iterator, Reversible, Set
from datetime import datetime, timedelta
from enum import auto, Enum, unique
from functools import cached_property, total_ordering


@unique
class _RangeOrdering(Enum):
    ASCENDING = auto()
    DESCENDING = auto()


@total_ordering
class DatetimeRange(Reversible[datetime], Set[datetime]):

    def __init__(self, start: datetime, stop: datetime, step: timedelta, *args, **kwargs):
        if step == timedelta():
            raise ValueError("step must be non-null")

        super().__init__(*args, **kwargs)
        self._start = start
        self._stop = stop
        self._step = step

    @property
    def start(self) -> datetime:
        return self._start

    @property
    def stop(self) -> datetime:
        return self._stop

    @property
    def step(self) -> timedelta:
        return self._step

    @cached_property
    def last(self) -> datetime:
        return self.start + len(self) * self.step

    @cached_property
    def _ordering(self) -> _RangeOrdering:
        return _RangeOrdering.ASCENDING if self.step > timedelta() else _RangeOrdering.DESCENDING

    def _canonical(self) -> DatetimeRange:
        return DatetimeRange(self.start, self.last + self.step, self.step)

    def __iter__(self) -> Iterator[datetime]:
        start = self.start
        if self._ordering is _RangeOrdering.ASCENDING:
            while start < self.stop:
                yield start
                start += self.step
        else:
            while start > self.stop:
                yield start
                start += self.step

    def __len__(self) -> int:
        return int((self.stop - self.start).total_seconds() // self.step.total_seconds())

    def __reversed__(self) -> DatetimeRange:
        return DatetimeRange(self.stop - self.step, self.start - self.step, -self.step)

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
        return not len(self & other)

    def __eq__(self, other: DatetimeRange) -> bool:
        if not isinstance(other, DatetimeRange):
            return NotImplemented

        return self.start == other.start and self.step == other.step and self.last == other.last

    def __le__(self, other: DatetimeRange) -> bool:
        if not isinstance(other, DatetimeRange):
            return NotImplemented

        try:
            return (self | other) == other
        except ValueError:
            return False

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

        if self.start == other.last:
            return DatetimeRange(self.start + self.step, self.stop, self.step)
        if self.last == other.start:
            return DatetimeRange(self.start, self.last, self.step)

        if self.step * 2 != other.step:
            raise ValueError("Cannot substract two DatetimeRanges with different steps that overlap on more than one datetime unless the subtrahend's step is twice that of the minuend's")

        if self.start == other.start and self.last == other.last:
            return DatetimeRange(self.start + self.step, self.last, other.step)
        elif self.start + self.step == other.start and self.last == other.last + self.step:
            return DatetimeRange(self.start, self.stop, other.step)
        raise ValueError("Cannot substract two DatetimeRanges with different steps when their boundaries are not sufficiently aligned")

    def __xor__(self, other: DatetimeRange) -> DatetimeRange:
        return self - other | other - self
