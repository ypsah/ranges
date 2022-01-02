from contextlib import suppress
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional

import pytest
from hypothesis import given
from hypothesis.strategies import DrawFn, composite, datetimes, integers, timedeltas as timedeltas_

from ranges.datetimerange import DatetimeRange

@composite
@wraps(timedeltas_)
def timedeltas(draw: DrawFn, *args, nonnull: bool = False, **kwargs):
    if nonnull:
        return draw(timedeltas_(*args, **kwargs).filter(lambda x: bool(x)))
    return draw(timedeltas_(*args, **kwargs))

def test_dtrange_cannot_have_a_null_step():
    with pytest.raises(ValueError):
        DatetimeRange(datetime.min, datetime.min, timedelta())

def test_empty_dtranges():
    assert not DatetimeRange(datetime.min, datetime.min, timedelta.resolution)
    assert not DatetimeRange(datetime.max, datetime.min, timedelta.resolution)
    assert not DatetimeRange(datetime.min, datetime.max, -timedelta.resolution)

@pytest.mark.parametrize("attribute", ("start", "stop", "step"))
def test_dtrange_attributes_are_immutable(attribute: str):
    dtrange = DatetimeRange(datetime.min, datetime.min, timedelta.resolution)
    with pytest.raises(AttributeError):
        setattr(dtrange, attribute, None)

_MAX_MIN_LENGTH = (datetime.max - datetime.min) // timedelta.resolution

@composite
def DatetimeRanges(draw: DrawFn, min_length: int = 0, max_length: Optional[int] = None) -> DatetimeRange:
    if min_length < 0:
        raise ValueError("min_length must be 0 or a positive integer")
    if min_length > _MAX_MIN_LENGTH:
        raise ValueError(f"min_length is too big, must be <= {_MAX_MIN_LENGTH}")
    if max_length is not None and max_length < 0:
        raise ValueError("max_length must be 0 or a positive integer")
    if max_length is not None and min_length > max_length:
        raise ValueError("min_length must be lower or equal to max_length")

    max_step = timedelta.max
    min_step = timedelta.min
    if min_length:
        max_step = (datetime.max - datetime.min) // min_length
        min_step = -max_step
    step = draw(timedeltas(nonnull=True, min_value=min_step, max_value=max_step))

    min_start = datetime.min
    max_start = datetime.max
    if min_length:
        if step > timedelta():
            with suppress(OverflowError):
                max_start = datetime.max - step * min_length
        else:
            with suppress(OverflowError):
                min_start = datetime.min - step * min_length
    start = draw(datetimes(min_value=min_start, max_value=max_start))

    min_stop = datetime.min
    max_stop = datetime.max
    if max_length is not None:
        with suppress(OverflowError):
            spread = abs(step * max_length)
            with suppress(OverflowError):
                min_stop = start - spread
            with suppress(OverflowError):
                max_stop = start + spread
    if min_length:
        with suppress(OverflowError):
            if step > timedelta():
                min_stop = start + step * min_length
            else:
                max_stop = start + step * min_length
    stop = draw(datetimes(min_value=min_stop, max_value=max_stop))

    dtrange = DatetimeRange(start, stop, step)
    assert len(dtrange) >= min_length
    if max_length is not None:
        assert len(dtrange) <= max_length
    return dtrange

@given(DatetimeRanges())
def test_dtrange_is_hashable(dtrange: DatetimeRange):
    hash(dtrange)

@given(DatetimeRanges(max_length=0), DatetimeRanges(max_length=0))
def test_empty_dtranges_are_equal(left: DatetimeRange, right: DatetimeRange):
    assert left == right

@given(DatetimeRanges())
def test_dtrange_is_iterable(dtrange: DatetimeRange):
    iter(dtrange)

@given(DatetimeRanges(max_length=3))
def test_dtrange_is_a_sequence(dtrange: DatetimeRange):
    for i, dt in enumerate(dtrange):
        assert dt == dtrange[i]

@given(DatetimeRanges())
def test_dtrange_is_reversible(dtrange: DatetimeRange):
    try:
        rev = reversed(dtrange)
    except OverflowError:
        with pytest.raises(OverflowError):
            dtrange.start - dtrange.step
        return

    assert isinstance(rev, DatetimeRange)
    assert len(rev) == len(dtrange)
    if dtrange:
        assert rev[0] == dtrange[-1]
        assert rev[-1] == dtrange[0]
        assert rev.step == -dtrange.step

@given(DatetimeRanges(min_length=1), integers())
def test_dtrange(dtrange: DatetimeRange, i: int):
    i %= len(dtrange)
    assert dtrange[i] == dtrange.start + i * dtrange.step

@given(DatetimeRanges(min_length=1), integers())
def test_dtrange_is_negatively_indexable(dtrange: DatetimeRange, i: int):
    i %= len(dtrange)
    assert dtrange[-len(dtrange) + i] == dtrange[i]

@given(DatetimeRanges(max_length=0), datetimes())
def test_empty_dtrange_contains_nothing(dtrange: DatetimeRange, dt: datetime):
    assert dt not in dtrange

@given(DatetimeRanges(min_length=1, max_length=3))
def test_empty_dtrange_contains_its_elements(dtrange: DatetimeRange):
    for dt in dtrange:
        assert dt in dtrange

@given(DatetimeRanges())
def test_dtrange_does_not_contain_stop(dtrange: DatetimeRange):
    assert dtrange.stop not in dtrange

def test_non_overlapping_dtranges_do_not_intersect():
    left = DatetimeRange(datetime.min, datetime.min + timedelta.resolution, timedelta.resolution)
    right = DatetimeRange(datetime.max - timedelta.resolution, datetime.max, timedelta.resolution)
    assert not left & right

def test_dtranges_with_opposite_directions_do_not_intersect():
    left = DatetimeRange(datetime.min + timedelta.resolution, datetime.max, timedelta.resolution)
    right = DatetimeRange(datetime.max - timedelta.resolution, datetime.min, -timedelta.resolution)
    with pytest.raises(ValueError):
        assert not left & right

@given(DatetimeRanges())
def test_dtrange_intersects_with_itself(dtrange: DatetimeRange):
    assert dtrange & dtrange == dtrange

@given(DatetimeRanges())
def test_dtrange_intersection(dtrange: DatetimeRange):
    if dtrange.step > timedelta():
        assert dtrange & DatetimeRange(datetime.min, datetime.max, timedelta.resolution) == dtrange
    else:
        assert dtrange & DatetimeRange(datetime.max, datetime.min, -timedelta.resolution) == dtrange
