from datetime import datetime, timedelta

import pytest
from hypothesis import given
from hypothesis.strategies import DrawFn, composite, datetimes, timedeltas as timedeltas_

from ranges.datetimerange import DatetimeRange

@composite
def timedeltas(draw, *args, nonnull: bool = False, **kwargs):
    if nonnull:
        return draw(timedeltas_(*args, **kwargs).filter(lambda x: bool(x)))
    return draw(timedeltas_(*args, **kwargs))

def test_dtrange_null_step():
    with pytest.raises(ValueError):
        DatetimeRange(datetime.min, datetime.min, timedelta())

def test_dtrange_empty():
    assert not DatetimeRange(datetime.min, datetime.min, timedelta.resolution)
    assert not DatetimeRange(datetime.max, datetime.min, timedelta.resolution)
    assert not DatetimeRange(datetime.min, datetime.max, -timedelta.resolution)

@pytest.mark.parametrize("attribute", ("start", "stop", "step"))
def test_dtrange_immutable_attributes(attribute):
    dtrange = DatetimeRange(datetime.min, datetime.min, timedelta.resolution)
    with pytest.raises(AttributeError):
        setattr(dtrange, attribute, None)

@composite
def EmptyDatetimeRanges(draw: DrawFn) -> DatetimeRange:
    start = draw(datetimes())
    stop = draw(datetimes())
    if start == stop:
        step = draw(timedeltas(nonnull=True))
    elif start < stop:
        step = draw(timedeltas(nonnull=True, max_value=timedelta()))
    else:
        step = draw(timedeltas(nonnull=True, min_value=timedelta()))
    return DatetimeRange(start, stop, step)

@composite
def DatetimeRanges(draw: DrawFn) -> DatetimeRange:
    return DatetimeRange(draw(datetimes()), draw(datetimes()), draw(timedeltas(nonnull=True)))

@given(EmptyDatetimeRanges(), EmptyDatetimeRanges())
def test_dtrange_all_empties_are_equal(left, right):
    assert left == right
