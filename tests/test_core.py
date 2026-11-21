import math

import pytest

import waypoints

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

PTS = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]


def _w():
    return waypoints.Waypoints(PTS)


def _check_start():
    return _w().point_at(0.0) == (0.0, 0.0)


def _check_mid():
    return _w().point_at(1.5) == (1.5, 0.0)


def _check_corner():
    return _w().point_at(3.0) == (3.0, 0.0)


def _check_after_corner():
    p = _w().point_at(5.0)
    return abs(p[1] - 2.0) < 1e-9


def _check_end():
    return _w().point_at(100.0) == (3.0, 4.0)


def _check_total():
    return abs(_w().total_length() - 7.0) < 1e-9


def _check_negative():
    return _raises(lambda: _w().point_at(-1.0))


def _check_too_few():
    return _raises(lambda: waypoints.Waypoints([(0.0, 0.0)]))


def _check_det():
    return _w().point_at(2.0) == _w().point_at(2.0)


def _check_type():
    return isinstance(_w().point_at(1.0), tuple)


def _check_no_mutation():
    src = list(PTS)
    _w().point_at(1.0)
    return PTS == src


def _check_two_points():
    return abs(waypoints.Waypoints([(0.0, 0.0), (0.0, 2.0)]).point_at(1.0)[1] - 1.0) < 1e-9


def _check_zero_length():
    return _raises(lambda: waypoints.Waypoints([(1.0, 1.0), (1.0, 1.0)]))


def _check_float():
    return isinstance(_w().point_at(0.5)[0], float)


def _check_length_positive():
    return _w().total_length() > 0.0

CASES = [("start", "_check_start"), ("mid", "_check_mid"), ("corner", "_check_corner"), ("after_corner", "_check_after_corner"), ("end", "_check_end"), ("total", "_check_total"), ("negative", "_check_negative"), ("too_few", "_check_too_few"), ("det", "_check_det"), ("type", "_check_type"), ("no_mutation", "_check_no_mutation"), ("two_points", "_check_two_points"), ("zero_length", "_check_zero_length"), ("float", "_check_float"), ("length_positive", "_check_length_positive")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
