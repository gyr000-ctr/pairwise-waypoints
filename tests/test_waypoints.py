"""新增能力测试：等距重采样、切线、Catmull-Rom 平滑、性能、并发与可区分错误。

不改动既有 tests/test_core.py；用例名均含 waypoint 以匹配 `-k waypoint`。
"""

import math
import threading
import time

import pytest

import waypoints
from waypoints import (
    DuplicatePointError,
    NegativeDistanceError,
    NonFinitePointError,
    TooFewPointsError,
    Waypoints,
    WaypointsError,
)


def _path():
    return Waypoints([(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)])


def test_waypoint_resample_equidistant():
    w = _path()
    samples = w.resample_by_count(70)
    assert len(samples) == 71
    assert samples[0] == (0.0, 0.0)
    assert samples[-1] == (3.0, 4.0)
    # 等距：相邻采样点弧长差一致（直线段上弧长即欧氏距离的分段和）
    step = w.total_length() / 70
    for i in range(71):
        p = w.point_at(step * i)
        assert math.dist(p, samples[i]) < 1e-9


def test_waypoint_resample_spacing_covers_path():
    w = _path()
    samples = w.resample(0.5)
    assert samples[0] == w.points[0]
    assert samples[-1] == w.points[-1]
    # 严格等距：间距 = total / round(total/spacing)
    n = len(samples) - 1
    assert abs(w.total_length() / n - 0.5) < 0.5


def test_waypoint_tangent_unit_and_direction():
    w = _path()
    tx, ty = w.tangent_at(1.0)
    assert abs(tx - 1.0) < 1e-9 and abs(ty) < 1e-9
    tx, ty = w.tangent_at(5.0)
    assert abs(tx) < 1e-9 and abs(ty - 1.0) < 1e-9
    # 越界夹取不报错
    assert w.tangent_at(1e300) == w.tangent_at(w.total_length())


def test_waypoint_smooth_passes_through_and_bounded():
    pts = [(0.0, 0.0), (1.0, 2.0), (3.0, 0.0), (4.0, 3.0), (6.0, 1.0)]
    w = Waypoints(pts)
    tol = 0.05
    sm = w.smooth(tol)
    assert isinstance(sm, Waypoints)
    # 平滑曲线仍经过全部原始点
    for p in pts:
        assert any(math.dist(p, q) < 1e-9 for q in sm.points)
    # 采样折线与高细分参考曲线的偏差不超过上界
    ref = w.smooth(1e-4)
    ref_pts = ref.resample_by_count(4000)[::20]
    worst = 0.0
    sp = sm.points
    for i in range(len(sp) - 1):
        a, b = sp[i], sp[i + 1]
        for j in range(1, 8):
            t = j / 8
            p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            worst = max(worst, min(math.dist(p, q) for q in ref_pts))
    assert worst <= tol


def test_waypoint_smooth_identity_on_straight_line():
    w = Waypoints([(0.0, 0.0), (10.0, 0.0)])
    sm = w.smooth(0.01)
    for p in sm.points:
        assert abs(p[1]) < 1e-9


def test_waypoint_distinguishable_errors():
    with pytest.raises(DuplicatePointError):
        Waypoints([(0.0, 0.0), (1.0, 1.0), (1.0, 1.0)])
    with pytest.raises(NonFinitePointError):
        Waypoints([(0.0, 0.0), (float("nan"), 1.0)])
    with pytest.raises(NonFinitePointError):
        Waypoints([(0.0, 0.0), (float("inf"), 1.0)])
    with pytest.raises(NegativeDistanceError):
        _path().point_at(-1e-9)
    with pytest.raises(TooFewPointsError):
        Waypoints([(0.0, 0.0)])
    # 均可区分且互不相同的子类
    assert len({DuplicatePointError, NonFinitePointError,
                NegativeDistanceError, TooFewPointsError}) == 4
    # 同时都是 ValueError，兼容既有 _raises 约定
    for exc in (DuplicatePointError, NonFinitePointError,
                NegativeDistanceError, TooFewPointsError):
        assert issubclass(exc, ValueError)


def test_waypoint_huge_distance_clamps():
    w = _path()
    assert w.point_at(1e300) == (3.0, 4.0)
    assert w.point_at(float("inf")) == (3.0, 4.0)


def test_waypoint_performance_100k_segments():
    pts = [(float(i), math.sin(i * 0.01) * 50.0) for i in range(100001)]
    w = Waypoints(pts)
    t0 = time.perf_counter()
    samples = w.resample_by_count(100000)
    elapsed = time.perf_counter() - t0
    assert len(samples) == 100001
    assert elapsed <= 0.300, f"10 万段等距采样耗时 {elapsed * 1e3:.1f} ms > 300 ms"


def test_waypoint_thread_safety():
    w = _path()
    expected = (w.point_at(2.5), w.tangent_at(2.5), w.resample(0.3))
    errors = []

    def work():
        try:
            for _ in range(500):
                got = (w.point_at(2.5), w.tangent_at(2.5), w.resample(0.3))
                assert got == expected
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
