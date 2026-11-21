"""waypoints 核心 API。

按弧长参数化的折线路径：

- ``point_at(distance)``   按距离取点（超过总长夹取到终点，负距离报错）
- ``tangent_at(distance)`` 该距离处的单位切线
- ``resample(spacing)`` / ``resample_by_count(count)`` 按弧长等距重采样
- ``smooth(tolerance)``    Catmull-Rom 平滑，采样折线与平滑曲线的弦偏差不超过 tolerance

所有累积长度在构造时一次性计算并冻结为不可变 tuple，全部方法只读、
不持有任何可变共享状态，因此多线程并发调用是安全的。

可区分错误（均为 ``ValueError`` 子类）：

- ``TooFewPointsError``    点数不足 2 个
- ``NonFinitePointError``  坐标含 NaN/inf
- ``DuplicatePointError``  相邻重复点（零长度段）
- ``ZeroLengthError``      路径总长为零
- ``NegativeDistanceError`` 负距离
"""

from __future__ import annotations

import bisect
import math

__all__ = [
    "Waypoints",
    "WaypointsError",
    "TooFewPointsError",
    "NonFinitePointError",
    "DuplicatePointError",
    "ZeroLengthError",
    "NegativeDistanceError",
]


class WaypointsError(ValueError):
    """waypoints 全部参数错误的基类。"""


class TooFewPointsError(WaypointsError):
    """点数不足（少于 2 个）。"""


class NonFinitePointError(WaypointsError):
    """坐标含 NaN 或无穷。"""


class DuplicatePointError(WaypointsError):
    """相邻重复点（零长度段）。"""


class ZeroLengthError(WaypointsError):
    """路径总长为零。"""


class NegativeDistanceError(WaypointsError):
    """距离为负。"""


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Waypoints:
    """按弧长参数化的折线路径（构造后不可变，并发安全）。"""

    __slots__ = ("_pts", "_cum", "_total", "_dim")

    def __init__(self, points):
        try:
            raw = list(points)
        except TypeError as exc:
            raise TooFewPointsError("points 必须是点的可迭代序列") from exc

        pts = []
        for p in raw:
            try:
                coord = tuple(float(c) for c in p)
            except (TypeError, ValueError) as exc:
                raise WaypointsError(f"坐标无法解析为数值: {p!r}") from exc
            for c in coord:
                if not math.isfinite(c):
                    raise NonFinitePointError(f"坐标含非有限值: {p!r}")
            pts.append(coord)

        if len(pts) < 2:
            raise TooFewPointsError(f"至少需要 2 个点，实际 {len(pts)} 个")

        dim = len(pts[0])
        if dim == 0:
            raise WaypointsError("点维度不能为空")
        for p in pts:
            if len(p) != dim:
                raise WaypointsError("所有点的维度必须一致")

        cum = [0.0]
        for i in range(1, len(pts)):
            seg = math.dist(pts[i - 1], pts[i])
            if seg == 0.0:
                raise DuplicatePointError(
                    f"相邻重复点（零长度段）: 索引 {i - 1} 与 {i}: {pts[i]!r}"
                )
            cum.append(cum[-1] + seg)

        total = cum[-1]
        if not total > 0.0:
            raise ZeroLengthError("路径总长为零")

        self._pts = tuple(pts)
        self._cum = tuple(cum)
        self._total = total
        self._dim = dim

    # ---------------------------------------------------------------- 基础查询

    def total_length(self):
        """路径总弧长（float，恒为正）。"""
        return self._total

    @property
    def points(self):
        """原始路径点（tuple 的 tuple，只读）。"""
        return self._pts

    def __len__(self):
        return len(self._pts)

    def __repr__(self):
        return f"Waypoints(n={len(self._pts)}, length={self._total:.6g})"

    def _check_distance(self, distance):
        if not _is_number(distance):
            raise WaypointsError(f"距离必须是数值: {distance!r}")
        d = float(distance)
        if math.isnan(d):
            raise WaypointsError("距离不能为 NaN")
        if d < 0.0:
            raise NegativeDistanceError(f"距离为负: {d}")
        return d

    def _segment_index(self, d):
        """返回满足 cum[i-1] <= d < cum[i] 的段索引 i（1..n-1）。"""
        i = bisect.bisect_right(self._cum, d)
        if i <= 0:
            return 1
        if i >= len(self._pts):
            return len(self._pts) - 1
        return i

    def point_at(self, distance):
        """按弧长距离取点；超过总长夹取到终点，负距离抛 NegativeDistanceError。"""
        d = self._check_distance(distance)
        if d <= 0.0:
            return self._pts[0]
        if d >= self._total:
            return self._pts[-1]
        i = self._segment_index(d)
        cum = self._cum
        t = (d - cum[i - 1]) / (cum[i] - cum[i - 1])
        a = self._pts[i - 1]
        b = self._pts[i]
        return tuple(a[k] + (b[k] - a[k]) * t for k in range(self._dim))

    def tangent_at(self, distance):
        """该弧长处的单位切线向量（tuple）；距离越界时夹取到端部段。"""
        d = self._check_distance(distance)
        i = self._segment_index(min(d, self._total))
        a = self._pts[i - 1]
        b = self._pts[i]
        seg = self._cum[i] - self._cum[i - 1]
        return tuple((b[k] - a[k]) / seg for k in range(self._dim))

    # ---------------------------------------------------------------- 等距重采样

    def resample(self, spacing):
        """按目标间距等距重采样，返回点 tuple 列表。

        实际间距为 total/round(total/spacing)，保证严格等距且覆盖全路径。
        """
        if not _is_number(spacing):
            raise WaypointsError(f"间距必须是数值: {spacing!r}")
        s = float(spacing)
        if not math.isfinite(s) or s <= 0.0:
            raise WaypointsError(f"间距必须为正的有限值: {spacing!r}")
        n = max(1, int(round(self._total / s)))
        return self.resample_by_count(n)

    def resample_by_count(self, count):
        """重采样为 count 段等弧长间隔，返回 count+1 个点（含首尾）。"""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise WaypointsError(f"段数必须是 >= 1 的整数: {count!r}")
        step = self._total / count
        out = [self.point_at(step * i) for i in range(count)]
        out.append(self._pts[-1])
        return out

    # ---------------------------------------------------------------- Catmull-Rom 平滑

    def smooth(self, tolerance=0.01):
        """Catmull-Rom 平滑，返回新的 Waypoints。

        均匀参数化 Catmull-Rom 样条过全部原始路径点；每段转为三次 Bezier
        后按 de Casteljau 自适应细分，控制点到弦的最大距离不超过
        ``tolerance``，从而采样折线与真实平滑曲线的偏差不超过
        0.75 * tolerance < tolerance。
        """
        if not _is_number(tolerance):
            raise WaypointsError(f"偏差上界必须是数值: {tolerance!r}")
        tol = float(tolerance)
        if not math.isfinite(tol) or tol <= 0.0:
            raise WaypointsError(f"偏差上界必须为正的有限值: {tolerance!r}")

        pts = self._pts
        dim = self._dim
        ext = (pts[0],) + pts + (pts[-1],)
        out = [pts[0]]
        for i in range(len(pts) - 1):
            p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
            b0 = p1
            b1 = tuple(p1[k] + (p2[k] - p0[k]) / 6.0 for k in range(dim))
            b2 = tuple(p2[k] - (p3[k] - p1[k]) / 6.0 for k in range(dim))
            b3 = p2
            _flatten_bezier(b0, b1, b2, b3, tol, dim, out, 0)
        return Waypoints(out)


def _chord_distance(p, a, b, dim):
    """点 p 到线段 a-b 的距离（N 维）。"""
    dx = tuple(b[k] - a[k] for k in range(dim))
    seg2 = 0.0
    for v in dx:
        seg2 += v * v
    if seg2 == 0.0:
        return math.dist(p, a)
    t = 0.0
    for k in range(dim):
        t += (p[k] - a[k]) * dx[k]
    t /= seg2
    if t <= 0.0:
        return math.dist(p, a)
    if t >= 1.0:
        return math.dist(p, b)
    return math.dist(p, tuple(a[k] + dx[k] * t for k in range(dim)))


def _flatness(b0, b1, b2, b3, dim):
    return max(
        _chord_distance(b1, b0, b3, dim),
        _chord_distance(b2, b0, b3, dim),
    )


def _mid(p, q, dim):
    return tuple((p[k] + q[k]) * 0.5 for k in range(dim))


_MAX_DEPTH = 24


def _flatten_bezier(b0, b1, b2, b3, tol, dim, out, depth):
    if depth >= _MAX_DEPTH or _flatness(b0, b1, b2, b3, dim) <= tol:
        out.append(b3)
        return
    # de Casteljau 在 t=0.5 处分裂
    m01 = _mid(b0, b1, dim)
    m12 = _mid(b1, b2, dim)
    m23 = _mid(b2, b3, dim)
    m012 = _mid(m01, m12, dim)
    m123 = _mid(m12, m23, dim)
    m0123 = _mid(m012, m123, dim)
    _flatten_bezier(b0, m01, m012, m0123, tol, dim, out, depth + 1)
    _flatten_bezier(m0123, m123, m23, b3, tol, dim, out, depth + 1)
