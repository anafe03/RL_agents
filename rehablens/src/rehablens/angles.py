"""Joint-angle geometry.

A joint angle is the angle at a vertex point B formed by the two segments
A-B and B-C. All measurements happen in the image-plane (2D); the depth
coordinate (z) MediaPipe also returns is ignored for v1 to keep the
math reliable on a single still photo.
"""

from __future__ import annotations

import math

from rehablens.models import Landmark, PointXY, PoseFrame


def angle_at(a: PointXY, b: PointXY, c: PointXY) -> float:
    """Return the angle at `b` formed by rays b→a and b→c, in degrees (0..180)."""
    ax, ay = a.x - b.x, a.y - b.y
    cx, cy = c.x - b.x, c.y - b.y
    dot = ax * cx + ay * cy
    mag = math.hypot(ax, ay) * math.hypot(cx, cy)
    if mag == 0:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos_theta))


def joint_angle(frame: PoseFrame, a: Landmark, b: Landmark, c: Landmark) -> float | None:
    """Joint angle at `b` from a pose frame, or None if any landmark is missing."""
    pa = frame.point(a)
    pb = frame.point(b)
    pc = frame.point(c)
    if pa is None or pb is None or pc is None:
        return None
    return angle_at(pa, pb, pc)


def horizontal_drop(left: PointXY, right: PointXY) -> float:
    """Signed vertical drop between two paired landmarks (e.g., left vs right hip),
    in normalized image units. Positive = right side drops below left."""
    return right.y - left.y


def signed_y_diff(higher_expected: PointXY, lower_expected: PointXY) -> float:
    """Positive when `lower_expected` is actually below `higher_expected`."""
    return lower_expected.y - higher_expected.y
