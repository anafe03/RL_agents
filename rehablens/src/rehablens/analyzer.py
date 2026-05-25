"""The form-rule analyzer.

For each exercise, one rule function takes a `PoseFrame` and returns the
list of `FormCheck`s it produced. Keep clinical judgment OUT of this
module — these are illustrative thresholds for a portfolio project, not
validated assessment criteria.
"""

from __future__ import annotations

from collections.abc import Callable

from rehablens.angles import joint_angle
from rehablens.exercises import Exercise, get_exercise
from rehablens.models import FormCheck, FormStatus, Landmark, PoseFrame, SessionResult


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _check_angle(name: str, value: float | None, *,
                 min_ok: float | None = None,
                 max_ok: float | None = None,
                 unit: str = "deg") -> FormCheck:
    """Pass/fail a measured angle against an OK band."""
    if value is None:
        return FormCheck(name=name, status=FormStatus.UNKNOWN, target="—",
                         note="landmark not detected")
    parts = []
    if min_ok is not None:
        parts.append(f"≥ {min_ok:g}")
    if max_ok is not None:
        parts.append(f"≤ {max_ok:g}")
    target = " and ".join(parts) + ("°" if unit == "deg" else "")
    status = FormStatus.OK
    if min_ok is not None and value < min_ok:
        status = FormStatus.FAIL
    if max_ok is not None and value > max_ok:
        status = FormStatus.FAIL
    return FormCheck(name=name, status=status, value=round(value, 1),
                     unit=unit, target=target)


# -- per-exercise rules ------------------------------------------------------

def _squat_rules(frame: PoseFrame) -> list[FormCheck]:
    """Depth (knee flexion), hip flexion, and trunk lean on a bodyweight squat."""
    lk = joint_angle(frame, Landmark.LEFT_HIP,  Landmark.LEFT_KNEE,  Landmark.LEFT_ANKLE)
    rk = joint_angle(frame, Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE)
    lh = joint_angle(frame, Landmark.LEFT_SHOULDER,  Landmark.LEFT_HIP,  Landmark.LEFT_KNEE)
    rh = joint_angle(frame, Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE)
    knee = _mean([lk, rk])
    hip = _mean([lh, rh])

    # Squat depth: smaller knee angle = deeper. Parallel ≈ 90°.
    # We report depth as `180 - knee_angle` so larger = deeper.
    depth = None if knee is None else 180 - knee
    return [
        _check_angle("knee flexion depth", depth, min_ok=90,
                     unit="deg"),  # ≥ 90° depth = at or below parallel
        _check_angle("hip flexion", None if hip is None else 180 - hip,
                     min_ok=70, unit="deg"),
        _check_angle("knee symmetry (L vs R)",
                     None if lk is None or rk is None else abs(lk - rk),
                     max_ok=15, unit="deg"),
    ]


def _shoulder_rom_rules(frame: PoseFrame) -> list[FormCheck]:
    """Overhead reach — shoulder flexion ≥ 160°, elbow stays extended."""
    ls = joint_angle(frame, Landmark.LEFT_HIP,  Landmark.LEFT_SHOULDER,  Landmark.LEFT_ELBOW)
    rs = joint_angle(frame, Landmark.RIGHT_HIP, Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW)
    le = joint_angle(frame, Landmark.LEFT_SHOULDER,  Landmark.LEFT_ELBOW,  Landmark.LEFT_WRIST)
    re = joint_angle(frame, Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST)
    shoulder = _mean([ls, rs])
    elbow = _mean([le, re])
    return [
        _check_angle("shoulder flexion", shoulder, min_ok=160, unit="deg"),
        _check_angle("elbow extension", elbow, min_ok=160, unit="deg"),
    ]


def _single_leg_stand_rules(frame: PoseFrame) -> list[FormCheck]:
    """Front-view single-leg stand — hip level (Trendelenburg) and standing-knee extension.

    Picks the standing leg as the one whose ankle is *lower* in the frame
    (smaller y in image coords means higher up — the lifted leg).
    """
    lk = joint_angle(frame, Landmark.LEFT_HIP,  Landmark.LEFT_KNEE,  Landmark.LEFT_ANKLE)
    rk = joint_angle(frame, Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE)
    left_ankle = frame.point(Landmark.LEFT_ANKLE)
    right_ankle = frame.point(Landmark.RIGHT_ANKLE)
    left_hip = frame.point(Landmark.LEFT_HIP)
    right_hip = frame.point(Landmark.RIGHT_HIP)

    standing_knee = None
    if left_ankle and right_ankle and lk is not None and rk is not None:
        # Standing leg = the one whose ankle is lower in the frame (larger y).
        standing_knee = lk if left_ankle.y >= right_ankle.y else rk

    # Trendelenburg-style sign: stance-side hip drop. Normalized image units.
    hip_drop = None
    if left_hip and right_hip:
        hip_drop = abs(left_hip.y - right_hip.y) * 100  # report as 100×normalized

    return [
        _check_angle("standing-leg knee extension", standing_knee,
                     min_ok=160, unit="deg"),
        _check_angle("pelvic level (Trendelenburg)", hip_drop,
                     max_ok=4.0, unit="%img"),
    ]


_RULES: dict[str, Callable[[PoseFrame], list[FormCheck]]] = {
    "squat": _squat_rules,
    "shoulder_rom": _shoulder_rom_rules,
    "single_leg_stand": _single_leg_stand_rules,
}


def analyze(frame: PoseFrame, exercise: Exercise | str) -> SessionResult:
    """Run the form-rule check for the chosen exercise on this pose frame."""
    ex = exercise if isinstance(exercise, Exercise) else get_exercise(exercise)
    if not frame.detected:
        return SessionResult(
            exercise_id=ex.id, detected=False, checks=[],
            summary="No pose detected — try a clearer, full-body photo.",
        )
    rule = _RULES.get(ex.rules)
    if rule is None:
        return SessionResult(exercise_id=ex.id, detected=True, checks=[],
                             summary=f"No rule defined for {ex.rules!r}.")
    checks = rule(frame)
    result = SessionResult(exercise_id=ex.id, detected=True, checks=checks)
    result.summary = _summary(checks)
    return result


def _summary(checks: list[FormCheck]) -> str:
    fails = [c.name for c in checks if c.status == FormStatus.FAIL]
    warns = [c.name for c in checks if c.status == FormStatus.WARN]
    if fails:
        return "Falls short on: " + ", ".join(fails)
    if warns:
        return "Watch: " + ", ".join(warns)
    return "Within target on every measured rule."
