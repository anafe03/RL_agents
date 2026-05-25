"""Tests for rehablens.

The pose-detection layer needs a real image + MediaPipe, so it's not
tested here. Everything else — joint-angle geometry, the exercise
library, the rule analyzer — is pure and is fully covered, using
hand-built `PoseFrame`s with synthetic landmarks.
"""

from __future__ import annotations

from rehablens.analyzer import analyze
from rehablens.angles import angle_at, joint_angle
from rehablens.exercises import EXERCISES, SQUAT, get_exercise
from rehablens.models import FormStatus, Landmark, PointXY, PoseFrame


# -- angle geometry ----------------------------------------------------------

def _p(x: float, y: float) -> PointXY:
    return PointXY(x=x, y=y)


def test_angle_at_right_angle():
    # A right angle at the origin.
    assert abs(angle_at(_p(1, 0), _p(0, 0), _p(0, 1)) - 90.0) < 1e-6


def test_angle_at_straight():
    # A straight line through the vertex.
    assert abs(angle_at(_p(-1, 0), _p(0, 0), _p(1, 0)) - 180.0) < 1e-6


def test_angle_at_45():
    assert abs(angle_at(_p(1, 0), _p(0, 0), _p(1, 1)) - 45.0) < 1e-6


def test_angle_at_degenerate_returns_zero():
    # Coincident point — undefined; we choose 0 rather than NaN.
    assert angle_at(_p(0, 0), _p(0, 0), _p(1, 1)) == 0.0


# -- joint_angle pulls landmarks from a PoseFrame ---------------------------

def _frame_with(points: dict[Landmark, PointXY]) -> PoseFrame:
    """Build a PoseFrame with the given landmarks populated; rest at (0,0)."""
    landmarks = [PointXY(x=0.0, y=0.0, visibility=0.0) for _ in range(33)]
    for lm, p in points.items():
        landmarks[int(lm)] = p
    return PoseFrame(detected=True, landmarks=landmarks)


def test_joint_angle_reads_three_landmarks():
    # Straight leg: hip above knee above ankle → 180°.
    frame = _frame_with({
        Landmark.LEFT_HIP:   _p(0.5, 0.3),
        Landmark.LEFT_KNEE:  _p(0.5, 0.6),
        Landmark.LEFT_ANKLE: _p(0.5, 0.9),
    })
    angle = joint_angle(frame, Landmark.LEFT_HIP, Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE)
    assert angle is not None and abs(angle - 180.0) < 1e-6


def test_joint_angle_missing_landmark_returns_none():
    frame = PoseFrame(detected=False, landmarks=[])
    assert joint_angle(frame, Landmark.LEFT_HIP, Landmark.LEFT_KNEE,
                       Landmark.LEFT_ANKLE) is None


# -- exercises library ------------------------------------------------------

def test_every_exercise_has_a_rule_handler():
    """If we declared an exercise we should be able to analyze it."""
    from rehablens.analyzer import _RULES
    for ex in EXERCISES.values():
        assert ex.rules in _RULES, f"exercise {ex.id} has no handler for rules={ex.rules}"


def test_get_exercise_unknown_raises():
    try:
        get_exercise("not_an_exercise")
    except ValueError as e:
        assert "Unknown exercise" in str(e)
    else:
        raise AssertionError("expected ValueError")


# -- analyzer ---------------------------------------------------------------

def test_analyze_no_pose_detected():
    result = analyze(PoseFrame(detected=False), SQUAT)
    assert not result.detected
    assert result.overall == FormStatus.UNKNOWN
    assert "No pose detected" in result.summary


def _good_squat_frame() -> PoseFrame:
    """A side-view deep squat — knee flexion clearly past parallel, trunk
    leaning forward over the bent knees, symmetric. Comfortable margin on
    every threshold so floating-point edge cases can't trip the test."""
    return _frame_with({
        Landmark.LEFT_SHOULDER:  _p(0.55, 0.40), Landmark.RIGHT_SHOULDER: _p(0.55, 0.40),
        Landmark.LEFT_HIP:   _p(0.40, 0.65), Landmark.RIGHT_HIP:   _p(0.40, 0.65),
        Landmark.LEFT_KNEE:  _p(0.65, 0.75), Landmark.RIGHT_KNEE:  _p(0.65, 0.75),
        Landmark.LEFT_ANKLE: _p(0.50, 0.85), Landmark.RIGHT_ANKLE: _p(0.50, 0.85),
    })


def test_squat_passes_on_deep_symmetric_pose():
    result = analyze(_good_squat_frame(), SQUAT)
    assert result.detected
    # All measured checks should be OK on the synthetic deep-squat frame.
    assert result.overall in (FormStatus.OK, FormStatus.UNKNOWN)
    assert not any(c.status == FormStatus.FAIL for c in result.checks)


def test_squat_fails_on_shallow_depth():
    # Almost straight legs — barely bent. Depth should fail.
    frame = _frame_with({
        Landmark.LEFT_SHOULDER:  _p(0.5, 0.10), Landmark.RIGHT_SHOULDER: _p(0.5, 0.10),
        Landmark.LEFT_HIP:   _p(0.5, 0.30), Landmark.RIGHT_HIP:   _p(0.5, 0.30),
        Landmark.LEFT_KNEE:  _p(0.5, 0.60), Landmark.RIGHT_KNEE:  _p(0.5, 0.60),
        Landmark.LEFT_ANKLE: _p(0.5, 0.90), Landmark.RIGHT_ANKLE: _p(0.5, 0.90),
    })
    result = analyze(frame, "squat")
    depth_checks = [c for c in result.checks if "depth" in c.name]
    assert depth_checks and depth_checks[0].status == FormStatus.FAIL


def test_shoulder_rom_fails_on_low_reach():
    # Arms hanging down — no overhead reach. Shoulder flexion ~0°.
    frame = _frame_with({
        Landmark.LEFT_HIP:      _p(0.4, 0.6), Landmark.RIGHT_HIP:     _p(0.6, 0.6),
        Landmark.LEFT_SHOULDER: _p(0.4, 0.3), Landmark.RIGHT_SHOULDER: _p(0.6, 0.3),
        Landmark.LEFT_ELBOW:    _p(0.4, 0.5), Landmark.RIGHT_ELBOW:    _p(0.6, 0.5),
        Landmark.LEFT_WRIST:    _p(0.4, 0.7), Landmark.RIGHT_WRIST:    _p(0.6, 0.7),
    })
    result = analyze(frame, "shoulder_rom")
    shoulder_checks = [c for c in result.checks if "shoulder" in c.name]
    assert shoulder_checks and shoulder_checks[0].status == FormStatus.FAIL


def test_overall_status_is_worst_check():
    """If any check fails, overall is FAIL even when others pass."""
    result = analyze(_good_squat_frame(), SQUAT)
    # Manually inject a FAIL.
    from rehablens.models import FormCheck
    result.checks.append(FormCheck(name="fake", status=FormStatus.FAIL, target=""))
    assert result.overall == FormStatus.FAIL


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
