"""The exercise library.

Each `Exercise` packages two things: the joint angles to measure on a
frame, and the rules an `analyzer` applies to those measurements to
produce `FormCheck`s. Adding an exercise = adding one entry here.

The thresholds are illustrative — they are NOT validated clinical
criteria. A real deployment would replace them with literature-backed
values per exercise.
"""

from __future__ import annotations

from dataclasses import dataclass

from rehablens.models import Landmark


@dataclass(frozen=True)
class AngleSpec:
    """A named joint angle to measure, defined by its three landmarks."""

    name: str  # e.g. "left knee", "trunk lean"
    a: Landmark
    b: Landmark  # the vertex
    c: Landmark


@dataclass(frozen=True)
class Exercise:
    id: str
    name: str
    description: str
    angles: list[AngleSpec]
    # `rules` is referenced by name from analyzer.py — each exercise has a
    # dedicated rule function that turns angles into FormCheks.
    rules: str


SQUAT = Exercise(
    id="squat",
    name="Bodyweight squat — bottom position",
    description=(
        "Side-view photo at the bottom of a squat. Checks knee flexion depth, "
        "hip flexion, and trunk lean."
    ),
    angles=[
        AngleSpec("left knee",  Landmark.LEFT_HIP,  Landmark.LEFT_KNEE,  Landmark.LEFT_ANKLE),
        AngleSpec("right knee", Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
        AngleSpec("left hip",   Landmark.LEFT_SHOULDER,  Landmark.LEFT_HIP,  Landmark.LEFT_KNEE),
        AngleSpec("right hip",  Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE),
    ],
    rules="squat",
)

SHOULDER_ROM = Exercise(
    id="shoulder_rom",
    name="Shoulder flexion — overhead reach",
    description=(
        "Side-view photo at maximum overhead reach. Checks shoulder flexion "
        "angle (target ≥ 160°) and elbow extension during the reach."
    ),
    angles=[
        AngleSpec("left shoulder",  Landmark.LEFT_HIP,  Landmark.LEFT_SHOULDER,  Landmark.LEFT_ELBOW),
        AngleSpec("right shoulder", Landmark.RIGHT_HIP, Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW),
        AngleSpec("left elbow",  Landmark.LEFT_SHOULDER,  Landmark.LEFT_ELBOW,  Landmark.LEFT_WRIST),
        AngleSpec("right elbow", Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
    ],
    rules="shoulder_rom",
)

SINGLE_LEG_STAND = Exercise(
    id="single_leg_stand",
    name="Single-leg stand",
    description=(
        "Front-view photo holding a single-leg balance. Checks hip-level "
        "(Trendelenburg sign) and standing-leg knee extension."
    ),
    angles=[
        AngleSpec("left knee",  Landmark.LEFT_HIP,  Landmark.LEFT_KNEE,  Landmark.LEFT_ANKLE),
        AngleSpec("right knee", Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
    ],
    rules="single_leg_stand",
)


EXERCISES: dict[str, Exercise] = {
    SQUAT.id: SQUAT,
    SHOULDER_ROM.id: SHOULDER_ROM,
    SINGLE_LEG_STAND.id: SINGLE_LEG_STAND,
}


def get_exercise(exercise_id: str) -> Exercise:
    if exercise_id not in EXERCISES:
        raise ValueError(f"Unknown exercise: {exercise_id!r}. "
                         f"Available: {sorted(EXERCISES)}")
    return EXERCISES[exercise_id]
