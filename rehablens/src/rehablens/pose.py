"""MediaPipe pose-detection wrapper.

A single call: `detect_pose(image_bytes)` → `PoseFrame`. MediaPipe is
imported lazily so that test code that mocks the pose layer (and never
touches a real image) doesn't pay the import cost.
"""

from __future__ import annotations

import io

from rehablens.models import PointXY, PoseFrame


def detect_pose(image_bytes: bytes) -> PoseFrame:
    """Detect a pose in an image (PNG/JPEG bytes). Returns landmarks in 0..1 image space."""
    if not image_bytes:
        return PoseFrame(detected=False)

    import numpy as np
    from PIL import Image
    import mediapipe as mp

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb_array = np.array(image)

    pose_solver = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )
    try:
        result = pose_solver.process(rgb_array)
    finally:
        pose_solver.close()

    if not result.pose_landmarks:
        return PoseFrame(detected=False)

    landmarks = [
        PointXY(x=float(lm.x), y=float(lm.y), visibility=float(lm.visibility))
        for lm in result.pose_landmarks.landmark
    ]
    return PoseFrame(detected=True, landmarks=landmarks)
