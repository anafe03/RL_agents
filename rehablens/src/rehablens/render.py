"""Render a pose skeleton onto an image for the UI."""

from __future__ import annotations

import io

from rehablens.models import PoseFrame

# MediaPipe's canonical body-connection edges (simplified to the ones we care
# about). Each pair is (start landmark idx, end landmark idx).
_SKELETON = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16),
    # Left leg
    (23, 25), (25, 27),
    # Right leg
    (24, 26), (26, 28),
]

# Keypoints we draw a labeled dot at — the ones the form rules read.
_KEYPOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def overlay_pose(image_bytes: bytes, frame: PoseFrame) -> bytes:
    """Return PNG bytes of the input image with the pose skeleton drawn on top."""
    from PIL import Image, ImageDraw

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if not frame.detected:
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()

    draw = ImageDraw.Draw(image)
    w, h = image.size

    def _xy(lm_idx: int) -> tuple[float, float] | None:
        if lm_idx >= len(frame.landmarks):
            return None
        lm = frame.landmarks[lm_idx]
        return (lm.x * w, lm.y * h)

    # Skeleton edges
    for a_idx, b_idx in _SKELETON:
        a = _xy(a_idx)
        b = _xy(b_idx)
        if a is None or b is None:
            continue
        draw.line([a, b], fill=(34, 211, 238), width=4)  # cyan

    # Keypoints
    for idx in _KEYPOINTS:
        p = _xy(idx)
        if p is None:
            continue
        r = 6
        draw.ellipse(
            [(p[0] - r, p[1] - r), (p[0] + r, p[1] + r)],
            fill=(34, 211, 238), outline=(255, 255, 255), width=2,
        )

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
