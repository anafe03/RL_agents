# rehablens

**Vision-based motion analysis for physical-medicine rehab.**

Upload a photo of yourself (or a patient) doing a rehab exercise.
MediaPipe detects body landmarks, rehablens computes the joint angles
that matter for that exercise, and an analyzer flags deviations from
proper form — knee depth on a squat, hip compensation, shoulder
range-of-motion, single-leg balance.

```
photo  ──▶  MediaPipe pose  ──▶  joint angles  ──▶  form checks
              33 landmarks       knee, hip, shoulder, …    ✓ depth 92° (≥90)
                                                          ⚠ hip drop 14°
                                                          ✓ knee tracks toe
```

## Why this exists

PM&R (Physical Medicine & Rehabilitation) lives on motion assessment —
range of motion, gait, form on functional movements. Today that
assessment is done in-clinic with the human eye. A pose-estimation tool
that works on a webcam photo gives a low-cost, repeatable first pass,
and writes structured measurements straight into a record. Useful for
home rehab tracking, telehealth assessments, and shrinking the
in-clinic documentation burden.

## Quickstart

```bash
cd rehablens
uv sync
uv run streamlit run src/rehablens/ui/app.py     # the demo
uv run rehablens analyze path/to/photo.jpg --exercise squat   # CLI
uv run rehablens exercises                       # list exercises
```

## How it works

| Module        | Responsibility |
|---|---|
| `pose.py`     | MediaPipe wrapper — image → 33 body landmarks |
| `angles.py`   | Joint-angle geometry (three points → angle at the vertex) |
| `exercises.py`| The exercise library — squat, shoulder ROM, single-leg stand |
| `analyzer.py` | Given measured angles + exercise rules → `FormCheck`s |
| `render.py`   | Draw the pose skeleton on the image for the UI |

## Scope, honestly

v0.1 — static images only. Each photo is treated as a snapshot of peak
position for the chosen exercise (the bottom of a squat, full overhead
reach, mid-balance). Video / live webcam, multi-frame range-of-motion
arcs, and clinically-validated thresholds are all the next step.

This is a portfolio project, not a clinical tool. The thresholds are
illustrative — not validated assessment criteria, and not a substitute
for a physiatrist or physical therapist.
