"""The Computer Use agent loop.

Drives a browser via Playwright (executor.py) under the direction of Claude
(llm.py). Loops:

  user task → claude (tool_use) → executor → tool_result(screenshot)
                ↑__________________________________________|

until Claude says it's done (or hits max_steps / detects submit-button click
in --dry-run mode).

This module is intentionally separate from `mock.py`: the mock plays back
canned steps without ever running Claude or Playwright.
"""

from __future__ import annotations

from typing import Any

from autofill import llm
from autofill.models import (
    ComplaintInput,
    FormTarget,
    StepAction,
    SubmissionResult,
    SubmissionStep,
)


DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_STEPS = 30


def _system_prompt(target: FormTarget, complaint: ComplaintInput, dry_run: bool) -> str:
    submit_rule = (
        "DO NOT click the final 'Submit' button. Stop one step before submit, "
        "so the human can review the filled form first. If you reach a submit "
        "button, take a screenshot showing the completed form and stop."
        if dry_run else
        "Submission is authorized for this run. Fill the form completely, "
        "then take a screenshot of the completed form, then click Submit. "
        "After clicking Submit, take one final screenshot and stop."
    )
    return f"""You are filling out a public insurance complaint form on behalf of a consumer.

# Target form
**Name:** {target.name}
**URL:** {target.url}
{target.notes}

# Complaint data (verbatim — use these exact values, do not embellish)
- Complainant name: {complaint.complainant_name}
- Complainant email: {complaint.complainant_email}
- Complainant phone: {complaint.complainant_phone}
- Complainant address: {complaint.complainant_address}
- Insurer name: {complaint.insurer_name}
- Member ID: {complaint.insurer_member_id}
- Claim number: {complaint.claim_number}
- Requested service: {complaint.requested_service}
- Denial reason summary: {complaint.denial_reason}

# Narrative to paste into the "Description" / "Details" field

{complaint.narrative}

# Field mapping hints (look for fields with labels resembling these)
{_format_hints(target.field_hints)}

# Workflow rules
1. Start by taking a screenshot of the current page.
2. Navigate to the complaint form if needed (the page may have multiple flows; for HEALTH insurance complaints follow the relevant link).
3. Fill each form field with the exact data above, matching by semantic label.
4. {submit_rule}
5. Narrate what you're doing in plain text between tool calls so the operator can follow along.
6. If the form has CAPTCHAs or auth gates, stop and report that the form is not suitable for automated submission.

Be deliberate. Re-screenshot after each major action. Do not type random text; only use the data above."""


def _format_hints(hints: dict[str, list[str]]) -> str:
    lines: list[str] = []
    for category, labels in hints.items():
        lines.append(f"- `{category}`: look for {', '.join(repr(l) for l in labels)}")
    return "\n".join(lines)


def run_submission(
    complaint: ComplaintInput,
    target: FormTarget,
    dry_run: bool = True,
    model: str = DEFAULT_MODEL,
    executor: Any = None,
    on_step: Any = None,
) -> SubmissionResult:
    """Drive Claude Computer Use against `target` to fill `complaint`.

    `executor`: a PlaywrightExecutor (or compatible mock). If None, the
    real PlaywrightExecutor is instantiated — pass an in-memory mock for
    testing.

    `on_step(step: SubmissionStep)`: optional callback fired after every
    step is recorded. Streamlit UI uses this to update the display.
    """
    result = SubmissionResult(
        complaint_id=complaint.id,
        target_id=target.id,
        dry_run=dry_run,
    )

    if executor is None:
        from autofill.executor import PlaywrightExecutor
        executor = PlaywrightExecutor()

    try:
        executor.goto(target.url)
        initial = executor.screenshot()
        # First message: kick off the agent with the initial screenshot
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Begin filling the form. Take stock first, then fill methodically."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": initial.screenshot_b64,
                        },
                    },
                ],
            }
        ]

        for step_index in range(MAX_STEPS):
            response = llm.chat(
                model=model,
                system=_system_prompt(target, complaint, dry_run),
                messages=messages,
            )
            result.cost_usd += response.cost_usd

            assistant_text = ""
            tool_uses: list[Any] = []
            for block in response.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    assistant_text += block.text
                elif btype == "tool_use":
                    tool_uses.append(block)

            if assistant_text:
                step = SubmissionStep(
                    step_id=len(result.steps),
                    action=StepAction.OBSERVE,
                    narration=assistant_text.strip(),
                )
                result.steps.append(step)
                if on_step:
                    on_step(step)

            if response.stop_reason == "end_turn" and not tool_uses:
                break

            # Append assistant content to messages so the SDK's tool-use
            # chain stays intact.
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool_use, return tool_results in one user msg
            tool_results: list[dict[str, Any]] = []
            for tu in tool_uses:
                action = tu.input.get("action", "")
                params = {k: v for k, v in tu.input.items() if k != "action"}
                exec_res = executor.execute(action, params)
                step_action = _map_step_action(action, dry_run, params)
                step = SubmissionStep(
                    step_id=len(result.steps),
                    action=step_action,
                    target_label=params.get("text", "") or str(params.get("coordinate", "")),
                    value=params.get("text", ""),
                    coordinate=tuple(params["coordinate"]) if "coordinate" in params and isinstance(params["coordinate"], (list, tuple)) else None,
                )
                result.steps.append(step)
                if on_step:
                    on_step(step)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": exec_res.screenshot_b64,
                            },
                        }
                    ],
                })

            messages.append({"role": "user", "content": tool_results})

        result.completed_to_review = True
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
    finally:
        try:
            executor.close()
        except Exception:
            pass
        from datetime import datetime, timezone
        result.ended_at = datetime.now(timezone.utc)

    return result


def _map_step_action(action: str, dry_run: bool, params: dict[str, Any]) -> StepAction:
    if action == "screenshot":
        return StepAction.SCREENSHOT
    if action in ("left_click", "right_click", "middle_click", "double_click"):
        return StepAction.CLICK
    if action == "type":
        return StepAction.TYPE
    if action == "key":
        return StepAction.KEY
    if action == "scroll":
        return StepAction.SCROLL
    if action == "mouse_move":
        return StepAction.OBSERVE
    return StepAction.OBSERVE
