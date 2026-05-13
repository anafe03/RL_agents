"""Mock playback — a canned step sequence the UI can replay without
running Claude or Playwright.

When the agent runs for real, every step is captured into a
`SubmissionResult`. The bundled mock playback is what a representative
CA DOI run would look like — narrated step by step. The Streamlit UI
walks through these in mock mode and shows what the agent would do.
"""

from __future__ import annotations

from autofill.models import StepAction, SubmissionResult, SubmissionStep


def mock_ca_doi_glp1_run() -> SubmissionResult:
    """A canned playback of filling the CA DOI consumer complaint form for the
    GLP-1 / Cascade Health Plan denial case (data/complaints/glp1_denial.yaml).

    All step narrations are written to feel like a real Claude trace.
    Screenshot paths are placeholder — `data/recordings/ca_doi/` is where
    real captures would go.
    """
    steps_data: list[tuple[StepAction, str, str, str]] = [
        (StepAction.NAVIGATE, "", "https://www.insurance.ca.gov/01-consumers/101-help/", "Navigated to the California Department of Insurance consumer help page."),
        (StepAction.SCREENSHOT, "", "", "Initial screenshot. I can see the CA DOI consumer help landing page with several options including 'File a Complaint'."),
        (StepAction.OBSERVE, "", "", "This is a health-insurance prior-authorization denial complaint. I'll follow the 'File a Complaint' link, then navigate to the health-plan section."),
        (StepAction.CLICK, "File a Complaint link", "", "Clicked 'File a Complaint' to enter the complaint flow."),
        (StepAction.SCREENSHOT, "", "", "I'm now on the complaint type selection page. Options include Auto, Health, Life, etc. Selecting Health."),
        (StepAction.CLICK, "Health insurance category", "", "Clicked the 'Health Insurance' category."),
        (StepAction.SCREENSHOT, "", "", "The complaint form is now visible. I can see fields for complainant info, insurer info, and a narrative box. Starting with complainant info."),
        (StepAction.CLICK, "First Name field", "", "Clicked into the First Name field."),
        (StepAction.TYPE, "First Name", "Sample", "Typed the complainant's first name."),
        (StepAction.CLICK, "Last Name field", "", "Clicked into the Last Name field."),
        (StepAction.TYPE, "Last Name", "Patient", "Typed the complainant's last name."),
        (StepAction.CLICK, "Email field", "", "Clicked into the Email field."),
        (StepAction.TYPE, "Email", "sample.patient@example.com", "Typed the complainant's email."),
        (StepAction.CLICK, "Phone field", "", "Clicked into the Phone field."),
        (StepAction.TYPE, "Phone", "(415) 555-0182", "Typed the complainant's phone number."),
        (StepAction.CLICK, "Address field", "", "Clicked into the Address field."),
        (StepAction.TYPE, "Address", "123 Demo Street, San Francisco, CA 94110", "Typed the complainant's address."),
        (StepAction.SCROLL, "page down", "", "Scrolled down to find the insurer information section."),
        (StepAction.CLICK, "Insurance Company field", "", "Clicked into the Insurance Company / Insurer field."),
        (StepAction.TYPE, "Insurer Name", "Cascade Health Plan", "Typed the insurer name."),
        (StepAction.CLICK, "Member ID field", "", "Clicked into the Member ID / Policy Number field."),
        (StepAction.TYPE, "Member ID", "CHP-7741620-A", "Typed the member ID."),
        (StepAction.CLICK, "Claim Number field", "", "Clicked into the Claim Number / Reference Number field."),
        (StepAction.TYPE, "Claim Number", "PA-2026-0114-7741620", "Typed the claim number."),
        (StepAction.CLICK, "Nature of Complaint dropdown", "", "Clicked the 'Nature of Complaint' dropdown."),
        (StepAction.CLICK, "Prior authorization / denial option", "", "Selected the most relevant complaint category: 'Prior authorization denial' (or closest available)."),
        (StepAction.SCROLL, "page down", "", "Scrolled to the narrative / description text area."),
        (StepAction.CLICK, "Description text area", "", "Clicked into the Description / Tell us what happened text area."),
        (
            StepAction.TYPE,
            "Description (narrative)",
            "I am filing this complaint because Cascade Health Plan has denied medically necessary therapy for my type 2 diabetes despite documented intolerance to the required step-therapy agents. [...full narrative pasted from the ComplaintInput...]",
            "Pasted the full narrative from the complaint input — clinical background, the Cascade step-therapy policy citation, the ADA Standards of Care 2024 reference, and the request that the DOI direct Cascade to apply its own step-therapy exception correctly.",
        ),
        (StepAction.SCROLL, "page down", "", "Scrolled to the supporting documents upload section."),
        (StepAction.OBSERVE, "", "", "Supporting docs would be attached here. Listed in the complaint: metformin_intolerance_note_2023_09.pdf, ER_hypoglycemia_report_2023_11.pdf, denial_letter_2026_01.pdf, ADA_standards_of_care_2024_section_9_4.pdf. For this dry-run I'll note them rather than perform actual file uploads."),
        (StepAction.SCROLL, "to Review section", "", "Scrolled to the bottom Review / Submit section."),
        (StepAction.SCREENSHOT, "filled form ready for review", "", "Form is fully filled. Submit button is visible. Stopping here per the dry-run rule — the human reviews and clicks Submit themselves."),
        (StepAction.HALT, "stop before submit", "", "Halted before clicking Submit. The complaint is ready for human review and submission."),
    ]

    result = SubmissionResult(
        complaint_id="glp1_denial",
        target_id="ca_doi",
        dry_run=True,
        completed_to_review=True,
        cost_usd=0.0,  # mock playback is free
    )
    for i, (action, target_label, value, narration) in enumerate(steps_data):
        result.steps.append(
            SubmissionStep(
                step_id=i,
                action=action,
                narration=narration,
                target_label=target_label,
                value=value,
                screenshot_path=f"data/recordings/ca_doi/step_{i:02d}.png",
            )
        )
    return result


_RECORDINGS = {
    ("ca_doi", "glp1_denial"): mock_ca_doi_glp1_run,
}


def get_mock_run(target_id: str, complaint_id: str) -> SubmissionResult | None:
    """Return the canned playback for (target, complaint), or None if not recorded."""
    factory = _RECORDINGS.get((target_id, complaint_id))
    return factory() if factory else None
