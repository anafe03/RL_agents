"""Registry of supported complaint-form targets.

A target is a public-record insurance complaint webform with a stable
URL and a known set of semantic fields. Adding a target requires
verifying the form is meant for unauthenticated consumer complaints
(state insurance commissioners are the canonical example).
"""

from __future__ import annotations

from autofill.models import FormTarget


CA_DOI = FormTarget(
    id="ca_doi",
    name="California Department of Insurance — Consumer Complaint",
    state="CA",
    # The CA DOI consumer-complaint flow lives under their public site. The
    # exact endpoint may change; the agent's first step is to navigate to
    # the landing page and follow signs to the complaint form.
    url="https://www.insurance.ca.gov/01-consumers/101-help/index.cfm",
    notes=(
        "CA DOI complaint flow has multiple sub-options (health, auto, life). "
        "Health insurance complaints route through the 'Independent Medical "
        "Review' or 'File a Complaint' link. The agent should follow the "
        "health-insurance complaint path."
    ),
    field_hints={
        "complainant_name": ["First Name", "Last Name", "Full Name"],
        "complainant_email": ["Email", "Email Address"],
        "complainant_phone": ["Phone", "Telephone", "Daytime Phone"],
        "complainant_address": ["Street", "Address", "City", "Zip", "Postal"],
        "insurer_name": ["Insurance Company", "Insurer", "Carrier", "Plan Name"],
        "insurer_member_id": ["Member ID", "Policy Number", "Subscriber ID"],
        "claim_number": ["Claim Number", "Reference Number"],
        "denial_reason": ["Reason", "Type of Complaint", "Nature of Complaint"],
        "narrative": ["Description", "Details", "Explain", "Tell us what happened"],
    },
)


REGISTRY: dict[str, FormTarget] = {
    "ca_doi": CA_DOI,
}


def get_target(target_id: str) -> FormTarget:
    if target_id not in REGISTRY:
        raise ValueError(f"Unknown form target: {target_id!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[target_id]
