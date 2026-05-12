"""Deterministic mock chat for the public demo.

Returns canned, well-formed JSON responses keyed off the case in the user
payload, so the full retriever → drafter → assessor pipeline produces a
realistic-looking result without calling Anthropic.
"""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic

from priorauth import llm


def _usage(input_tokens: int = 1200, output_tokens: int = 400) -> anthropic.types.Usage:
    return anthropic.types.Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=700,
        cache_read_input_tokens=500,
        server_tool_use=None,
        service_tier=None,
    )


# ---- canned content per case + per stage ----------------------------------

_RETRIEVER_RESPONSES = {
    "glp1_t2d": ["ada_2024_glp1_intolerance", "ada_2024_obesity_glp1", "ada_2024_sulfonylurea_hypoglycemia"],
    "lumbar_mri": ["acr_cauda_equina_2023", "acr_back_pain_red_flags", "nass_cauda_equina_emergency"],
    "biologic_psoriasis": [
        "aad_npf_2020_biologic_contraindication",
        "aad_npf_methotrexate_hbv",
        "aad_il17_safety_hbv",
        "aad_psoriasis_bsa_severity",
    ],
}

_DRAFTER_RESPONSES = {
    "glp1_t2d": {
        "opening": "I am writing to appeal the denial of semaglutide (Ozempic) 1mg weekly for our patient with type 2 diabetes (member CHP-7741620-A). The denial cites Cascade Health Plan Diabetes Step Therapy Policy Section 4.1; we believe the medical record clearly satisfies the documented-intolerance exception within that policy framework, and below we cite ADA Standards of Care supporting GLP-1 RA as appropriate therapy in this patient's clinical situation.",
        "clinical_rationale": [
            "The patient has documented intolerance to metformin (severe GI symptoms requiring discontinuation in September 2023) and documented intolerance to sulfonylurea therapy (symptomatic hypoglycemia event requiring emergency department evaluation in November 2023). Both first-line agents required by the cited step-therapy policy have been trialed, and both were discontinued for clinically appropriate reasons.",
            "Per the American Diabetes Association Standards of Care 2024, when a patient has documented intolerance to metformin or is unable to titrate to an effective dose, alternative therapy with a GLP-1 receptor agonist or other glucose-lowering agent is recommended; step therapy should not delay appropriate care.",
            "The patient additionally meets ADA criteria for GLP-1 RA preference based on obesity (BMI 34.2 kg/m^2) and persistently uncontrolled glycemia (HbA1c 8.4%) despite first- and second-line agents and ongoing dietitian-supported lifestyle intervention. ADA Standards of Care recommend GLP-1 RA for patients with T2D and obesity requiring additional glucose lowering.",
            "Per ADA guidance, rechallenge with a sulfonylurea after a hypoglycemia event is not recommended; an agent with lower hypoglycemia risk such as a GLP-1 RA is preferred."
        ],
        "citations": [
            {"claim": "Step therapy should not delay appropriate care when first-line agents are not tolerated.", "guideline_id": "ada_2024_glp1_intolerance", "quoted_excerpt": "alternative first-line therapy with a GLP-1 receptor agonist or other glucose-lowering agent is recommended. Step therapy requirements should not delay appropriate care when first-line agents are clinically contraindicated or not tolerated."},
            {"claim": "GLP-1 RA is recommended for T2D patients with obesity requiring additional glucose lowering.", "guideline_id": "ada_2024_obesity_glp1", "quoted_excerpt": "GLP-1 receptor agonists are recommended for individuals with type 2 diabetes who require additional glucose lowering AND have associated cardiovascular disease, established kidney disease, or obesity (BMI ≥ 30 kg/m²)."},
            {"claim": "Rechallenge with a sulfonylurea after a hypoglycemia event is not recommended.", "guideline_id": "ada_2024_sulfonylurea_hypoglycemia", "quoted_excerpt": "In patients who have experienced symptomatic hypoglycemia on a sulfonylurea, continuation or rechallenge with the same drug class is not recommended; an agent with lower hypoglycemia risk such as a GLP-1 receptor agonist, DPP-4 inhibitor, or SGLT-2 inhibitor is preferred."}
        ],
        "closing": "Based on documented intolerance to both first-line and second-line agents and the supporting ADA Standards of Care, I respectfully request that Cascade Health Plan overturn the denial and approve semaglutide for this patient. Additional documentation is available upon request."
    },
    "lumbar_mri": {
        "opening": "I am writing to appeal the denial of MRI lumbar spine without contrast for our patient (member VIG-330481-B). The denial cites Vantage Spinal Imaging Policy Section 2.3 (conservative-trial requirement), but this patient presents with multiple cauda equina red flags that, per ACR Appropriateness Criteria and NASS position, bypass the conservative-trial requirement.",
        "clinical_rationale": [
            "The patient has THREE concurrent cauda equina red flags documented in the current visit: (1) new-onset bowel incontinence (4 days), (2) saddle anesthesia in S2-S4 distribution (3 days), and (3) progressive left lower-extremity weakness with documented decline in dorsiflexion strength from 5/5 to 3/5 over two weeks. The clinical picture is consistent with cauda equina syndrome.",
            "Per ACR Appropriateness Criteria for Low Back Pain (2023), urgent MRI lumbar spine in suspected cauda equina syndrome is Category 9 — 'Usually Appropriate' — and the presence of red flags BYPASSES the typical 6-week conservative-trial requirement.",
            "The North American Spine Society identifies cauda equina syndrome as a surgical emergency. Coverage policies that delay MRI in patients meeting clinical criteria are not consistent with standard of care; the cited Vantage policy Section 2.3 governs non-specific chronic low back pain, not red-flag presentations.",
            "Delay in MRI risks permanent neurological deficit including persistent bowel/bladder dysfunction; the urgency is clinical, not elective."
        ],
        "citations": [
            {"claim": "Urgent MRI is Category 9 'Usually Appropriate' in suspected cauda equina syndrome and bypasses the conservative-trial requirement.", "guideline_id": "acr_cauda_equina_2023", "quoted_excerpt": "urgent MRI of the lumbar spine without contrast is designated Category 9 (\"Usually Appropriate\"). The presence of cauda equina red flags BYPASSES the typical 6-week conservative-trial requirement for lumbar MRI in chronic non-specific low back pain."},
            {"claim": "Red flags including bowel/bladder dysfunction, saddle anesthesia, and progressive neurologic deficit require immediate advanced imaging regardless of conservative-therapy duration.", "guideline_id": "acr_back_pain_red_flags", "quoted_excerpt": "(1) new bowel or bladder dysfunction, (2) saddle anesthesia, (3) progressive neurologic deficit, (4) suspicion of malignancy, infection, or fracture. Patients meeting any single red flag criterion are eligible for urgent MRI without completing the conservative trial period otherwise required for non-specific low back pain."},
            {"claim": "Cauda equina syndrome is a surgical emergency; delays in diagnosis are associated with permanent deficits.", "guideline_id": "nass_cauda_equina_emergency", "quoted_excerpt": "Cauda equina syndrome is a surgical emergency. Delays in diagnosis and decompression are strongly associated with permanent neurological deficits, including persistent bowel and bladder dysfunction. Urgent MRI is the imaging modality of choice."}
        ],
        "closing": "Based on the documented cauda equina red flags and the cited ACR and NASS guidance, I respectfully request that Vantage Insurance Group overturn the denial and authorize urgent MRI lumbar spine without contrast. The clinical urgency does not permit further delay."
    },
    "biologic_psoriasis": {
        "opening": "I am writing to appeal the denial of secukinumab (Cosentyx) for our patient with moderate-to-severe plaque psoriasis (member NPW-8821-557). The denial cites Northpoint Biologic Step Therapy Policy Section 3.2, which itself permits a biologic when methotrexate is contraindicated. This patient has a documented contraindication to methotrexate (chronic HBV carrier status with elevated baseline LFTs), and per AAD/NPF guidelines, biologic therapy — specifically an IL-17 inhibitor — is appropriate first-line in this scenario.",
        "clinical_rationale": [
            "The patient has moderate-to-severe plaque psoriasis (BSA 18%, PASI 14, DLQI 16), well above the AAD threshold for systemic therapy. Topical corticosteroid + calcipotriene over 6 months and narrowband UVB phototherapy over 4 months have failed to produce clinically meaningful response.",
            "Methotrexate, the conventional systemic step required by the cited policy, is CONTRAINDICATED for this patient. The patient is a chronic hepatitis B carrier (HBsAg+, on tenofovir suppression) with baseline mildly elevated liver enzymes (ALT 52, AST 44). Per AAD-NPF, methotrexate is hepatotoxic and increases risk of HBV reactivation; chronic HBV with elevated LFTs is a methotrexate contraindication.",
            "The cited Northpoint policy explicitly allows biologic therapy when a documented medical contraindication to methotrexate exists. AAD/NPF joint guidelines further support biologic therapy as first-line systemic when conventional systemic agents are contraindicated.",
            "IL-17 inhibitors (including secukinumab) are the preferred biologic class for HBV-positive patients given absence of meaningful HBV reactivation signal in controlled studies, unlike TNF-alpha inhibitors. This makes secukinumab the appropriate and safest choice for this patient."
        ],
        "citations": [
            {"claim": "Biologic therapy is appropriate first-line systemic when conventional systemics are contraindicated.", "guideline_id": "aad_npf_2020_biologic_contraindication", "quoted_excerpt": "biologic therapy is appropriate as first-line systemic treatment. Step therapy requiring a trial of methotrexate is NOT recommended when the patient has a documented contraindication."},
            {"claim": "Chronic HBV carrier status with elevated LFTs is a methotrexate contraindication.", "guideline_id": "aad_npf_methotrexate_hbv", "quoted_excerpt": "Active HBV infection and chronic HBV carrier status, particularly with baseline hepatic transaminase elevation, are considered contraindications to methotrexate therapy. Alternative therapies should be selected for these patients."},
            {"claim": "IL-17 inhibitors are a preferred biologic class in HBV-positive patients.", "guideline_id": "aad_il17_safety_hbv", "quoted_excerpt": "IL-17 inhibitors have not been associated with clinically significant HBV reactivation in controlled studies. IL-17 inhibitors are a preferred biologic class when systemic therapy is indicated in HBV-positive patients."},
            {"claim": "BSA > 10% and PASI > 10 define moderate-to-severe psoriasis warranting systemic therapy.", "guideline_id": "aad_psoriasis_bsa_severity", "quoted_excerpt": "Plaque psoriasis is classified as moderate-to-severe when any of the following are met: BSA > 10%, PASI > 10, or DLQI > 10 (significant impact on quality of life). Moderate-to-severe disease warrants consideration of systemic therapy."}
        ],
        "closing": "Based on documented contraindication to methotrexate and AAD-NPF guidance supporting biologic — specifically IL-17 inhibitor — therapy in this scenario, I respectfully request that Northpoint Wellness overturn the denial and approve secukinumab for this patient."
    },
}


_ASSESSOR_RESPONSES = {
    "glp1_t2d": {
        "verdict": "strong",
        "addressed_all_denial_criteria": True,
        "all_claims_cited": True,
        "patient_facts_accurate": True,
        "has_clear_ask": True,
        "reasoning": "The appeal directly addresses the step-therapy denial with documented intolerance to both required agents and cites three ADA guideline excerpts that support the requested service. Patient facts are accurately represented; the formal ask is clear.",
        "weak_points": [
            "Consider attaching the September 2023 chart note documenting metformin discontinuation for GI side effects, and the November 2023 ER record for the hypoglycemia event, so the insurer has primary-source verification of the intolerance claims."
        ]
    },
    "lumbar_mri": {
        "verdict": "excellent",
        "addressed_all_denial_criteria": True,
        "all_claims_cited": True,
        "patient_facts_accurate": True,
        "has_clear_ask": True,
        "reasoning": "Strong appeal. Each of the three red flags is named specifically with objective findings (e.g., dorsiflexion strength 5/5 → 3/5), the cited ACR and NASS guidelines directly authorize bypass of the conservative-trial requirement in this exact scenario, and the urgency framing is appropriate.",
        "weak_points": []
    },
    "biologic_psoriasis": {
        "verdict": "strong",
        "addressed_all_denial_criteria": True,
        "all_claims_cited": True,
        "patient_facts_accurate": True,
        "has_clear_ask": True,
        "reasoning": "The appeal directly engages the cited policy's contraindication carve-out, documents both severity criteria and the methotrexate contraindication, and justifies the specific drug class (IL-17 inhibitor) chosen. All four citations match the corpus.",
        "weak_points": [
            "If available, include the most recent hepatology note documenting the HBV management plan — that strengthens the methotrexate-contraindication position with specialist-level documentation."
        ]
    },
}


def _detect_case_id(user_text: str) -> str:
    """Best-effort case identification from anything the matchers would put in the prompt."""
    if "CHP-7741620-A" in user_text or "Cascade Health Plan" in user_text or "Semaglutide" in user_text:
        return "glp1_t2d"
    if "VIG-330481-B" in user_text or "Vantage" in user_text or "cauda equina" in user_text.lower():
        return "lumbar_mri"
    if "NPW-8821-557" in user_text or "Northpoint" in user_text or "Secukinumab" in user_text:
        return "biologic_psoriasis"
    return ""


def _detect_stage(system: str) -> str:
    """Which stage is calling — retriever / drafter / assessor."""
    if "clinical knowledge retriever" in system:
        return "retriever"
    if "clinical writer drafting" in system:
        return "drafter"
    if "independent reviewer" in system:
        return "assessor"
    return "drafter"


def make_mock_chat() -> Any:
    def fn(*, system: str, messages: list, **kwargs: Any) -> llm.ChatResult:
        user_text = ""
        if messages and isinstance(messages[0].get("content"), str):
            user_text = messages[0]["content"]
        case_id = _detect_case_id(user_text)
        stage = _detect_stage(system)

        if stage == "retriever":
            ids = _RETRIEVER_RESPONSES.get(case_id, [])
            payload = {"relevant_guideline_ids": ids}
            return llm.ChatResult(
                text=json.dumps(payload), usage=_usage(), cost_usd=0.003, model="claude-sonnet-4-6"
            )
        if stage == "assessor":
            payload = _ASSESSOR_RESPONSES.get(
                case_id,
                {
                    "verdict": "moderate",
                    "addressed_all_denial_criteria": False,
                    "all_claims_cited": False,
                    "patient_facts_accurate": True,
                    "has_clear_ask": True,
                    "reasoning": "(mock fallback)",
                    "weak_points": ["Mock could not identify case."],
                },
            )
            return llm.ChatResult(
                text=json.dumps(payload), usage=_usage(800, 300), cost_usd=0.012, model="claude-opus-4-7"
            )
        # drafter
        payload = _DRAFTER_RESPONSES.get(case_id)
        if payload is None:
            payload = {
                "opening": "(mock fallback — case not recognized)",
                "clinical_rationale": [],
                "citations": [],
                "closing": "",
            }
        return llm.ChatResult(
            text=json.dumps(payload), usage=_usage(1500, 700), cost_usd=0.008, model="claude-sonnet-4-6"
        )

    return fn
