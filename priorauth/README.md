# PriorAuth Assist

> Insurance denied your prior-auth. Here's a cited appeal that quotes the right clinical guidelines.

**🏥 [Try the live demo →](https://priorauth-assist.streamlit.app/)** *(deploy URL — replace once published)*

US clinicians spend an estimated **14 hours per week per FTE** on prior-authorization workflows, and the AMA estimates **$93B/yr** in admin cost across the system. A meaningful chunk of that is drafting and re-drafting appeals when an insurer rejects a request. PriorAuth Assist drops in a denial letter + de-identified patient context and produces a structured appeal with cited clinical guidelines — exactly the kind of letter a clinic admin would otherwise draft from scratch.

## The engineering wedge

This is not an "LLM writes a polite letter" demo. The interesting work is the safety layer and the retrieval comparison.

- **Citation enforcement.** Every clinical claim in the generated appeal must trace to a retrieved guideline. If the model wants to claim "ADA recommends GLP-1 first-line for BMI > 30," there must be a guideline excerpt in the retrieval set that supports it, or the assessor flags the claim.
- **Structured output the whole way down.** No free-text in the appeal pipeline — `DenialLetter`, `PatientContext`, `Guideline`, `Citation`, `Appeal`, `AppealAssessment` are Pydantic models. The output gets validated, not just generated.
- **Rubric assessment.** Before the appeal is delivered, a separate "assessor" run evaluates the case strength against a fixed rubric: are all denial criteria addressed, does every clinical claim cite a guideline, is the patient context represented accurately, is there a clear ask. Cases come back as `excellent / strong / moderate / weak` with the reasons.
- **Pluggable retrieval + benchmark (v0.2).** Three retriever backends behind a common interface — `BM25` (keyword), `ChromaDB` (dense vector via ONNX-MiniLM), `LLMJudged` (zero-shot Sonnet 4.6). A built-in benchmark runs the same corpus through each, scores `precision@k` / `recall@k` / latency / cost against per-case golden guideline IDs. See **RAG comparison** below.

The split between *drafter* and *assessor* is deliberate — the drafter is incentivized to be persuasive; the assessor is incentivized to find weaknesses. Two LLM runs with different objectives, both structured, fast.

## RAG comparison (v0.2)

Three retrieval strategies, same corpus, same queries:

| Retriever | Strategy | Precision@5 | Recall@5 | Latency (avg, ms) | Cost per case |
|---|---|---:|---:|---:|---:|
| `bm25` | Classical keyword (`rank-bm25`) | 0.62 | 1.00 | 0.8 | $0 |
| `chroma_minilm` | Dense vector (Chroma + ONNX-MiniLM all-MiniLM-L6-v2) | 0.67 | 1.00 | ~190 | $0 |
| `llm_judged` | Zero-shot LLM picks (Claude Sonnet 4.6) | 1.00 | 1.00 | LLM-bound | ~$0.009 |

*Numbers from `priorauth benchmark` against the bundled 3 cases × 10 guidelines.*

The interesting finding for a small, well-curated corpus: **BM25 is competitive with dense vector retrieval on recall**, at zero cost and ~200× lower latency. The LLM-judged retriever wins precision because it only returns guidelines it actively believes are relevant — but at API cost.

Add a new retriever by implementing the `Retriever` ABC in `src/priorauth/retrievers/`. Future v0.2.1 will add Qdrant + Pinecone for the "same embedding, different storage" comparison.

```bash
# Try it
priorauth benchmark                              # local-only: BM25 + Chroma
priorauth benchmark --retrievers bm25 --retrievers chroma_minilm --retrievers llm_judged  # all three (needs API key)
```

## Three cases ship out of the box

All synthetic, no PHI. Each is a realistic denial archetype:

1. **GLP-1 receptor agonist for Type 2 diabetes** — denied for "step therapy not completed." Patient has documented intolerance to metformin (severe GI) and a hypoglycemia episode on a sulfonylurea. Rebuttal: ADA Standards of Care recognizes intolerance as bypassing step therapy.

2. **Lumbar MRI for chronic back pain** — denied as "not medically necessary." Patient has cauda equina red flags (new bowel/bladder dysfunction, saddle anesthesia, progressive lower-extremity weakness). Rebuttal: ACR appropriateness criteria designates this as Category 9 — urgent imaging.

3. **Biologic (IL-17 inhibitor) for plaque psoriasis** — denied for "step therapy requires methotrexate trial." Patient is a chronic hepatitis B carrier with mildly elevated LFTs (contraindication to methotrexate). Rebuttal: AAD/NPF guidelines support biologic as appropriate first-line when methotrexate is contraindicated.

## Quick start — local UI

```bash
cd priorauth
uv sync
uv run streamlit run src/priorauth/ui/app.py
# Demo mode works without an API key.
```

## Quick start — CLI

```bash
cd priorauth
uv sync
export ANTHROPIC_API_KEY=...

priorauth list-cases
priorauth appeal data/cases/glp1_t2d.yaml --report appeal.md
```

## Architecture

```
priorauth/
├── src/priorauth/
│   ├── models.py        ← Pydantic: PatientContext, DenialLetter, Guideline, Citation,
│   │                                Appeal, AppealAssessment, RubricVerdict
│   ├── llm.py           ← Anthropic SDK wrapper with prompt caching
│   ├── retriever.py     ← LLM-judged selection of relevant guidelines from the corpus
│   ├── drafter.py       ← The agent: drafts an appeal with required citations
│   ├── assessor.py      ← Separate run: scores the draft against a fixed rubric
│   ├── mock.py          ← Demo-mode chat function (no API key needed)
│   ├── cli.py
│   └── ui/app.py
├── data/
│   ├── guidelines/      ← Hand-curated synthetic clinical guideline excerpts
│   └── cases/           ← Synthetic case files (denial + patient context per case)
└── tests/
```

## Roadmap

- **v0.1** — 3 cases, 10 hand-curated guideline excerpts, drafter + assessor + rubric, Streamlit UI with mock + live modes.
- **v0.2** *(current)* — Pluggable retriever interface, BM25 + ChromaDB + LLM-judged backends, benchmark harness with precision@k / recall@k / latency / cost, retriever selector in UI + CLI.
- **v0.2.1** — Qdrant retriever (same embedding family as Chroma; different storage). Comparison of "same embedding, different store" — the more honest vector-DB comparison.
- **v0.3** — Output formats: editable letter + DOCX export. Integration sketch for an EHR.
- **v0.4** — Outcome tracker: did the appeal win? Feedback loop to tune the rubric.

## Disclaimers

This is a portfolio project. **It is not a medical device, not a legal substitute for clinician judgment, and not authorized for use in patient care.** All cases and guidelines shipped here are synthetic, written for demonstration. Real prior-auth appeals must be reviewed and signed by a licensed clinician.

## License

MIT.
