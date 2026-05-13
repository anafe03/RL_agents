# AutoFill

> The loop after PriorAuth. An agent that *actually submits* the appeal.

**🤖 [Try the live demo →](https://autofill-claims.streamlit.app/)** *(deploy URL — replace once published)*

PriorAuth Assist drafts the appeal. **AutoFill submits it.** A browser-driving agent that takes a structured complaint (insurer, denial reason, narrative, complainant info) and fills out a real public-record insurance complaint form on a state insurance commissioner's portal — using Anthropic's Computer Use API and Playwright as the browser executor.

The full demo loop:

```
PriorAuth (draft cited appeal)
         ↓ structured Appeal object
AutoFill (fill state DOI complaint form)
         ↓ filled form, paused at submit
Human reviews → clicks submit
```

## The engineering wedge

The interesting work is the **safety + executor split**:

- **Computer Use as the brain, Playwright as the body.** Claude (via Anthropic's Computer Use API) decides what to click and type. Playwright actually moves the cursor in a Chromium instance. This is lighter than the Docker reference implementation Anthropic ships, easier to test, and works in a Streamlit container locally.
- **"Fill but don't submit" by default.** The agent's system prompt and a hard `--dry-run` flag prevent it from ever clicking the final Submit button on a real DOI form. Demos show the filled form ready to be reviewed by a human, never an actual filing.
- **Step-by-step playback for the public demo.** A recorded session of the agent filling the CA DOI form replays in Streamlit Cloud (no browser required, no API key required). Live mode runs the actual Computer Use loop locally.

## Quick start — local UI (mock playback)

```bash
cd autofill
uv sync
uv run streamlit run src/autofill/ui/app.py
# Watch the recorded session of the agent filling the CA DOI form
```

## Quick start — live Computer Use (your API key)

```bash
cd autofill
uv sync --extra browser
uv run playwright install chromium
export ANTHROPIC_API_KEY=...

autofill submit data/complaints/glp1_denial.yaml --target ca_doi --dry-run
# Opens Chromium, the agent fills the form, stops before submit
```

`--dry-run` is the default. Even with `--no-dry-run` the agent's system prompt instructs it to halt before clicking Submit unless the human in the loop explicitly approves.

## Architecture

```
autofill/
├── src/autofill/
│   ├── models.py           ← ComplaintInput, FormTarget, SubmissionStep, SubmissionResult
│   ├── targets.py          ← registry of supported forms (CA DOI to start)
│   ├── llm.py              ← Anthropic SDK wrapper with Computer Use beta header
│   ├── agent.py            ← the Computer Use loop (real, requires Playwright)
│   ├── executor.py         ← Playwright bridge — turns Claude's tool_use into browser actions
│   ├── mock.py             ← canned step sequence for demo-mode playback
│   ├── cli.py
│   └── ui/app.py
├── data/
│   ├── complaints/         ← synthetic complaint inputs (one per scenario)
│   └── recordings/         ← captured playback files for mock mode
└── tests/
```

## Why a state DOI form (not a real insurer portal)

State insurance commissioner complaint portals are **public-record, no-login required**, *built to take consumer complaints when an insurer denies a claim.* California, New York, Texas, and Washington all have webforms accepting: complainant info, insurer name, claim/policy number, denial reason, narrative.

Targeting these forms is **legitimate** — they exist specifically for the use case this agent automates. Targeting an insurer's member portal would risk ToS violations and require credentials. The state DOI is the right surface.

v0.1 ships **California Department of Insurance** as the first target. Future targets:

- **New York DFS** — different field layout, tests cross-state generalization
- **Texas DOI** — same
- **HealthCare.gov marketplace appeal forms** — federal, harder
- **CMS Standard Appeals (PDF)** — different category (PDF form-filling, not browser)

## Roadmap

- **v0.1** *(current)* — CA DOI target, 1 synthetic complaint, mock playback in UI, real Computer Use via Playwright when run locally.
- **v0.2** — Two more state DOI targets (NY, TX). Cross-target generalization test.
- **v0.3** — Direct PriorAuth Appeal → AutoFill ComplaintInput conversion. The full *draft → submit* loop in one button.
- **v0.4** — Form-field caching: build a structured schema of each form's fields by visiting it once, so subsequent fills are deterministic instead of Computer-Use-driven.

## Safety + ethics

- **No real submissions in the public demo.** Mock mode replays a recording; no HTTP requests to any real form.
- **`--dry-run` is the default in live mode.** The agent fills the form but never clicks Submit unless explicitly authorized.
- **All complaint data is synthetic.** No real PHI, no real complainants, no real insurer disputes — only invented scenarios that mirror the shape of real ones.
- **Public forms only.** The bundled targets are state insurance commissioner public complaint portals. Adding a target requires confirming the form is meant for unauthenticated consumer use.

## License

MIT.
