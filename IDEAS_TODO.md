# Ideas & TODO

A persistent tracker so threads don't get lost between sessions.

## ⚙️ Setup / how to work
- Machine: **MacBook Air M2, 8 GB RAM, disk ~2.3 GB free** → keep everything
  LIGHTWEIGHT. No heavy WebGL/3D, no big installs, no headless-browser
  screenshots, no full-disk `du` scans. Kill specific PIDs, never whole apps.
- Workspace: launch Claude Code from `~/Documents/GitHub` so ONE session sees
  both **my-portfolio** (Astro site) and **RL_agents** (this demos repo).
  Do NOT git-merge them yet (disk risk + would force repointing ~12 Streamlit
  deploys). Portfolio site should just LINK to demos.

## 💡 Option (not committed) — Agent Engineer take-home
Not the current priority — still brainstorming direction. Parked idea:
LangGraph + LangChain + LangSmith agent, RAG + ≥1 tool + eval + traces.
- Plan: **"Benefits & Prior-Auth Triage Assistant"** (healthcare — plays to
  priorauth/autofill domain; PII-redaction guardrail is natural). Music/tour
  alt possible.
- Graph: intake → PII-redact → classify → {RAG policy answer | tool eligibility
  | escalate} → guardrail → stream response.
- Lean deps: **BM25 retrieval** (no torch/embeddings), Claude via
  `langchain-anthropic`. New folder in RL_agents, `src/` + `uv` convention.
- Build in parts: (1) scaffold+KB+graph skeleton (2) tool+RAG (3) guardrails+
  streaming (4) LangSmith eval + README + mermaid diagram. Then record walkthrough video.

## 🔴 Blocking — do this first
- [ ] **Free up disk space.** Disk is ~100% full (only ~1.3 GB free of 228 GB).
      This is what makes the Mac crash/lag — macOS has no room to swap.
      → System Settings → General → Storage → Manage. Aim for 20–30 GB free.
- [ ] (optional) delete `albumjourney/node_modules` (~248 MB, regenerable with
      `npm install`) to reclaim a little.

## 🟢 Active interest — Quant / algorithmic trading
Background fit: CS + finance + ML → closest to **quant developer** and
**independent algo trading**. Viable. Lightweight on this machine (no GPU).
- [x] Weekend starter: backtest ONE simple strategy (MA crossover on SPY) —
      built in `quant/backtest.py` (real Stooq data, honest costs + metrics).
      Run: `cd quant && python3 backtest.py`
- [ ] Learn the stack: pandas, numpy, `backtesting.py` / `vectorbt`; metrics
      (returns, vol, Sharpe, drawdown, CAGR).
- [ ] Fill gaps vs ML: time series (ARIMA/GARCH/Kalman/cointegration),
      financial econometrics (CAPM, Fama–French), market microstructure.
- [ ] Books: Chan (Algorithmic Trading), Lopez de Prado (Advances in Fin ML),
      Harris (Trading and Exchanges), Hull (Options/Futures).
- [ ] Frame as learning quant *research*, not get-rich — skills transfer to
      quant dev / risk / financial-ML jobs regardless.
- Possible next step: scaffold a starter backtest project in this repo.

## 🟢 Emergence / complexity demo (`emergence/`)
Biological-emergence showcase, lightweight 2D canvas (safe for this machine).
- [x] Flocking / Boids — `emergence/index.html` (interactive, tunable, mouse=predator).
- [ ] Add reaction–diffusion (Gray–Scott, animal patterns) — recommended next.
- [ ] Add Physarum slime-mold and/or predator–prey; wrap in a gallery nav.

## 🟢 Quant Streamlit demo — deploy pending
- `quant/app.py` built (interactive backtest, dark navy/gold theme). Not deployed.
- [ ] Deploy on share.streamlit.io: repo `anafe03/RL_agents`, branch main,
      main file `quant/app.py` → get `https://<name>.streamlit.app`, link on site.

## ⏸️ Deferred — Blender cinematic audio-reactive visuals
Come back after disk is freed (renders/caches need many GB).
- Idea: use Blender's **Bake Sound to F-Curve** to drive a scene from a WAV —
  proper cinematic visuals for the album.
- Note: Blender makes *rendered video*, not a playable game. Playable+immersive
  would be Blender (assets) + Godot (engine).

## ⏸️ Paused — "Playable album" web experience (`albumjourney/`)
Vite + React + three.js audio-reactive visualizer (Milestone 1 built).
- Status: loads without JS error, but WebGL + full disk overwhelmed the machine.
- Revisit only after disk is freed; run lean (low particle count, bloom off).
- Reference: `alsworld/` (.als parser + Ableton OSC bridge) — lighter, works.

## 💤 Other threads (from earlier)
- Healthcare job applications + PriorAuth/AutoFill startup idea.
- Standalone browser games: flappybeer, drunkwalk, nightdrive, abyssdiver,
  neonbloom (all work, unrelated to the album/quant directions).
