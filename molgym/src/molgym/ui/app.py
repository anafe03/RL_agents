"""Streamlit UI for molgym.

Run locally:
    cd molgym
    uv sync
    uv run streamlit run src/molgym/ui/app.py

Training is tabular Q-learning on a small molecular MDP — it runs live in
the browser in well under a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from molgym.agent import QLearningAgent, RandomAgent
from molgym.chem import compute_properties, lipinski_pass, render_png
from molgym.env import OBJECTIVES, MoleculeEnv
from molgym.scaffolds import SCAFFOLDS
from molgym.train import train

st.set_page_config(page_title="molgym", page_icon="⚛️", layout="wide",
                   initial_sidebar_state="expanded")

ACCENT = "#2dd4bf"
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #0c1116; }}
    .mg-header {{
        background: linear-gradient(135deg, #141b22 0%, #0c1116 100%);
        border: 1px solid #1f2a33;
        border-left: 4px solid {ACCENT};
        border-radius: 8px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.6rem;
    }}
    .mg-header .title {{
        font-size: 1.7rem; font-weight: 800; color: {ACCENT};
        font-family: -apple-system, "Segoe UI", sans-serif; letter-spacing: 0.03em;
    }}
    .mg-header .sub {{ color: #8b97a3; font-size: 0.87rem; margin-top: 0.2rem; }}
    .mg-how {{
        background: #141b22; border: 1px solid #1f2a33; border-radius: 8px;
        padding: 0.85rem 1.15rem; margin-bottom: 0.8rem;
        font-size: 0.84rem; color: #aab4be; line-height: 1.6;
        font-family: -apple-system, sans-serif;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

GITHUB_URL = "https://github.com/anafe03/RL_agents/tree/main/molgym"

if "result" not in st.session_state:
    st.session_state.result = None


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.markdown("# ⚛️ molgym")
    st.caption("A reinforcement-learning gym for molecules.")
    st.markdown("---")

    scaffold_id = st.selectbox(
        "Scaffold", list(SCAFFOLDS.keys()),
        format_func=lambda k: SCAFFOLDS[k].name,
        help="The molecular core the agent decorates.",
    )
    scaffold = SCAFFOLDS[scaffold_id]
    st.caption(scaffold.description)

    objective = st.selectbox(
        "Objective", list(OBJECTIVES.keys()),
        help="drug_likeness = QED. solubility = a logP-based score.",
    )
    episodes = st.slider("Training episodes", 200, 5000, 2000, step=200)
    st.markdown("---")
    st.caption(
        f"Scaffold has {scaffold.slots} slots × 8 functional groups → "
        f"{8 ** scaffold.slots} possible molecules. Tabular Q-learning, CPU, "
        "no GPU or API key."
    )
    st.markdown(f"[GitHub repo]({GITHUB_URL})")


# --- header -----------------------------------------------------------------

st.markdown(
    """
    <div class="mg-header">
        <div class="title">molgym</div>
        <div class="sub">an RL agent learns to decorate a molecular scaffold
        to maximize a chemical property</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="mg-how">
    <b style="color:{ACCENT}">How it works</b> &nbsp;
    <b>state</b> = which functional group sits in each scaffold slot &nbsp;·&nbsp;
    <b>action</b> = put a group in a slot &nbsp;·&nbsp;
    <b>reward</b> = the change in the property score.
    The agent plays many episodes, learns a Q-table, and converges on a
    molecule that maximizes the objective. A random-search agent runs
    alongside as the baseline to beat.
    </div>
    """,
    unsafe_allow_html=True,
)


# --- train ------------------------------------------------------------------

if st.button("⚛️ Train the agent", type="primary", use_container_width=True):
    env_q = MoleculeEnv(scaffold, objective=objective)
    env_r = MoleculeEnv(scaffold, objective=objective)
    with st.spinner(f"Training Q-learning for {episodes} episodes..."):
        q_result = train(env_q, QLearningAgent(env_q.n_actions), episodes=episodes)
        r_result = train(env_r, RandomAgent(env_r.n_actions), episodes=episodes)
    st.session_state.result = {
        "scaffold": scaffold.name,
        "objective": objective,
        "q": q_result,
        "random": r_result,
    }


# --- results ----------------------------------------------------------------

result = st.session_state.result
if result is None:
    st.info("Pick a scaffold and objective in the sidebar, then **Train the agent**.")
    st.stop()

q_result = result["q"]
r_result = result["random"]
objective = result["objective"]

# Headline comparison.
c1, c2, c3 = st.columns(3)
c1.metric(f"Q-learning best {objective}", f"{q_result.best_score:.3f}")
c2.metric("Random-search baseline", f"{r_result.best_score:.3f}")
delta = q_result.best_score - r_result.best_score
c3.metric("Q-learning advantage", f"{delta:+.3f}",
          help="How much the learned policy beat random search.")

# Learning curve — best score found so far, per episode.
st.markdown("##### Learning curve — best score found so far")
st.caption("Q-learning should climb and converge; random search drifts up by luck.")
st.line_chart(
    {"Q-learning": q_result.best_score_curve, "random baseline": r_result.best_score_curve},
    height=240,
)

# Best molecule.
st.markdown("##### Best molecule found")
best_col, prop_col = st.columns([1, 1])
with best_col:
    png = render_png(q_result.best_smiles, size=(340, 260))
    if png:
        st.image(png, caption=q_result.best_smiles)
with prop_col:
    props = compute_properties(q_result.best_smiles)
    st.markdown(f"**Objective ({objective}):** `{q_result.best_score:.3f}`")
    st.markdown(f"**QED (drug-likeness):** {props.qed:.3f}")
    st.markdown(f"**Mol. weight:** {props.mol_weight:g}")
    st.markdown(f"**logP:** {props.logp:g}")
    st.markdown(f"**H-bond donors / acceptors:** {props.h_donors} / {props.h_acceptors}")
    st.markdown(f"**Rotatable bonds:** {props.rotatable_bonds}")
    if lipinski_pass(props):
        st.success("Passes Lipinski's Rule of Five")
    else:
        st.warning("Fails Lipinski's Rule of Five")

# The learned optimization path — animated.
st.markdown("##### Learned optimization path")
st.caption("Watch the learned greedy policy decorate the scaffold one edit "
           "at a time. Drag the slider to scrub, or hit Play.")

# Collapse consecutive no-op steps so only real edits show.
_traj = q_result.trajectory
path = [_traj[0]]
for step in _traj[1:]:
    if step.smiles != path[-1].smiles:
        path.append(step)
last = len(path) - 1

stage = st.empty()


def _show_step(i: int) -> None:
    step = path[i]
    with stage.container():
        img_col, info_col = st.columns([3, 2])
        with img_col:
            png = render_png(step.smiles, size=(380, 300))
            if png:
                st.image(png)
        with info_col:
            st.markdown(f"### Step {i} / {last}")
            st.markdown(f"**Groups:** {', '.join(step.groups)}")
            st.markdown(f"`{step.smiles}`")
            st.markdown(
                f"**{objective}:** <span style='color:{ACCENT};font-size:1.5rem;"
                f"font-weight:800'>{step.score:.3f}</span>",
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, max(0.0, step.score)))
            if i > 0:
                delta = step.score - path[i - 1].score
                st.caption(f"last edit changed the score by {delta:+.3f}")


if last > 0:
    step_idx = st.slider("Step", 0, last, last, key="traj_step")
    _show_step(step_idx)
    if st.button("▶ Play optimization", use_container_width=True):
        import time

        for i in range(last + 1):
            _show_step(i)
            time.sleep(0.9)
else:
    _show_step(0)
    st.caption("The agent kept the bare scaffold — try more episodes or another objective.")

st.divider()
st.caption(
    "Teaching-scale RL: a small, fully-enumerable molecular MDP and classic "
    "tabular Q-learning. The reward is real RDKit chemistry (QED, logP, "
    "Lipinski). A portfolio project — not a drug-discovery pipeline."
)
