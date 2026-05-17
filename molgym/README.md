# molgym

**A reinforcement-learning gym for molecules.**

An RL agent starts from a bare chemical scaffold and learns, edit by edit,
to decorate it with functional groups that maximize a target property —
drug-likeness, solubility, whatever you point it at. RDKit does the
chemistry; tabular Q-learning does the learning.

```
  scaffold + substituent slots          the RL loop
  ──────────────────────────            ───────────
  state   = which group is in each slot
  action  = put group G in slot S
  reward  = change in the property score (QED, logP-based, ...)

  episode 1     ▒▒▒░░░░░   QED 0.41
  episode 500   ▒▒▒▒▒▒░░   QED 0.78
  episode 2000  ▒▒▒▒▒▒▒▒   QED 0.91   ← learned policy
```

## Why this project

It's a deliberate two-for-one. The **reward function is the chemistry**:
drug-likeness is scored by QED, built from molecular weight, logP, hydrogen-
bond donors/acceptors, and rotatable bonds — exactly the concepts in an
intro chemistry course. The **loop is the RL**: states, actions, rewards,
an epsilon-greedy policy, a Q-table. Learn both at once, and watch a
molecule visibly improve while you do.

## Quickstart

```bash
cd molgym
uv sync
uv run streamlit run src/molgym/ui/app.py        # the demo
uv run molgym train --objective drug_likeness    # train from the CLI
uv run molgym scaffolds                          # list scaffolds
```

## How it works

- **Scaffold** — a molecular core with a few open substitution positions
  (e.g. a 1,3,5-trisubstituted benzene with three slots).
- **Substituents** — a small library of functional groups: hydroxyl,
  methyl, fluoro, amino, carboxyl, methoxy, and so on.
- **Environment** (`env.py`) — state = the group in each slot; an action
  sets one slot; reward = the change in the objective score.
- **Agent** (`agent.py`) — tabular Q-learning with an epsilon-greedy
  policy, plus a random-search agent as a baseline to beat.
- **Objectives** — `drug_likeness` (QED) and `solubility` (a logP-based
  score). Each is a function of RDKit-computed properties.

Everything runs on CPU in well under a second — no GPU, no API key, no
network. The Streamlit demo trains live.

## Scope, honestly

This is a **teaching-scale** project: a small, fully-enumerable molecular
MDP and classic tabular Q-learning — the canonical way to learn RL, here
applied to a real chemistry reward. It is not REINVENT-scale generative
chemistry, and the molecules are decorated scaffolds, not novel drugs.
The point is to learn RL and cheminformatics together and see it work.
