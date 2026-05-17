"""The training loop and policy rollout.

`train` runs episodes of the molecule environment, lets the agent learn,
and tracks both the per-episode return and the best molecule found so far.
`greedy_trajectory` then rolls out the learned policy once, recording the
step-by-step optimization path for display.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from molgym.agent import QLearningAgent, RandomAgent
from molgym.chem import compute_properties
from molgym.env import MoleculeEnv
from molgym.scaffolds import group_names


@dataclass
class StepRecord:
    """One molecule along the learned optimization path."""

    state: tuple[int, ...]
    smiles: str
    groups: list[str]
    score: float


@dataclass
class TrainResult:
    episode_returns: list[float] = field(default_factory=list)
    best_score_curve: list[float] = field(default_factory=list)  # best-so-far per episode
    best_state: tuple[int, ...] = ()
    best_smiles: str = ""
    best_score: float = 0.0
    trajectory: list[StepRecord] = field(default_factory=list)


def _record(env: MoleculeEnv, state: tuple[int, ...]) -> StepRecord:
    return StepRecord(
        state=state,
        smiles=env.smiles(state),
        groups=group_names(state),
        score=round(env.score(state), 4),
    )


def train(
    env: MoleculeEnv,
    agent: QLearningAgent | RandomAgent,
    episodes: int = 2000,
) -> TrainResult:
    """Run `episodes` of training. Works for both the Q-learner and the
    random baseline (the baseline simply has nothing to update)."""
    result = TrainResult()
    best_score = -1.0
    best_state: tuple[int, ...] = env.reset()
    learner = isinstance(agent, QLearningAgent)

    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        done = False
        while not done:
            action = agent.choose(state)
            next_state, reward, done = env.step(action)
            if learner:
                agent.update(state, action, reward, next_state, done)
            total += reward
            state = next_state
            score = env.score(state)
            if score > best_score:
                best_score = score
                best_state = state
        if learner:
            agent.decay_epsilon()
        result.episode_returns.append(round(total, 4))
        result.best_score_curve.append(round(best_score, 4))

    result.best_state = best_state
    result.best_smiles = env.smiles(best_state)
    result.best_score = round(best_score, 4)
    result.trajectory = greedy_trajectory(env, agent)
    return result


def greedy_trajectory(
    env: MoleculeEnv,
    agent: QLearningAgent | RandomAgent,
) -> list[StepRecord]:
    """Roll out the agent's greedy policy once, recording every molecule."""
    state = env.reset()
    records = [_record(env, state)]
    done = False
    while not done:
        action = agent.choose(state, greedy=True)
        state, _, done = env.step(action)
        records.append(_record(env, state))
    return records


def best_properties(result: TrainResult):
    """RDKit properties of the best molecule found."""
    return compute_properties(result.best_smiles)
