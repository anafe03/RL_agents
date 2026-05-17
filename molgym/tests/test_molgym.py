"""Tests for molgym — chemistry layer, the RL environment, and learning.

Everything is CPU and deterministic. The learning tests use fixed seeds so
"the agent improved" is a reproducible assertion, not a flaky one.
"""

from __future__ import annotations

from itertools import product

from molgym.agent import QLearningAgent, RandomAgent
from molgym.chem import compute_properties, lipinski_pass, render_png
from molgym.env import OBJECTIVES, MoleculeEnv
from molgym.scaffolds import N_GROUPS, SCAFFOLDS, build_smiles
from molgym.train import greedy_trajectory, train


# -- chemistry ---------------------------------------------------------------

def test_every_substituent_combo_builds_a_valid_molecule():
    """Catches any scaffold template / substituent fragment that RDKit rejects."""
    from rdkit import Chem

    for scaffold in SCAFFOLDS.values():
        for combo in product(range(N_GROUPS), repeat=scaffold.slots):
            smiles = build_smiles(scaffold, combo)
            assert Chem.MolFromSmiles(smiles) is not None, (
                f"{scaffold.id} {combo} produced invalid SMILES: {smiles}"
            )


def test_compute_properties_benzene():
    props = compute_properties("c1ccccc1")
    assert props.valid
    assert props.rings == 1
    assert 0.0 <= props.qed <= 1.0


def test_compute_properties_invalid_smiles():
    props = compute_properties("this is not a molecule")
    assert not props.valid


def test_lipinski_pass_on_small_molecule():
    assert lipinski_pass(compute_properties("c1ccccc1O"))  # phenol — small, drug-like


def test_render_png_returns_png_bytes():
    png = render_png("c1ccccc1")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


# -- environment -------------------------------------------------------------

def test_env_reset_is_bare_scaffold():
    env = MoleculeEnv(SCAFFOLDS["benzene_135"])
    state = env.reset()
    assert state == (0, 0, 0)  # all H


def test_env_score_in_unit_range():
    for name in OBJECTIVES:
        env = MoleculeEnv(SCAFFOLDS["benzene_135"], objective=name)
        for combo in product(range(N_GROUPS), repeat=3):
            assert 0.0 <= env.score(combo) <= 1.0


def test_env_step_reward_is_score_delta():
    env = MoleculeEnv(SCAFFOLDS["benzene_135"], "drug_likeness")
    start = env.reset()
    before = env.score(start)
    new_state, reward, _ = env.step(1)  # set slot 0 to a non-H group
    assert abs(reward - (env.score(new_state) - before)) < 1e-9


def test_env_episode_ends_at_max_steps():
    env = MoleculeEnv(SCAFFOLDS["benzene_135"], max_steps=4)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done = env.step(0)
        steps += 1
    assert steps == 4


# -- learning ----------------------------------------------------------------

def test_q_learning_improves_over_the_bare_scaffold():
    env = MoleculeEnv(SCAFFOLDS["benzene_135"], "drug_likeness")
    start_score = env.score(env.reset())
    result = train(env, QLearningAgent(env.n_actions, seed=1), episodes=1500)
    assert result.best_score > start_score, "Q-learning found nothing better than the bare scaffold"
    assert result.trajectory, "no greedy trajectory recorded"


def test_learned_greedy_policy_reaches_a_strong_molecule():
    env = MoleculeEnv(SCAFFOLDS["benzene_135"], "drug_likeness")
    agent = QLearningAgent(env.n_actions, seed=2)
    result = train(env, agent, episodes=2500)
    # The greedy rollout of the learned policy should land near the best
    # molecule the agent found anywhere during training.
    final = result.trajectory[-1].score
    assert final >= result.best_score - 0.1, (
        f"greedy policy ended at {final}, far below best {result.best_score}"
    )


def test_q_learning_is_competitive_with_random_search():
    # The MDP is tiny, so random search also finds good molecules — but the
    # Q-learner should at least match it.
    env_q = MoleculeEnv(SCAFFOLDS["benzene_135"], "drug_likeness")
    env_r = MoleculeEnv(SCAFFOLDS["benzene_135"], "drug_likeness")
    q = train(env_q, QLearningAgent(env_q.n_actions, seed=1), episodes=2000)
    r = train(env_r, RandomAgent(env_r.n_actions, seed=1), episodes=2000)
    assert q.best_score >= r.best_score - 0.05


def test_greedy_trajectory_length_bounded_by_max_steps():
    env = MoleculeEnv(SCAFFOLDS["aniline_35"], "solubility", max_steps=5)
    agent = QLearningAgent(env.n_actions, seed=0)
    train(env, agent, episodes=300)
    traj = greedy_trajectory(env, agent)
    assert 1 < len(traj) <= 6  # start state + up to max_steps edits


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
