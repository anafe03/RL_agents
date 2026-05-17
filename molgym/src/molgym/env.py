"""The molecule RL environment.

A small, fully-enumerable Markov decision process:
  - state  = a tuple of group indices, one per scaffold slot
  - action = (slot, group) — put `group` in `slot`
  - reward = the change in the objective score caused by that edit

Because reward is the *change* in score, an episode's total return equals
the score of the final molecule minus the score of the starting one.
"""

from __future__ import annotations

from collections.abc import Callable

from molgym.chem import Properties, compute_properties
from molgym.scaffolds import N_GROUPS, Scaffold, build_smiles


def _drug_likeness(props: Properties) -> float:
    """QED — a 0..1 estimate of drug-likeness."""
    return props.qed


def _solubility(props: Properties) -> float:
    """A 0..1 solubility-favoring score: higher when logP is low.

    Aqueous solubility tracks inversely with logP; this maps a logP of
    roughly -2..4 onto 1..0.
    """
    return max(0.0, min(1.0, (4.0 - props.logp) / 6.0))


OBJECTIVES: dict[str, Callable[[Properties], float]] = {
    "drug_likeness": _drug_likeness,
    "solubility": _solubility,
}


class MoleculeEnv:
    """A decorate-the-scaffold RL environment."""

    def __init__(
        self,
        scaffold: Scaffold,
        objective: str = "drug_likeness",
        max_steps: int = 6,
    ) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective: {objective!r}. "
                             f"Choose from {sorted(OBJECTIVES)}.")
        self.scaffold = scaffold
        self.objective_name = objective
        self._objective = OBJECTIVES[objective]
        self.max_steps = max_steps
        # Every (slot, group) pair is one action.
        self.actions: list[tuple[int, int]] = [
            (slot, group)
            for slot in range(scaffold.slots)
            for group in range(N_GROUPS)
        ]
        self.state: tuple[int, ...] = tuple([0] * scaffold.slots)
        self.steps = 0

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def reset(self) -> tuple[int, ...]:
        """Reset to the bare scaffold (every slot unsubstituted)."""
        self.state = tuple([0] * self.scaffold.slots)
        self.steps = 0
        return self.state

    def smiles(self, state: tuple[int, ...]) -> str:
        return build_smiles(self.scaffold, state)

    def score(self, state: tuple[int, ...]) -> float:
        """The objective score of a state, in 0..1."""
        props = compute_properties(self.smiles(state))
        return self._objective(props) if props.valid else 0.0

    def step(self, action_index: int) -> tuple[tuple[int, ...], float, bool]:
        """Apply an action. Returns (new_state, reward, done)."""
        slot, group = self.actions[action_index]
        before = self.score(self.state)
        new_state = list(self.state)
        new_state[slot] = group
        self.state = tuple(new_state)
        after = self.score(self.state)
        self.steps += 1
        reward = after - before
        done = self.steps >= self.max_steps
        return self.state, reward, done
