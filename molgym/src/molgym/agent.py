"""RL agents — tabular Q-learning, and a random-search baseline.

Q-learning is the canonical first RL algorithm: a table of action values
per state, updated toward `reward + gamma * max(next state's values)`,
with an epsilon-greedy policy that explores early and exploits later.
"""

from __future__ import annotations

import random

State = tuple[int, ...]


class QLearningAgent:
    """Tabular Q-learning with an epsilon-greedy policy."""

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.5,  # learning rate
        gamma: float = 0.95,  # discount factor
        epsilon: float = 1.0,  # initial exploration rate
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q: dict[State, list[float]] = {}
        self._rng = random.Random(seed)

    def _row(self, state: State) -> list[float]:
        if state not in self.q:
            self.q[state] = [0.0] * self.n_actions
        return self.q[state]

    def choose(self, state: State, greedy: bool = False) -> int:
        """Pick an action. Epsilon-greedy unless `greedy` is forced."""
        row = self._row(state)
        if not greedy and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_actions)
        best = max(row)
        # Break ties randomly so the policy isn't biased toward low indices.
        candidates = [i for i, v in enumerate(row) if v == best]
        return self._rng.choice(candidates)

    def update(self, state: State, action: int, reward: float,
               next_state: State, done: bool) -> None:
        row = self._row(state)
        future = 0.0 if done else self.gamma * max(self._row(next_state))
        row[action] += self.alpha * (reward + future - row[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class RandomAgent:
    """Picks actions uniformly at random — the baseline Q-learning must beat."""

    def __init__(self, n_actions: int, seed: int = 0) -> None:
        self.n_actions = n_actions
        self._rng = random.Random(seed)

    def choose(self, state: State, greedy: bool = False) -> int:
        return self._rng.randrange(self.n_actions)
