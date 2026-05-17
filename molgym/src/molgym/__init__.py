"""molgym — a reinforcement-learning gym for molecules."""

from molgym.agent import QLearningAgent, RandomAgent
from molgym.env import OBJECTIVES, MoleculeEnv
from molgym.scaffolds import SCAFFOLDS, Scaffold, Substituent
from molgym.train import TrainResult, train

__version__ = "0.0.1"

__all__ = [
    "OBJECTIVES",
    "SCAFFOLDS",
    "MoleculeEnv",
    "QLearningAgent",
    "RandomAgent",
    "Scaffold",
    "Substituent",
    "TrainResult",
    "train",
]
