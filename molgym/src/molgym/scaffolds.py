"""Chemical scaffolds and the substituent library.

A scaffold is a molecular core with N open substitution positions. The RL
agent's job is to choose, for each position, one functional group from the
substituent library. A molecule is built by string-templating the chosen
group SMILES fragments into the scaffold template.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Substituent:
    """One functional group the agent can place in a slot."""

    name: str
    smiles: str  # fragment inserted as a branch on a scaffold atom


# The action vocabulary — small on purpose so tabular Q-learning is tractable.
SUBSTITUENTS: list[Substituent] = [
    Substituent("H", "[H]"),  # an unsubstituted position
    Substituent("hydroxyl", "O"),
    Substituent("methyl", "C"),
    Substituent("fluoro", "F"),
    Substituent("chloro", "Cl"),
    Substituent("amino", "N"),
    Substituent("carboxyl", "C(=O)O"),
    Substituent("methoxy", "OC"),
]

N_GROUPS = len(SUBSTITUENTS)


@dataclass(frozen=True)
class Scaffold:
    """A molecular core with `slots` substitution positions."""

    id: str
    name: str
    template: str  # str.format template with {0}, {1}, ... for each slot
    slots: int
    description: str


SCAFFOLDS: dict[str, Scaffold] = {
    "benzene_135": Scaffold(
        id="benzene_135",
        name="1,3,5-trisubstituted benzene",
        template="c1c({0})cc({1})cc1{2}",
        slots=3,
        description="A benzene ring with three open substitution positions.",
    ),
    "aniline_35": Scaffold(
        id="aniline_35",
        name="3,5-disubstituted aniline",
        template="Nc1cc({0})cc({1})c1",
        slots=2,
        description="An aniline (aminobenzene) with two open positions.",
    ),
}


def build_smiles(scaffold: Scaffold, group_indices: tuple[int, ...]) -> str:
    """Build a SMILES string by placing the chosen groups into the scaffold."""
    fragments = [SUBSTITUENTS[i].smiles for i in group_indices]
    return scaffold.template.format(*fragments)


def group_names(group_indices: tuple[int, ...]) -> list[str]:
    return [SUBSTITUENTS[i].name for i in group_indices]
