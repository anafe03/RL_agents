"""RDKit chemistry layer — compute molecular properties and render molecules.

Property computation is memoized: the molecular MDP has only a few hundred
distinct molecules, so after the first pass every lookup is free. That is
what lets the Streamlit demo train an agent live in under a second.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Properties:
    """RDKit-computed properties of one molecule."""

    smiles: str
    valid: bool
    qed: float  # Quantitative Estimate of Drug-likeness, 0..1
    mol_weight: float
    logp: float  # calculated octanol-water partition coefficient
    h_donors: int
    h_acceptors: int
    rotatable_bonds: int
    rings: int


_INVALID = Properties("", False, 0.0, 0.0, 0.0, 0, 0, 0, 0)


@lru_cache(maxsize=8192)
def compute_properties(smiles: str) -> Properties:
    """Compute properties for a SMILES string. Memoized on the string."""
    from rdkit import Chem
    from rdkit.Chem import QED, Crippen, Descriptors, Lipinski

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _INVALID
    return Properties(
        smiles=smiles,
        valid=True,
        qed=round(QED.qed(mol), 4),
        mol_weight=round(Descriptors.MolWt(mol), 2),
        logp=round(Crippen.MolLogP(mol), 3),
        h_donors=Lipinski.NumHDonors(mol),
        h_acceptors=Lipinski.NumHAcceptors(mol),
        rotatable_bonds=Descriptors.NumRotatableBonds(mol),
        rings=Descriptors.RingCount(mol),
    )


def lipinski_pass(props: Properties) -> bool:
    """Lipinski's Rule of Five — a rough drug-likeness screen."""
    return (
        props.valid
        and props.mol_weight <= 500
        and props.logp <= 5
        and props.h_donors <= 5
        and props.h_acceptors <= 10
    )


def render_png(smiles: str, size: tuple[int, int] = (300, 230)) -> bytes:
    """Render a molecule to PNG bytes. Returns b'' if the SMILES is invalid."""
    from rdkit import Chem
    from rdkit.Chem import Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return b""
    image = Draw.MolToImage(mol, size=size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
