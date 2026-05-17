"""molgym CLI."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from molgym.agent import QLearningAgent, RandomAgent
from molgym.chem import compute_properties, lipinski_pass
from molgym.env import OBJECTIVES, MoleculeEnv
from molgym.scaffolds import SCAFFOLDS
from molgym.train import train

app = typer.Typer(help="molgym — a reinforcement-learning gym for molecules.")
console = Console()


@app.command()
def scaffolds() -> None:
    """List the available scaffolds."""
    table = Table(title="Scaffolds")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Slots", justify="right")
    table.add_column("Description")
    for s in SCAFFOLDS.values():
        table.add_row(s.id, s.name, str(s.slots), s.description)
    console.print(table)
    console.print(f"\nObjectives: {', '.join(OBJECTIVES)}")


@app.command()
def train_cmd(
    scaffold: str = typer.Option("benzene_135", help="Scaffold id."),
    objective: str = typer.Option("drug_likeness", help="What to optimize."),
    episodes: int = typer.Option(2000, help="Training episodes."),
    baseline: bool = typer.Option(True, help="Also run a random-search baseline."),
) -> None:
    """Train a Q-learning agent to optimize a molecule, and show the result."""
    if scaffold not in SCAFFOLDS:
        console.print(f"[red]Unknown scaffold[/red] — choose from {sorted(SCAFFOLDS)}")
        raise typer.Exit(code=1)
    if objective not in OBJECTIVES:
        console.print(f"[red]Unknown objective[/red] — choose from {sorted(OBJECTIVES)}")
        raise typer.Exit(code=1)

    sc = SCAFFOLDS[scaffold]
    env = MoleculeEnv(sc, objective=objective)

    q_result = train(env, QLearningAgent(env.n_actions), episodes=episodes)
    console.print(f"\n[bold]Q-learning[/bold] — best {objective} = "
                  f"[green]{q_result.best_score:.4f}[/green]")
    console.print(f"  molecule: {q_result.best_smiles}")

    if baseline:
        rand_result = train(env, RandomAgent(env.n_actions), episodes=episodes)
        verdict = "✓ beat" if q_result.best_score >= rand_result.best_score else "✗ lost to"
        console.print(f"[bold]Random search[/bold] — best = {rand_result.best_score:.4f}  "
                      f"[dim](Q-learning {verdict} the baseline)[/dim]")

    props = compute_properties(q_result.best_smiles)
    table = Table(title="\nBest molecule found")
    table.add_column("Property", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("QED (drug-likeness)", f"{props.qed:.3f}")
    table.add_row("Mol. weight", f"{props.mol_weight:g}")
    table.add_row("logP", f"{props.logp:g}")
    table.add_row("H-bond donors", str(props.h_donors))
    table.add_row("H-bond acceptors", str(props.h_acceptors))
    table.add_row("Lipinski Rule of Five", "PASS" if lipinski_pass(props) else "FAIL")
    console.print(table)

    console.print("\n[bold]Learned optimization path:[/bold]")
    for i, step in enumerate(q_result.trajectory):
        console.print(f"  {i}. {step.smiles}  [dim]({objective} {step.score:.3f})[/dim]")


if __name__ == "__main__":
    app()
