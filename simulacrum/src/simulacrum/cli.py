"""Simulacrum CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from simulacrum.engine import run_scenario
from simulacrum.models import load_scenario
from simulacrum.render import render_terminal, write_markdown

app = typer.Typer(help="Simulacrum — multi-agent scenario engine.")
console = Console()


@app.command()
def run(
    scenario_dir: Path = typer.Argument(..., help="Path to a scenario directory."),
    ticks: int = typer.Option(None, help="Override scenario max_ticks."),
    transcript: Path = typer.Option(None, help="Optional path to write markdown transcript."),
) -> None:
    """Run a scenario and stream the dialogue to the terminal."""
    scenario = load_scenario(scenario_dir)
    console.print(f"Running [bold]{scenario.title or scenario.name}[/bold]...\n")
    t = run_scenario(scenario, max_ticks=ticks)
    render_terminal(scenario, t, console)
    if transcript is not None:
        write_markdown(scenario, t, transcript)
        console.print(f"\n[green]Wrote markdown transcript → {transcript}[/green]")


@app.command(name="list-scenarios")
def list_scenarios(
    scenarios_dir: Path = typer.Option(Path("scenarios"), help="Where to look."),
) -> None:
    """List scenario folders under `scenarios/`."""
    if not scenarios_dir.exists():
        console.print(f"[red]No scenarios dir at {scenarios_dir}[/red]")
        raise typer.Exit(code=1)
    for child in sorted(scenarios_dir.iterdir()):
        if not child.is_dir():
            continue
        try:
            sc = load_scenario(child)
            console.print(f"  [cyan]{sc.name}[/cyan]  {sc.title}")
        except FileNotFoundError:
            continue


if __name__ == "__main__":
    app()
