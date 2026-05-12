"""Festival Companion CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from festival.matcher import score_lineup
from festival.models import TasteProfile, list_lineups, load_lineup
from festival.render import render_terminal, write_markdown
from festival.scheduler import build_schedule

app = typer.Typer(help="Festival Companion — taste-matched festival schedule planner.")
console = Console()


@app.command()
def plan(
    lineup: Path = typer.Argument(..., help="Path to a lineup YAML."),
    taste: str = typer.Option("", help="Freeform taste description."),
    must_see: list[str] = typer.Option([], "--must-see", help="Artist names you refuse to miss."),
    avoid: list[str] = typer.Option([], "--avoid", help="Artist names to never schedule."),
    favorites: list[str] = typer.Option([], "--fav", help="Favorite artist names to bias toward."),
    report: Path = typer.Option(None, help="Optional path for a markdown report."),
) -> None:
    """Score the lineup and build a personalized day-by-day schedule."""
    festival = load_lineup(lineup)
    profile = TasteProfile(
        description=taste,
        favorite_artists=list(favorites),
        must_see=list(must_see),
        avoid=list(avoid),
    )
    console.print(
        f"Scoring {len(festival.sets)} sets at [bold]{festival.name}[/bold]...\n"
    )
    recs, cost = score_lineup(festival, profile)
    schedule = build_schedule(festival, profile, recs)
    schedule.cost_usd = cost
    render_terminal(schedule, console)
    if report is not None:
        write_markdown(schedule, report)
        console.print(f"\n[green]Wrote markdown report → {report}[/green]")


@app.command(name="list-lineups")
def list_lineups_cmd(
    lineups_dir: Path = typer.Option(Path("data/lineups"), help="Lineups directory."),
) -> None:
    """List bundled / on-disk festival lineups."""
    for name, p in list_lineups(lineups_dir):
        console.print(f"  [cyan]{p.stem}[/cyan]  {name}")


if __name__ == "__main__":
    app()
