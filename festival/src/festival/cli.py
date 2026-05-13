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


@app.command()
def healthcheck(
    model: str = "gpt-4o-mini",
) -> None:
    """One LLM call against the real API to verify your key + provider work.

    Defaults to gpt-4o-mini (cheap). Pass --model claude-haiku-4-5 etc to test other providers.
    """
    from festival import llm
    llm.require_api_key(model=model)
    console.print(f"Pinging [bold]{model}[/bold] ...")
    try:
        result = llm.chat(
            model=model,
            system="Reply with exactly one word: pong.",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=16,
        )
    except Exception as e:
        console.print(f"[red]Failed:[/red] {type(e).__name__}: {e}")
        raise typer.Exit(code=1)
    text = getattr(result, "text", None) or getattr(result, "content", "")
    console.print(f"[green]OK[/green] model={getattr(result, 'model', model)} cost=$\{result.cost_usd:.6f\}")
