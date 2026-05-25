"""rehablens CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rehablens.analyzer import analyze
from rehablens.exercises import EXERCISES, get_exercise
from rehablens.models import FormStatus
from rehablens.pose import detect_pose

app = typer.Typer(help="rehablens — vision-based motion analysis for PM&R rehab.")
console = Console()

_STATUS_STYLE = {
    FormStatus.OK: "[green]✓[/green]",
    FormStatus.WARN: "[yellow]⚠[/yellow]",
    FormStatus.FAIL: "[red]✗[/red]",
    FormStatus.UNKNOWN: "[dim]·[/dim]",
}


@app.command()
def exercises() -> None:
    """List the exercises rehablens knows."""
    table = Table(title="Exercises")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Description")
    for ex in EXERCISES.values():
        table.add_row(ex.id, ex.name, ex.description)
    console.print(table)


@app.command()
def analyze_cmd(
    image: Path = typer.Argument(..., help="Path to a photo of the patient."),
    exercise: str = typer.Option("squat", help="Exercise id (see `rehablens exercises`)."),
) -> None:
    """Run pose detection + form checks on one image."""
    if not image.exists():
        console.print(f"[red]No such file:[/red] {image}")
        raise typer.Exit(code=1)
    ex = get_exercise(exercise)

    frame = detect_pose(image.read_bytes())
    if not frame.detected:
        console.print("[red]No pose detected.[/red] Try a clearer, full-body photo.")
        raise typer.Exit(code=1)

    result = analyze(frame, ex)

    console.print(f"Exercise: [bold]{ex.name}[/bold]")
    console.print(f"Overall:  {_STATUS_STYLE[result.overall]} {result.summary}\n")

    table = Table(title="Form checks")
    table.add_column("", justify="center")
    table.add_column("Measurement", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Target")
    for check in result.checks:
        value = "—" if check.value is None else f"{check.value:g} {check.unit}"
        table.add_row(_STATUS_STYLE[check.status], check.name, value, check.target or "—")
    console.print(table)


if __name__ == "__main__":
    app()
