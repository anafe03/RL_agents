"""AutoFill CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from autofill.models import load_complaint
from autofill.targets import REGISTRY, get_target

app = typer.Typer(help="AutoFill — Computer Use agent for public insurance complaint forms.")
console = Console()


@app.command()
def submit(
    complaint_path: Path = typer.Argument(..., help="Path to a complaint YAML."),
    target: str = typer.Option("ca_doi", help="Form target id."),
    dry_run: bool = typer.Option(
        True,
        help="If true (default), agent stops BEFORE clicking Submit so a human can review.",
    ),
    model: str = typer.Option("claude-sonnet-4-6", help="Anthropic model id."),
) -> None:
    """Run the live Computer Use agent against a target form.

    Requires ANTHROPIC_API_KEY and `playwright install chromium` (see README).
    """
    complaint = load_complaint(complaint_path)
    tgt = get_target(target)
    console.print(f"Complaint: [bold]{complaint.id}[/bold]")
    console.print(f"Target:    [bold]{tgt.name}[/bold]  ({tgt.url})")
    console.print(f"Dry-run:   [bold]{dry_run}[/bold]\n")

    try:
        from autofill.agent import run_submission
    except ImportError as e:
        console.print(f"[red]Failed to import agent: {e}[/red]")
        console.print(
            "Live submission needs Playwright. Install with: "
            "`uv sync --extra browser` and `uv run playwright install chromium`"
        )
        raise typer.Exit(code=1) from e

    result = run_submission(complaint, tgt, dry_run=dry_run, model=model)
    console.print(f"\n[green]Completed {result.step_count} steps[/green]")
    console.print(f"Cost: ${result.cost_usd:.4f}")
    if result.error:
        console.print(f"[red]Error: {result.error}[/red]")


@app.command(name="list-targets")
def list_targets_cmd() -> None:
    """List supported form targets."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("State")
    table.add_column("URL")
    for tid, t in REGISTRY.items():
        table.add_row(tid, t.name, t.state, t.url)
    console.print(table)


@app.command(name="list-complaints")
def list_complaints_cmd(
    complaints_dir: Path = typer.Option(Path("data/complaints"), help="Complaints directory."),
) -> None:
    """List bundled complaint inputs."""
    for p in sorted(complaints_dir.glob("*.yaml")):
        c = load_complaint(p)
        console.print(f"  [cyan]{p.stem}[/cyan]  {c.insurer_name} — {c.requested_service[:60]}")


@app.command(name="play-mock")
def play_mock_cmd(
    target: str = typer.Option("ca_doi"),
    complaint: str = typer.Option("glp1_denial"),
) -> None:
    """Print the bundled mock playback for (target, complaint), step by step."""
    from autofill.mock import get_mock_run

    result = get_mock_run(target, complaint)
    if result is None:
        console.print(f"[red]No mock playback for target={target}, complaint={complaint}[/red]")
        raise typer.Exit(code=1)
    for s in result.steps:
        console.print(f"[cyan][{s.step_id:02d}][/cyan] [bold]{s.action.value:>10}[/bold]  {s.target_label}")
        if s.value:
            console.print(f"           value: \"{s.value[:100]}\"")
        if s.narration:
            console.print(f"           [dim]{s.narration}[/dim]")


if __name__ == "__main__":
    app()
