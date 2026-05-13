"""Octagon CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from octagon.defenders import REGISTRY, get_defender
from octagon.report import render_terminal, write_markdown
from octagon.runner import load_attacks, run_audit

app = typer.Typer(help="Octagon — adversarial audit and tournament platform for LLM agents.")
console = Console()


@app.command()
def attack(
    defender: str = typer.Option("internal_it", help="Defender to test."),
    attacks_dir: Path = typer.Option(Path("attacks"), help="Directory of attack YAML files."),
    report: Path = typer.Option(
        None, help="Optional path to write a markdown audit report."
    ),
) -> None:
    """Run every attack against a defender and produce an audit report."""
    defender_inst = get_defender(defender)
    attacks = load_attacks(attacks_dir)
    if not attacks:
        console.print(f"[red]No attacks found under {attacks_dir}[/red]")
        raise typer.Exit(code=1)
    console.print(
        f"Loaded {len(attacks)} attacks. Running against [bold]{defender}[/bold]...\n"
    )
    audit = run_audit(defender_inst, attacks)
    render_terminal(audit, console)
    if report is not None:
        write_markdown(audit, report)
        console.print(f"\n[green]Wrote markdown report → {report}[/green]")


@app.command(name="list-attacks")
def list_attacks_cmd(
    attacks_dir: Path = typer.Option(Path("attacks"), help="Directory of attack YAML files."),
) -> None:
    """List all loaded attacks."""
    attacks = load_attacks(attacks_dir)
    for a in attacks:
        console.print(f"  [cyan]{a.id}[/cyan]  [{a.category.value}]  {a.name}")
    console.print(f"\nTotal: {len(attacks)}")


@app.command(name="list-defenders")
def list_defenders_cmd() -> None:
    """List available defenders."""
    for name, cls in sorted(REGISTRY.items()):
        console.print(f"  [cyan]{name}[/cyan]  {cls.__name__}")


if __name__ == "__main__":
    app()


@app.command()
def healthcheck(
    model: str = "gpt-4o-mini",
) -> None:
    """One LLM call against the real API to verify your key + provider work.

    Defaults to gpt-4o-mini (cheap). Pass --model claude-haiku-4-5 etc to test other providers.
    """
    from octagon import llm
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
