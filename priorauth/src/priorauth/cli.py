"""PriorAuth Assist CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from priorauth.assessor import assess_appeal
from priorauth.drafter import draft_appeal
from priorauth.models import load_case, load_guideline_corpus
from priorauth.render import render_terminal, write_markdown
from priorauth.retriever import retrieve_relevant

app = typer.Typer(help="PriorAuth Assist — cited prior-authorization appeal drafter.")
console = Console()


@app.command()
def appeal(
    case_path: Path = typer.Argument(..., help="Path to a case YAML."),
    guidelines_dir: Path = typer.Option(Path("data/guidelines"), help="Guideline corpus directory."),
    report: Path = typer.Option(None, help="Optional path to write the markdown appeal."),
) -> None:
    """Run the full retriever → drafter → assessor pipeline on a case."""
    case = load_case(case_path)
    corpus = load_guideline_corpus(guidelines_dir)
    console.print(f"Loaded case [bold]{case.id}[/bold] and {len(corpus)} guidelines.\n")
    console.print("→ Retrieving relevant guidelines...")
    selected, retr_cost = retrieve_relevant(case, corpus)
    console.print(f"  Selected: {', '.join(g.id for g in selected)}\n")
    console.print("→ Drafting appeal...")
    appeal, draft_cost = draft_appeal(case, selected)
    console.print("→ Assessing draft...")
    assessment = assess_appeal(case, appeal, selected)
    total_cost = retr_cost + draft_cost + assessment.cost_usd
    console.print(f"[dim]Total LLM cost: ${total_cost:.4f}[/dim]\n")
    render_terminal(case, appeal, assessment, console)
    if report is not None:
        write_markdown(case, appeal, assessment, report)
        console.print(f"\n[green]Wrote markdown report → {report}[/green]")


@app.command(name="list-cases")
def list_cases_cmd(
    cases_dir: Path = typer.Option(Path("data/cases"), help="Cases directory."),
) -> None:
    """List bundled cases."""
    for p in sorted(cases_dir.glob("*.yaml")):
        case = load_case(p)
        console.print(f"  [cyan]{p.stem}[/cyan]  {case.title}")


if __name__ == "__main__":
    app()
