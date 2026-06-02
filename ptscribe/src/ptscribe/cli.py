"""ptscribe CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ptscribe import llm
from ptscribe.eval import run_eval
from ptscribe.mock import make_mock_chat
from ptscribe.models import RunRecord
from ptscribe.monitoring import aggregate_stats, log_from_eval
from ptscribe.scribe import extract_soap

app = typer.Typer(help="ptscribe — ambient SOAP-note generation for PT/OT/ST.")
console = Console()


@app.command()
def scribe(
    transcript_path: Path = typer.Argument(..., help="Path to a transcript text file."),
    live: bool = typer.Option(False, help="Use a real LLM (needs API key)."),
    model: str = typer.Option("claude-sonnet-4-6", help="LLM model id."),
    judge: bool = typer.Option(False, help="Run LLM-as-judge on the narrative."),
    log: bool = typer.Option(True, help="Log this run to the monitoring DB."),
) -> None:
    """Run the scribe pipeline on a transcript and print the structured SOAP note."""
    if not transcript_path.exists():
        console.print(f"[red]Not found:[/red] {transcript_path}")
        raise typer.Exit(code=1)
    transcript = transcript_path.read_text()
    transcript_id = transcript_path.stem

    if live:
        llm.reset_chat_fn()
    else:
        llm.set_chat_fn(make_mock_chat())

    try:
        note, chat_result = extract_soap(transcript, model=model)
        eval_result, judge_cost = run_eval(
            transcript, note, transcript_id=transcript_id,
            use_judge=judge and live, judge_model=model,
        )
    finally:
        llm.reset_chat_fn()

    console.print(f"[bold]{note.patient_label}[/bold]")
    console.print(f"[dim]{note.visit_type} · {note.discipline}[/dim]\n")
    console.print(json.dumps(note.model_dump(), indent=2, default=str))

    console.print("\n[bold]Eval[/bold]")
    table = Table(show_header=False)
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("Overall", eval_result.overall.value)
    table.add_row("Completeness", f"{eval_result.completeness_score:.2f}")
    table.add_row("All sections present", "yes" if eval_result.has_all_sections else "no")
    table.add_row("Hallucination findings", str(len(eval_result.hallucination_findings)))
    if eval_result.judge_score is not None:
        table.add_row("Judge score", f"{eval_result.judge_score:.2f}")
    console.print(table)

    for finding in eval_result.hallucination_findings:
        console.print(f"  [yellow]⚠[/yellow] {finding.field_path}: {finding.claim} — {finding.note}")

    if log:
        log_from_eval(
            transcript_id=transcript_id,
            model=chat_result.model,
            mode="live" if live else "demo",
            cost_usd=chat_result.cost_usd + judge_cost,
            latency_ms=chat_result.latency_ms,
            input_chars=len(transcript),
            output_chars=len(chat_result.text),
            eval_result=eval_result,
        )
        console.print("\n[dim]Run logged.[/dim]")


@app.command()
def stats() -> None:
    """Show roll-up stats from the monitoring DB."""
    s = aggregate_stats()
    if s.get("total_runs", 0) == 0:
        console.print("No runs logged yet.")
        raise typer.Exit()
    table = Table(title="Run stats")
    table.add_column(style="cyan")
    table.add_column(justify="right")
    for k, v in s.items():
        table.add_row(k, str(v))
    console.print(table)


@app.command()
def eval_all(
    transcripts_dir: Path = typer.Option(Path("data/transcripts"), help="Folder of transcript files."),
    live: bool = typer.Option(False, help="Use a real LLM."),
    judge: bool = typer.Option(False, help="Run LLM-as-judge."),
) -> None:
    """Run the full eval harness across every bundled transcript."""
    if not transcripts_dir.exists():
        console.print(f"[red]No transcripts folder:[/red] {transcripts_dir}")
        raise typer.Exit(code=1)

    if live:
        llm.reset_chat_fn()
    else:
        llm.set_chat_fn(make_mock_chat())

    table = Table(title="Eval harness")
    table.add_column("Transcript", style="cyan")
    table.add_column("Overall")
    table.add_column("Completeness", justify="right")
    table.add_column("Hallucinations", justify="right")
    table.add_column("Judge", justify="right")
    table.add_column("Cost", justify="right")

    try:
        for path in sorted(transcripts_dir.glob("*.txt")):
            transcript = path.read_text()
            note, chat_result = extract_soap(transcript)
            eval_result, judge_cost = run_eval(
                transcript, note, transcript_id=path.stem,
                use_judge=judge and live,
            )
            color = {
                "pass": "green", "warn": "yellow", "fail": "red",
            }.get(eval_result.overall.value, "white")
            judge_text = ("—" if eval_result.judge_score is None
                          else f"{eval_result.judge_score:.2f}")
            table.add_row(
                path.stem,
                f"[{color}]{eval_result.overall.value}[/{color}]",
                f"{eval_result.completeness_score:.2f}",
                str(len(eval_result.hallucination_findings)),
                judge_text,
                f"${chat_result.cost_usd + judge_cost:.5f}",
            )
    finally:
        llm.reset_chat_fn()

    console.print(table)


if __name__ == "__main__":
    app()
