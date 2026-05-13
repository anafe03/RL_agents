"""Earnings Call Inspector CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from earningscall.extractor import (
    extract_analyst_questions,
    extract_metrics,
    extract_surprises,
    extract_tone,
)
from earningscall.models import EarningsReport, load_transcript
from earningscall.verifier import verify_quotes

app = typer.Typer(help="Earnings Call Inspector — multi-pass structured extraction with citation enforcement.")
console = Console()


@app.command()
def inspect(
    transcript_path: Path = typer.Argument(..., help="Path to a transcript YAML."),
    report: Path = typer.Option(None, help="Optional path to write markdown report."),
) -> None:
    """Run all 4 extraction passes + citation verification on a transcript."""
    transcript = load_transcript(transcript_path)
    console.print(Panel.fit(f"Inspecting [bold]{transcript.company}[/bold] {transcript.period}", style="cyan"))

    console.print("→ Pass 1: metrics...")
    metrics, c1 = extract_metrics(transcript)
    console.print(f"  Extracted {len(metrics)} metrics")

    console.print("→ Pass 2: tone...")
    tone, c2 = extract_tone(transcript)
    console.print(f"  Extracted {len(tone)} tone assessments")

    console.print("→ Pass 3: surprises...")
    surprises, c3 = extract_surprises(transcript)
    console.print(f"  Flagged {len(surprises)} surprises")

    console.print("→ Pass 4: analyst Q&A...")
    questions, c4 = extract_analyst_questions(transcript)
    console.print(f"  Scored {len(questions)} analyst questions")

    console.print("→ Verifying citations against transcript...")
    citation_results = verify_quotes(transcript, metrics, tone, surprises, questions)
    verified = sum(1 for r in citation_results if r.found)
    total = len(citation_results)
    style = "green" if verified == total else "yellow"
    console.print(f"  [{style}]{verified}/{total} quotes verified[/{style}]")

    er = EarningsReport(
        transcript_id=transcript.id,
        company=transcript.company,
        period=transcript.period,
        metrics=metrics,
        tone=tone,
        surprises=surprises,
        analyst_questions=questions,
        citation_results=citation_results,
        cost_usd=c1 + c2 + c3 + c4,
    )
    _render_terminal(er, console)

    if report is not None:
        report.write_text(_render_markdown(er))
        console.print(f"\n[green]Wrote markdown report → {report}[/green]")


@app.command(name="list-transcripts")
def list_transcripts_cmd(
    transcripts_dir: Path = typer.Option(Path("data/transcripts"), help="Transcripts directory."),
) -> None:
    """List bundled transcripts."""
    for p in sorted(transcripts_dir.glob("*.yaml")):
        t = load_transcript(p)
        console.print(f"  [cyan]{p.stem}[/cyan]  {t.company} {t.period}")


# ---- rendering ------------------------------------------------------------


def _render_terminal(er: EarningsReport, console: Console) -> None:
    console.print()
    table = Table(title=f"{er.company} {er.period} — Metrics", show_header=True, header_style="bold")
    table.add_column("Metric"); table.add_column("Value"); table.add_column("vs Expectations")
    for m in er.metrics:
        table.add_row(m.name, m.value, m.vs_expectations or "—")
    console.print(table)

    console.print(f"\n[bold]Tone[/bold]")
    for t in er.tone:
        console.print(f"  [cyan]{t.speaker_name}[/cyan] — {t.segment}: [bold]{t.sentiment}[/bold]")
        console.print(f"    [dim]{t.note}[/dim]")

    if er.surprises:
        console.print(f"\n[bold]Surprises[/bold]")
        for s in er.surprises:
            console.print(f"  • [{_sig_color(s.significance)}]{s.significance.upper():6}[/] {s.headline}")
            console.print(f"    [dim]{s.rationale}[/dim]")

    if er.analyst_questions:
        console.print(f"\n[bold]Analyst Q&A (ranked)[/bold]")
        for q in sorted(er.analyst_questions, key=lambda x: -x.sharpness):
            console.print(f"  [{q.sharpness}/5] [cyan]{q.analyst_name}[/cyan] ({q.affiliation}) — {q.answer_quality}")
            console.print(f"    {q.question_summary}")
            console.print(f"    [dim]{q.rationale}[/dim]")

    console.print(f"\n[dim]Citations verified: {sum(1 for r in er.citation_results if r.found)}/{len(er.citation_results)} · cost ${er.cost_usd:.4f}[/dim]")


def _sig_color(sig: str) -> str:
    return {"high": "red", "medium": "yellow", "low": "dim"}.get(sig, "white")


def _render_markdown(er: EarningsReport) -> str:
    lines = [
        f"# {er.company} {er.period} — Earnings Call Inspection",
        "",
        f"- **Citations verified:** {sum(1 for r in er.citation_results if r.found)}/{len(er.citation_results)}",
        f"- **LLM cost:** ${er.cost_usd:.4f}",
        "",
        "## Metrics",
        "",
        "| Metric | Value | vs Expectations | Quote |",
        "|---|---|---|---|",
    ]
    for m in er.metrics:
        lines.append(f"| {m.name} | {m.value} | {m.vs_expectations or '—'} | \"{m.quote.text[:120]}\" |")
    lines.append("")
    lines.append("## Tone")
    for t in er.tone:
        lines.append(f"### {t.speaker_name} — {t.segment}")
        lines.append(f"**{t.sentiment}.** {t.note}")
        for q in t.evidence:
            lines.append(f"> \"{q.text}\"")
        lines.append("")
    if er.surprises:
        lines.append("## Surprises")
        for s in er.surprises:
            lines.append(f"### {s.headline}  *({s.significance.upper()})*")
            lines.append(f"_{s.rationale}_")
            for q in s.evidence:
                lines.append(f"> \"{q.text}\"")
            lines.append("")
    if er.analyst_questions:
        lines.append("## Analyst Q&A (ranked by sharpness)")
        for q in sorted(er.analyst_questions, key=lambda x: -x.sharpness):
            lines.append(f"### [{q.sharpness}/5] {q.analyst_name} ({q.affiliation}) — {q.answer_quality}")
            lines.append(q.question_summary)
            lines.append(f"> \"{q.quote.text}\"")
            lines.append(f"_{q.rationale}_")
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    app()


@app.command()
def healthcheck(
    model: str = "gpt-4o-mini",
) -> None:
    """One LLM call against the real API to verify your key + provider work.

    Defaults to gpt-4o-mini (cheap). Pass --model claude-haiku-4-5 etc to test other providers.
    """
    from earningscall import llm
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
    console.print(f"[green]OK[/green] model={getattr(result, 'model', model)} cost=${result.cost_usd:.6f}")
