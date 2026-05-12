"""Schedule rendering — terminal (rich) + markdown."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from festival.models import Schedule


def render_terminal(schedule: Schedule, console: Console | None = None) -> None:
    console = console or Console()
    console.print(
        Panel.fit(
            f"Festival Companion  ─  [bold]{schedule.festival_name}[/bold]",
            style="bold magenta",
        )
    )
    console.print(
        f"[dim]{schedule.total_picks} picks across {len(schedule.days)} days · "
        f"avg fit {schedule.average_score:.2f} · "
        f"LLM cost ${schedule.cost_usd:.4f}[/dim]\n"
    )

    for day in schedule.days:
        console.print(f"[bold]── {day.day} ──[/bold]")
        if not day.picks:
            console.print("[dim]  (no picks)[/dim]\n")
            continue
        table = Table(show_header=True, header_style="bold")
        table.add_column("Time", style="cyan", no_wrap=True)
        table.add_column("Artist", style="bold")
        table.add_column("Stage")
        table.add_column("Fit", justify="right")
        table.add_column("Why")
        for p in day.picks:
            time = f"{p.start}–{p.end}"
            fit = f"{p.score:.2f}" + (" ★" if p.must_see else "")
            table.add_row(time, p.artist, p.stage, fit, p.reasoning)
        console.print(table)

        if day.skipped_due_to_conflict:
            console.print("[dim]  Skipped due to conflicts (still strong picks):[/dim]")
            for s in day.skipped_due_to_conflict[:5]:
                console.print(
                    f"  [yellow]• {s.start}–{s.end}  {s.artist} ({s.stage})  fit {s.score:.2f}[/yellow]"
                )
                console.print(f"    [dim]{s.reasoning}[/dim]")
        console.print()


def render_markdown(schedule: Schedule) -> str:
    lines: list[str] = []
    lines.append(f"# {schedule.festival_name} — Personalized schedule")
    lines.append("")
    lines.append(f"- **Total picks:** {schedule.total_picks} across {len(schedule.days)} days")
    lines.append(f"- **Average fit:** {schedule.average_score:.2f}")
    lines.append(f"- **LLM cost:** ${schedule.cost_usd:.4f}")
    lines.append("")
    if schedule.taste.description:
        lines.append("## Taste profile")
        lines.append("")
        lines.append(f"> {schedule.taste.description}")
        lines.append("")

    for day in schedule.days:
        lines.append(f"## {day.day}")
        lines.append("")
        if not day.picks:
            lines.append("_(no picks)_")
            lines.append("")
            continue
        lines.append("| Time | Artist | Stage | Fit | Why |")
        lines.append("|---|---|---|---:|---|")
        for p in day.picks:
            star = " ★" if p.must_see else ""
            lines.append(
                f"| {p.start}–{p.end} | **{p.artist}**{star} | {p.stage} | {p.score:.2f} | {p.reasoning} |"
            )
        lines.append("")
        if day.skipped_due_to_conflict:
            lines.append("**Skipped due to conflicts (still strong picks):**")
            lines.append("")
            for s in day.skipped_due_to_conflict[:5]:
                lines.append(
                    f"- `{s.start}–{s.end}` **{s.artist}** ({s.stage}) — fit {s.score:.2f} — {s.reasoning}"
                )
            lines.append("")

    return "\n".join(lines)


def write_markdown(schedule: Schedule, path: Path | str) -> None:
    Path(path).write_text(render_markdown(schedule))
