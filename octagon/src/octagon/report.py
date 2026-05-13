"""Audit report rendering — terminal (rich) and markdown."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from octagon.models import AuditReport, Outcome


def _bar(blocked: int, total: int, width: int = 10) -> str:
    if total == 0:
        return "·" * width
    filled = round((blocked / total) * width)
    return "█" * filled + "░" * (width - filled)


def render_terminal(report: AuditReport, console: Console | None = None) -> None:
    console = console or Console()

    console.print(
        Panel.fit(
            f"Red Cell  ─  Adversarial Audit of [bold]{report.defender_name}[/bold]",
            style="bold red",
        )
    )

    table = Table(
        title=(
            f"Results by category — {report.total_blocked}/{report.total_attacks} blocked "
            f"({report.block_rate:.1%})"
        ),
        title_style="bold",
    )
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Block bar", no_wrap=True)
    table.add_column("Blocked", justify="right", style="green")
    table.add_column("Succeeded", justify="right", style="red")
    table.add_column("Total", justify="right")

    for cat, stats in sorted(report.by_category.items()):
        table.add_row(
            cat,
            _bar(stats["blocked"], stats["total"]),
            str(stats["blocked"]),
            str(stats["succeeded"]),
            str(stats["total"]),
        )
    console.print(table)

    succeeded = [r for r in report.results if r.outcome == Outcome.SUCCEEDED]
    if succeeded:
        console.print("\n[bold red]Top successful attacks (defender violations):[/bold red]")
        for r in succeeded[:5]:
            tools = ", ".join(r.tools_called) if r.tools_called else "no tool call"
            console.print(f"  • [cyan]{r.attack_id}[/cyan] — {tools}")
            console.print(f"    {r.judge_reasoning}", style="dim")


def render_markdown(report: AuditReport) -> str:
    lines: list[str] = []
    lines.append(f"# Red Cell — Adversarial Audit of `{report.defender_name}`")
    lines.append("")
    lines.append(f"- **Total attacks:** {report.total_attacks}")
    lines.append(f"- **Blocked:** {report.total_blocked}")
    lines.append(f"- **Succeeded:** {report.total_succeeded}")
    lines.append(f"- **Block rate:** {report.block_rate:.1%}")
    if report.ended_at:
        lines.append(f"- **Completed at:** {report.ended_at.isoformat()}")
    lines.append("")

    lines.append("## Results by category")
    lines.append("")
    lines.append("| Category | Blocked | Succeeded | Total |")
    lines.append("|---|---:|---:|---:|")
    for cat, stats in sorted(report.by_category.items()):
        lines.append(
            f"| `{cat}` | {stats['blocked']} | {stats['succeeded']} | {stats['total']} |"
        )
    lines.append("")

    succeeded = [r for r in report.results if r.outcome == Outcome.SUCCEEDED]
    if succeeded:
        lines.append("## Successful attacks (defender violations)")
        lines.append("")
        for r in succeeded:
            lines.append(f"### `{r.attack_id}`  *(category: {r.attack_category.value})*")
            lines.append("")
            lines.append(f"**Judge:** {r.judge_reasoning}")
            lines.append("")
            if r.tools_called:
                lines.append(f"**Tools called:** {', '.join(r.tools_called)}")
                lines.append("")

    return "\n".join(lines)


def write_markdown(report: AuditReport, path: Path | str) -> None:
    Path(path).write_text(render_markdown(report))
