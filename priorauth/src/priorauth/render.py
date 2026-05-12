"""Render an appeal + assessment to terminal and markdown."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from priorauth.models import Appeal, AppealAssessment, Case, RubricVerdict


_VERDICT_STYLE = {
    RubricVerdict.EXCELLENT: ("green", "EXCELLENT"),
    RubricVerdict.STRONG: ("green", "STRONG"),
    RubricVerdict.MODERATE: ("yellow", "MODERATE"),
    RubricVerdict.WEAK: ("red", "WEAK"),
}


def render_terminal(
    case: Case,
    appeal: Appeal,
    assessment: AppealAssessment,
    console: Console | None = None,
) -> None:
    console = console or Console()
    style, label = _VERDICT_STYLE.get(assessment.verdict, ("white", "UNKNOWN"))
    console.print(
        Panel.fit(
            f"PriorAuth Assist  ─  [bold]{case.title}[/bold]",
            style="bold cyan",
        )
    )
    console.print(f"[bold {style}]Assessment: {label}[/bold {style}]")
    console.print(f"  Addressed all denial criteria : {'✓' if assessment.addressed_all_denial_criteria else '✗'}")
    console.print(f"  All clinical claims cited     : {'✓' if assessment.all_claims_cited else '✗'}")
    console.print(f"  Patient facts accurate        : {'✓' if assessment.patient_facts_accurate else '✗'}")
    console.print(f"  Has clear ask                 : {'✓' if assessment.has_clear_ask else '✗'}")
    console.print(f"  [dim]{assessment.reasoning}[/dim]")
    if assessment.weak_points:
        console.print("\n[bold yellow]Weak points to review:[/bold yellow]")
        for wp in assessment.weak_points:
            console.print(f"  • {wp}")
    console.print()

    console.print(Panel(appeal.opening, title="Opening", border_style="cyan"))
    console.print("\n[bold]Clinical rationale[/bold]")
    for i, p in enumerate(appeal.clinical_rationale, 1):
        console.print(f"  {i}. {p}\n")
    console.print("[bold]Citations[/bold]")
    for c in appeal.citations:
        console.print(f"  • [cyan]{c.guideline_id}[/cyan] — {c.claim}")
        console.print(f"    [dim]\"{c.quoted_excerpt[:200]}\"[/dim]")
    console.print()
    console.print(Panel(appeal.closing, title="Closing", border_style="cyan"))


def render_markdown(case: Case, appeal: Appeal, assessment: AppealAssessment) -> str:
    style, label = _VERDICT_STYLE.get(assessment.verdict, ("", "UNKNOWN"))
    lines: list[str] = []
    lines.append(f"# Prior-Authorization Appeal — {case.title}")
    lines.append("")
    lines.append(f"- **Payer:** {case.denial.payer}")
    lines.append(f"- **Member ID:** {case.denial.member_id}")
    lines.append(f"- **Requested service:** {case.requested_service}")
    lines.append("")
    lines.append(f"## Assessment: **{label}**")
    lines.append("")
    lines.append(f"- Addressed all denial criteria: {'✓' if assessment.addressed_all_denial_criteria else '✗'}")
    lines.append(f"- All clinical claims cited: {'✓' if assessment.all_claims_cited else '✗'}")
    lines.append(f"- Patient facts accurate: {'✓' if assessment.patient_facts_accurate else '✗'}")
    lines.append(f"- Has clear ask: {'✓' if assessment.has_clear_ask else '✗'}")
    lines.append("")
    lines.append(f"> {assessment.reasoning}")
    lines.append("")
    if assessment.weak_points:
        lines.append("### Weak points to review")
        lines.append("")
        for wp in assessment.weak_points:
            lines.append(f"- {wp}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Letter")
    lines.append("")
    lines.append(appeal.opening)
    lines.append("")
    lines.append("**Clinical rationale**")
    lines.append("")
    for i, p in enumerate(appeal.clinical_rationale, 1):
        lines.append(f"{i}. {p}")
        lines.append("")
    lines.append("**Citations**")
    lines.append("")
    for c in appeal.citations:
        lines.append(f"- `{c.guideline_id}` — {c.claim}")
        lines.append(f"  > \"{c.quoted_excerpt}\"")
        lines.append("")
    lines.append(appeal.closing)
    lines.append("")
    return "\n".join(lines)


def write_markdown(case: Case, appeal: Appeal, assessment: AppealAssessment, path: Path | str) -> None:
    Path(path).write_text(render_markdown(case, appeal, assessment))
