"""Transcript rendering — terminal (rich) + markdown."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from simulacrum.models import Scenario, Transcript


def render_terminal(scenario: Scenario, transcript: Transcript, console: Console | None = None) -> None:
    console = console or Console()
    console.print(
        Panel.fit(
            f"Simulacrum  ─  [bold]{scenario.title or scenario.name}[/bold]",
            style="bold cyan",
        )
    )
    names = {a.id: a.name for a in scenario.agents}
    for tick in transcript.ticks:
        console.print(f"\n[bold]── Tick {tick.number} ──[/bold]")
        for action in tick.actions:
            name = names.get(action.actor_id, action.actor_id)
            if action.type.value == "pass":
                console.print(f"  [dim]{name} passes.[/dim]")
            else:
                console.print(f"  [cyan]{name}:[/cyan] {action.content}")
    if transcript.cost_usd:
        console.print(f"\n[dim]Total LLM cost: ${transcript.cost_usd:.4f}[/dim]")


def render_markdown(scenario: Scenario, transcript: Transcript) -> str:
    names = {a.id: a.name for a in scenario.agents}
    lines: list[str] = []
    lines.append(f"# {scenario.title or scenario.name}")
    lines.append("")
    if scenario.setting:
        lines.append(f"> **Setting.** {scenario.setting.strip()}")
        lines.append("")
    if scenario.shared_goal:
        lines.append(f"> **Goal.** {scenario.shared_goal.strip()}")
        lines.append("")
    lines.append("## Cast")
    lines.append("")
    for agent in scenario.agents:
        role = f" — *{agent.role}*" if agent.role else ""
        lines.append(f"- **{agent.name}**{role}")
    lines.append("")
    lines.append("## Transcript")
    lines.append("")
    for tick in transcript.ticks:
        lines.append(f"### Tick {tick.number}")
        lines.append("")
        for action in tick.actions:
            name = names.get(action.actor_id, action.actor_id)
            if action.type.value == "pass":
                lines.append(f"_{name} passes._")
            else:
                lines.append(f"**{name}:** {action.content}")
            lines.append("")
    if transcript.cost_usd:
        lines.append(f"---\n\n*LLM cost: ${transcript.cost_usd:.4f}*")
    return "\n".join(lines)


def write_markdown(scenario: Scenario, transcript: Transcript, path: Path | str) -> None:
    Path(path).write_text(render_markdown(scenario, transcript))
