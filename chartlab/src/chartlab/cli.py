"""chartlab CLI."""

from __future__ import annotations

import os

import typer
from rich.console import Console
from rich.table import Table

from chartlab import llm
from chartlab.chart import summarize
from chartlab.mock import demo_series, make_mock_chat
from chartlab.models import Transform
from chartlab.nlp import parse_request

app = typer.Typer(help="chartlab — natural-language financial charting agent.")
console = Console()


@app.command()
def chart(
    request: str = typer.Argument(..., help="Plain-English chart request."),
    live: bool = typer.Option(
        False, help="Use a real LLM + live Yahoo Finance data (needs an API key)."
    ),
    model: str = typer.Option("claude-sonnet-4-6", help="LLM model id."),
) -> None:
    """Parse a natural-language request into a chart spec and summarize it."""
    if live:
        llm.reset_chat_fn()
    else:
        llm.set_chat_fn(make_mock_chat())

    try:
        spec, cost = parse_request(request, model=model)
    finally:
        llm.reset_chat_fn()

    console.print(f"Request: [italic]{request}[/italic]")
    console.print(
        f"Spec:    tickers=[bold]{', '.join(spec.tickers)}[/bold] · "
        f"period=[bold]{spec.period}[/bold] · transform=[bold]{spec.transform.value}[/bold]"
    )
    if cost:
        console.print(f"LLM cost: ${cost:.5f}")

    table = Table(title=spec.title or "Chart")
    table.add_column("Ticker", style="cyan")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    table.add_column("Change", justify="right")

    for ticker in spec.tickers:
        if live:
            from chartlab.data import fetch_series
            try:
                series = fetch_series(ticker, spec.period)
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]{ticker}: fetch failed — {e}[/red]")
                continue
        else:
            series = demo_series(ticker, spec.period)
        s = summarize(series, spec.transform)
        arrow = "[green]▲[/green]" if s["change_pct"] >= 0 else "[red]▼[/red]"
        table.add_row(
            ticker, f"{s['start']:g}", f"{s['end']:g}",
            f"{arrow} {s['change_pct']:+.2f}%",
        )
    console.print(table)
    if not live:
        console.print("[dim]Demo mode — synthetic prices. Use --live for real data.[/dim]")


@app.command()
def healthcheck(model: str = typer.Option("claude-sonnet-4-6", help="LLM model id.")) -> None:
    """Verify the configured LLM provider answers."""
    key_var = "OPENAI_API_KEY" if llm._is_openai_model(model) else "ANTHROPIC_API_KEY"
    if not os.environ.get(key_var):
        console.print(f"[red]{key_var} is not set.[/red]")
        raise typer.Exit(code=1)
    try:
        result = llm.chat(
            model=model,
            system="Reply with exactly one word: pong.",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=16,
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Failed:[/red] {type(e).__name__}: {e}")
        raise typer.Exit(code=1)
    console.print(f"[green]OK[/green] model={getattr(result, 'model', model)} "
                  f"cost=${result.cost_usd:.6f}")


if __name__ == "__main__":
    app()
