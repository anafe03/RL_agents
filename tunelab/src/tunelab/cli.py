"""tunelab CLI."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tunelab.dataset import load_eval, load_train
from tunelab.eval import mock_baseline_model, run_eval

app = typer.Typer(help="tunelab — post-training experiments with verifiable rewards.")
console = Console()


@app.command(name="inspect-dataset")
def inspect_dataset() -> None:
    """Show summary of train/eval datasets."""
    train = load_train()
    eval_ = load_eval()
    console.print(f"Train examples: [bold]{len(train)}[/bold]")
    console.print(f"Eval examples:  [bold]{len(eval_)}[/bold]")
    console.print("\n[bold]Sample (first eval example):[/bold]")
    if eval_:
        ex = eval_[0]
        console.print(f"Input:    {ex.input[:200]}")
        console.print(f"Expected: {json.dumps(ex.expected, indent=2)[:400]}...")


@app.command(name="eval-mock")
def eval_mock() -> None:
    """Run the eval harness against the mock_baseline_model.

    Tests the harness end-to-end without needing a real model. Useful for
    smoke-testing the reward function + report formatting.
    """
    report = run_eval(mock_baseline_model)
    console.print(f"\n[bold]Mock baseline eval:[/bold]")
    console.print(f"  Mean reward       : {report.mean_reward:.3f}")
    console.print(f"  Schema pass rate  : {report.schema_pass_rate:.1%}")
    console.print(f"  Mean field-match  : {report.mean_field_match:.3f}\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim")
    table.add_column("Schema", justify="right")
    table.add_column("Field", justify="right")
    table.add_column("Reward", justify="right")
    table.add_column("Input (truncated)")
    for i, r in enumerate(report.results):
        table.add_row(
            str(i),
            f"{r.schema_ok:.0f}",
            f"{r.field_match:.2f}",
            f"{r.reward:.2f}",
            r.input[:60] + ("..." if len(r.input) > 60 else ""),
        )
    console.print(table)


@app.command()
def train(
    base_model: str = typer.Option("Qwen/Qwen2.5-1.5B", help="HF model id"),
    output_dir: Path = typer.Option(Path("runs/v1"), help="Where to save LoRA adapter."),
    epochs: int = typer.Option(3, help="Training epochs."),
) -> None:
    """Run GRPO training (requires --extra train; GPU expected)."""
    try:
        from tunelab.train import main as run_training
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    run_training(base_model=base_model, output_dir=str(output_dir), num_train_epochs=epochs)


if __name__ == "__main__":
    app()
