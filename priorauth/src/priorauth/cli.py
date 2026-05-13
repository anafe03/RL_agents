"""PriorAuth Assist CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from priorauth.assessor import assess_appeal
from priorauth.benchmark import load_golden, run_benchmark
from priorauth.drafter import draft_appeal
from priorauth.models import load_case, load_guideline_corpus
from priorauth.render import render_terminal, write_markdown
from priorauth.retrievers import REGISTRY, get_retriever

app = typer.Typer(help="PriorAuth Assist — cited prior-authorization appeal drafter.")
console = Console()


@app.command()
def appeal(
    case_path: Path = typer.Argument(..., help="Path to a case YAML."),
    guidelines_dir: Path = typer.Option(Path("data/guidelines"), help="Guideline corpus directory."),
    retriever: str = typer.Option(
        "llm_judged",
        help=f"Which retriever to use. Available: {', '.join(sorted(REGISTRY))}",
    ),
    report: Path = typer.Option(None, help="Optional path to write the markdown appeal."),
) -> None:
    """Run the full retriever → drafter → assessor pipeline on a case."""
    case = load_case(case_path)
    corpus = load_guideline_corpus(guidelines_dir)
    console.print(f"Loaded case [bold]{case.id}[/bold] and {len(corpus)} guidelines.")
    console.print(f"Using retriever: [cyan]{retriever}[/cyan]\n")

    console.print("→ Retrieving relevant guidelines...")
    retr = get_retriever(retriever)
    retr.index(corpus)
    selected = retr.retrieve(case, k=5)
    retr_cost = retr.cost_usd
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


@app.command()
def benchmark(
    cases_dir: Path = typer.Option(Path("data/cases"), help="Cases directory."),
    guidelines_dir: Path = typer.Option(Path("data/guidelines"), help="Guideline corpus."),
    golden_path: Path = typer.Option(Path("data/golden.yaml"), help="Golden expected-IDs YAML."),
    retrievers: list[str] = typer.Option(
        ["bm25", "chroma_minilm"],
        help="Which retrievers to benchmark (repeat the flag). Default is local-only; add 'llm_judged' to include the API-backed retriever (requires ANTHROPIC_API_KEY).",
    ),
    k: int = typer.Option(5, help="Retrieve top-k per query."),
) -> None:
    """Benchmark every selected retriever × every case. Prints precision@k, recall@k, latency, cost."""
    corpus = load_guideline_corpus(guidelines_dir)
    cases = [load_case(p) for p in sorted(cases_dir.glob("*.yaml"))]
    golden = load_golden(golden_path)
    rs = [get_retriever(name) for name in retrievers]
    console.print(
        f"Benchmarking {len(rs)} retrievers × {len(cases)} cases × {len(corpus)} guidelines (k={k})..."
    )
    report = run_benchmark(rs, cases, corpus, golden, k=k)

    table = Table(title=f"Retrieval benchmark (k={k})", show_header=True, header_style="bold")
    table.add_column("Retriever", style="cyan")
    table.add_column("Precision@k", justify="right")
    table.add_column("Recall@k", justify="right")
    table.add_column("Latency (ms, avg)", justify="right")
    table.add_column("Total cost (USD)", justify="right")
    for name, agg in report.by_retriever().items():
        table.add_row(
            name,
            f"{agg['precision_at_k']:.2f}",
            f"{agg['recall_at_k']:.2f}",
            f"{agg['latency_ms']:.1f}",
            f"${agg['cost_usd']:.4f}",
        )
    console.print(table)

    # Per-case detail
    console.print("\n[bold]Per-case detail[/bold]")
    for r in report.results:
        console.print(
            f"  [cyan]{r.retriever_name:<15}[/cyan] {r.case_id:<22} "
            f"P={r.precision_at_k:.2f}  R={r.recall_at_k:.2f}  "
            f"lat={r.latency_ms:6.1f}ms  retrieved={r.retrieved_ids}"
        )


@app.command(name="list-cases")
def list_cases_cmd(
    cases_dir: Path = typer.Option(Path("data/cases"), help="Cases directory."),
) -> None:
    """List bundled cases."""
    for p in sorted(cases_dir.glob("*.yaml")):
        case = load_case(p)
        console.print(f"  [cyan]{p.stem}[/cyan]  {case.title}")


@app.command(name="list-retrievers")
def list_retrievers_cmd() -> None:
    """List available retrieval backends."""
    for name in sorted(REGISTRY):
        r = get_retriever(name)
        console.print(f"  [cyan]{name}[/cyan]  category={r.category}")


if __name__ == "__main__":
    app()
