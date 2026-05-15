"""stemdeck CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from stemdeck.analyzer import analyze
from stemdeck.compat import rank_next, score_pair
from stemdeck.mock import demo_catalog
from stemdeck.models import Catalog
from stemdeck.parser import parse_als

app = typer.Typer(help="stemdeck — live element-level mashup tool for an Ableton catalog.")
console = Console()


def _catalog_table(catalog: Catalog) -> Table:
    table = Table(title="Catalog")
    table.add_column("ID", style="cyan")
    table.add_column("Title")
    table.add_column("BPM", justify="right")
    table.add_column("Key")
    table.add_column("Camelot", justify="center")
    table.add_column("Energy", justify="right")
    table.add_column("Tracks", justify="right")
    for s in catalog.songs:
        table.add_row(
            s.id, s.title, f"{s.bpm:g}", s.key or "—", s.camelot or "—",
            str(s.energy), str(len(s.tracks)),
        )
    return table


@app.command()
def demo() -> None:
    """Show the bundled demo catalog and its compatibility matrix."""
    catalog = demo_catalog()
    console.print(_catalog_table(catalog))

    matrix = Table(title="\nCompatibility matrix — rows transition INTO columns")
    matrix.add_column("from \\ to", style="cyan")
    for s in catalog.songs:
        matrix.add_column(s.id, justify="center")
    for a in catalog.songs:
        cells = []
        for b in catalog.songs:
            if a.id == b.id:
                cells.append("[dim]·[/dim]")
            else:
                p = score_pair(a, b)
                color = "green" if p.stars >= 4 else "yellow" if p.stars >= 3 else "red"
                cells.append(f"[{color}]{'★' * p.stars}[/{color}]")
        matrix.add_row(a.id, *cells)
    console.print(matrix)


@app.command()
def next(song_id: str = typer.Argument(..., help="Song id to find transitions out of.")) -> None:
    """Rank every other demo-catalog song as a candidate to follow SONG_ID."""
    catalog = demo_catalog()
    current = catalog.get(song_id)
    if current is None:
        console.print(f"[red]Unknown song:[/red] {song_id}. Known: {', '.join(catalog.song_ids)}")
        raise typer.Exit(code=1)
    console.print(f"Next options after [bold]{current.title}[/bold] "
                  f"({current.camelot} · {current.bpm:g} BPM):\n")
    for p in rank_next(current, catalog.songs):
        target = catalog.get(p.song_b)
        stars = "★" * p.stars + "☆" * (5 - p.stars)
        console.print(f"  [cyan]{stars}[/cyan]  {target.title:10s} "
                      f"{target.camelot:>4s} · {target.bpm:g} BPM — {p.note}")


@app.command()
def analyze_dir(
    directory: Path = typer.Argument(..., help="Folder of .als project files."),
) -> None:
    """Parse + analyze a folder of real .als files and print the catalog."""
    als_files = sorted(directory.glob("*.als"))
    if not als_files:
        console.print(f"[red]No .als files under[/red] {directory}")
        raise typer.Exit(code=1)
    songs = []
    for path in als_files:
        try:
            songs.append(analyze(parse_als(path)))
            console.print(f"[green]parsed[/green] {path.name}")
        except Exception as e:  # noqa: BLE001 - report and continue per-file
            console.print(f"[red]failed[/red] {path.name}: {type(e).__name__}: {e}")
    console.print(_catalog_table(Catalog(songs=songs)))


if __name__ == "__main__":
    app()
