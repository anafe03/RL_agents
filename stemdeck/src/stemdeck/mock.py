"""Bundled demo catalog — the hosted demo's data source.

The demo catalog is a set of pre-analyzed synthetic songs under
`data/catalog/`. It plays the role AutoFill's recorded playback does:
exercise the full UI with no external dependency (here, no Ableton).
"""

from __future__ import annotations

from pathlib import Path

from stemdeck.models import Catalog, load_catalog

_CATALOG_DIR = Path(__file__).resolve().parents[2] / "data" / "catalog"


def demo_catalog() -> Catalog:
    """Load the bundled synthetic catalog."""
    return load_catalog(_CATALOG_DIR)
