"""Series transforms + Plotly figure construction.

The LLM decides *what* to chart (`ChartSpec`); this module executes it
deterministically — applies the transform and builds the figure.
"""

from __future__ import annotations

from chartlab.models import ChartSpec, Series, Transform

_Y_AXIS_LABEL: dict[Transform, str] = {
    Transform.RAW: "Price (USD)",
    Transform.INDEXED: "Indexed to 100 at start",
    Transform.PCT_CHANGE: "% change from start",
}


def apply_transform(series: Series, transform: Transform) -> list[float]:
    """Apply a presentation transform to one series' close prices."""
    closes = series.closes
    if not closes:
        return []
    if transform == Transform.RAW:
        return list(closes)
    base = closes[0]
    if base == 0:
        return list(closes)
    if transform == Transform.INDEXED:
        return [round(c / base * 100, 3) for c in closes]
    if transform == Transform.PCT_CHANGE:
        return [round((c / base - 1) * 100, 3) for c in closes]
    return list(closes)


def summarize(series: Series, transform: Transform) -> dict[str, float]:
    """Headline numbers for one series — start, end, and total % change."""
    closes = series.closes
    if not closes:
        return {"start": 0.0, "end": 0.0, "change_pct": 0.0}
    start, end = closes[0], closes[-1]
    change = (end / start - 1) * 100 if start else 0.0
    return {"start": round(start, 2), "end": round(end, 2), "change_pct": round(change, 2)}


def build_figure(spec: ChartSpec, series_list: list[Series]):
    """Build a Plotly line figure for the spec + fetched series."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for series in series_list:
        y = apply_transform(series, spec.transform)
        fig.add_trace(
            go.Scatter(x=series.dates, y=y, mode="lines", name=series.ticker)
        )
    title = spec.title or " vs ".join(s.ticker for s in series_list) or "Chart"
    fig.update_layout(
        title=title,
        yaxis_title=_Y_AXIS_LABEL.get(spec.transform, "Value"),
        xaxis_title="Date",
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin={"l": 12, "r": 12, "t": 48, "b": 12},
        legend={"orientation": "h", "y": -0.2},
        height=460,
    )
    if spec.transform == Transform.PCT_CHANGE:
        fig.add_hline(y=0, line_dash="dot", line_color="#5d6b7a")
    return fig
