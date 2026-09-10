"""Plotly map figure builder for the job-listings map.

Turns the listing dicts from :mod:`src.app.listings_loader` into an interactive
Plotly figure rendered by ``gr.Plot`` in the app. Uses an OpenStreetMap base
layer so **no Mapbox access token is required**.

Plotly 6 renamed the Mapbox-based trace/layout (``Scattermapbox`` / ``mapbox``)
to the MapLibre-based ``Scattermap`` / ``map``. The project pins
``plotly==5.24.1`` (Mapbox names), but local/CI environments may have Plotly 6+.
:func:`build_map_figure` therefore tries the new API first and falls back to the
older one, so the same code renders on both.

Validates: interactive map view for the redesigned app.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Rough geographic centre of Germany — the corpus is German job listings.
GERMANY_CENTER = {"lat": 51.1657, "lon": 10.4515}
DEFAULT_ZOOM = 5.2
MARKER_COLOR = "#FF5F46"  # Databricks "Lava" accent.


def _hover_text(listing: Dict[str, Any]) -> str:
    """Build the HTML hover card for one listing marker."""
    title = listing.get("job_title") or "Untitled role"
    company = listing.get("company_name") or "Unknown company"
    location = listing.get("location_text") or ""
    industry = listing.get("industry") or ""
    seniority = listing.get("seniority_level") or ""
    parts = [f"<b>{title}</b>", company]
    if location:
        parts.append(location)
    tags = " · ".join(t for t in (seniority, industry) if t)
    if tags:
        parts.append(tags)
    return "<br>".join(parts)


def build_map_figure(listings: Optional[List[Dict[str, Any]]] = None):
    """Build a Plotly map figure from listing rows.

    Args:
        listings: Listing dicts with ``latitude``/``longitude`` (and the hover
            fields). ``None``/empty renders an empty map centred on Germany so
            the UI always shows a map rather than a blank panel.

    Returns:
        A ``plotly.graph_objects.Figure`` ready to hand to ``gr.Plot``.
    """
    import plotly.graph_objects as go

    listings = listings or []
    lats = [row["latitude"] for row in listings]
    lons = [row["longitude"] for row in listings]
    hover = [_hover_text(row) for row in listings]

    # Prefer the MapLibre trace (Plotly 6+); fall back to Mapbox (Plotly 5).
    use_map = hasattr(go, "Scattermap")
    trace_cls = go.Scattermap if use_map else go.Scattermapbox

    marker = {"size": 11, "color": MARKER_COLOR, "opacity": 0.85}
    # Cluster nearby markers so hundreds/thousands of listings stay legible;
    # zooming in progressively breaks clusters apart into individual points.
    cluster = {
        "enabled": True,
        "maxzoom": 11,
        "step": [10, 50, 200],
        "size": [16, 22, 30, 40],
        "color": ["#FF8A73", "#FF5F46", "#E8380D", "#B02800"],
    }
    trace = trace_cls(
        lat=lats,
        lon=lons,
        mode="markers",
        marker=marker,
        cluster=cluster,
        text=hover,
        hoverinfo="text",
        name="Listings",
    )

    fig = go.Figure(trace)

    base_layer = {
        "style": "open-street-map",
        "center": GERMANY_CENTER,
        "zoom": DEFAULT_ZOOM,
    }
    # `map` for MapLibre (Plotly 6+), `mapbox` for Mapbox (Plotly 5).
    layout_kwargs = {"map": base_layer} if use_map else {"mapbox": base_layer}
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False,
        **layout_kwargs,
    )
    return fig
