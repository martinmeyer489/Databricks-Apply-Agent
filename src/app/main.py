"""Databricks App (Gradio) — job-listings map + chat frontend.

A single-page app with two panels:

* **Left** — an interactive map of enriched German job listings, with filters
  (industry, seniority, employment type, company size) and a skill/keyword
  search. Filtering re-queries the SQL warehouse and redraws the map and the
  companion results table.
* **Right** — a chat box where the visitor asks free-text questions about the
  jobs currently shown on the map; answers are grounded on those listings via
  the workspace Foundation Model.

The heavy lifting lives in focused, UI-free modules so it stays testable:
``listings_loader`` (warehouse queries), ``map_figure`` (Plotly figure), and
``chat_handler`` (grounded Q&A).

Validates: Requirement 10 (frontend), reusing the enriched listings corpus and
the Foundation Model endpoint.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# The app is started as `python src/app/main.py`, which puts `src/app` (not the
# repo root) on sys.path[0], so `import src...` fails with ModuleNotFoundError.
# Add the repo root (two levels up from this file) to sys.path so the `src`
# package resolves at runtime.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gradio as gr

from src.app.chat_handler import answer_question
from src.app.listings_loader import (
    FILTER_COLUMNS,
    MAX_LISTINGS,
    load_filter_options,
    load_listings,
)
from src.app.map_figure import build_map_figure
from src.pipelines.attribute_normalization import BUCKET_CHOICES

# ---------------------------------------------------------------------------
# Results table shape
# ---------------------------------------------------------------------------
RESULTS_TABLE_COLUMNS = ["Job Title", "Company", "Location", "Seniority", "Industry", "Office", "Benefits", "Vibe", "Listing"]

# Rows fetched on first page load. Smaller than MAX_LISTINGS so the map paints
# fast; clustering still conveys overall density, and "Apply filters" pulls the
# full set on demand.
INITIAL_LOAD_LIMIT = 600


def _listings_to_table_rows(listings: List[Dict[str, Any]]) -> List[List[Any]]:
    """Project listing dicts onto the results-table columns.

    The final column is a Markdown link to the original job posting
    (``source_url``); the table renders it as a clickable "View job" link.
    """
    rows = []
    for row in listings:
        url = row.get("source_url") or ""
        link = f"[View job]({url})" if url else ""
        rows.append(
            [
                row.get("job_title"),
                row.get("company_name"),
                row.get("location_text"),
                row.get("seniority_level"),
                row.get("industry"),
                row.get("office_policy"),
                row.get("benefits_rating"),
                row.get("company_vibe"),
                link,
            ]
        )
    return rows


def _empty_table() -> Dict[str, Any]:
    """An empty results table shaped for ``gr.Dataframe``."""
    return {"headers": RESULTS_TABLE_COLUMNS, "data": []}


# ---------------------------------------------------------------------------
# Filter option loading (best-effort at startup)
# ---------------------------------------------------------------------------

def _safe_filter_options() -> Dict[str, List[str]]:
    """Return the fixed emoji-bucket choices for each filter.

    The enrichment pipeline stores every filterable attribute as one of a
    small set of fixed emoji+text buckets (see attribute_normalization), so the
    dropdowns offer exactly those buckets — no warehouse query needed. This
    keeps each dropdown to 4-5 stable choices and removes several SQL round
    trips from app startup / first paint. "All" is prepended as the no-filter
    default.
    """
    return {name: ["All", *choices] for name, choices in BUCKET_CHOICES.items()}


# ---------------------------------------------------------------------------
# Map / filter event handler
# ---------------------------------------------------------------------------

def refresh_map(
    industry: Optional[str],
    seniority_level: Optional[str],
    employment_type: Optional[str],
    company_size_band: Optional[str],
    office_policy: Optional[str],
    benefits_rating: Optional[str],
    company_vibe: Optional[str],
    skill_query: Optional[str],
    limit: Optional[int] = None,
) -> Tuple[Any, Any, List[Dict[str, Any]], str]:
    """Re-query listings for the selected filters and rebuild the map + table.

    Returns a 4-tuple of (map figure, results table, listings state, status
    markdown). On a warehouse failure the map/table are cleared and the status
    line explains why, while never raising into the UI.

    ``limit`` bounds how many rows are fetched (defaults to the loader's
    ``MAX_LISTINGS``); the initial page load passes a smaller value for a fast
    first paint, and "Apply filters" uses the full cap.
    """
    filters = {
        "industry": industry,
        "seniority_level": seniority_level,
        "employment_type": employment_type,
        "company_size_band": company_size_band,
        "office_policy": office_policy,
        "benefits_rating": benefits_rating,
        "company_vibe": company_vibe,
    }
    try:
        listings = load_listings(
            filters=filters, skill_query=skill_query, limit=limit or MAX_LISTINGS
        )
    except Exception as exc:  # noqa: BLE001
        status = (
            f"Could not load listings ({type(exc).__name__}). "
            "This app reads the enriched-listings table from the SQL warehouse, "
            "so it must run as a Databricks App (or with SQL_WAREHOUSE_ID and "
            "workspace credentials set) and the warehouse must be running."
        )
        return build_map_figure([]), _empty_table(), [], status

    count = len(listings)
    if count == 0:
        status = "No listings match these filters. Try widening them or clearing the skill search."
    else:
        capped = " (showing the first " + str(MAX_LISTINGS) + ")" if count >= MAX_LISTINGS else ""
        status = f"Showing **{count}** listings on the map{capped}."

    table = {"headers": RESULTS_TABLE_COLUMNS, "data": _listings_to_table_rows(listings)}
    return build_map_figure(listings), table, listings, status


# ---------------------------------------------------------------------------
# Chat event handler
# ---------------------------------------------------------------------------

def handle_chat(
    message: str,
    history: List[Dict[str, str]],
    listings: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, str]]]:
    """Answer a chat message grounded on the currently-visible listings.

    Uses the ``messages``-format chatbot (list of ``{"role", "content"}``
    dicts). Returns the cleared textbox value and the updated history.
    """
    history = list(history or [])
    if not message or not message.strip():
        return "", history

    answer = answer_question(message, listings)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return "", history


# ---------------------------------------------------------------------------
# Theming (Req 10 — polished, responsive single-page layout)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container {
    max-width: 1600px !important;
    margin: 0 auto !important;
}
#hero {
    background: linear-gradient(120deg, #1B3139 0%, #FF5F46 140%);
    color: #ffffff;
    padding: 22px 28px;
    border-radius: 16px;
    margin-bottom: 12px;
}
#hero h1 { margin: 0 0 4px 0; font-size: 1.6rem; color: #ffffff !important; }
#hero p { margin: 0; opacity: 0.9; color: #ffffff !important; }
#map_panel .js-plotly-plot, #map_panel .plotly { border-radius: 14px; }
#chat_panel { border-left: 1px solid rgba(0,0,0,0.06); padding-left: 8px; }
.map-status { font-size: 0.9rem; opacity: 0.85; }
"""

# A polished built-in theme with the Databricks-ish accent.
APP_THEME = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="slate",
    neutral_hue="slate",
)


def _make_chatbot():
    """Construct a messages-format Chatbot across Gradio versions.

    Gradio 5.x needs ``type="messages"`` to opt into the OpenAI-style
    ``{"role", "content"}`` history (its default is the deprecated tuples
    format); Gradio 6.x removed the ``type`` kwarg because messages is the
    only format. Pass ``type`` only when the installed version accepts it.
    """
    import inspect

    kwargs: Dict[str, Any] = {"label": "Job assistant", "height": 460}
    if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
        kwargs["type"] = "messages"
    return gr.Chatbot(**kwargs)


def build_app() -> gr.Blocks:
    """Construct the map + filters + chat Gradio Blocks app."""
    filter_options = _safe_filter_options()

    with gr.Blocks(title="Job Listings Map", theme=APP_THEME, css=CUSTOM_CSS) as demo:
        # Session-scoped state: the listings currently drawn on the map, so the
        # chat can ground its answers on exactly what the user is looking at.
        listings_state = gr.State([])

        gr.HTML(
            """
            <div id="hero">
              <h1>🗺️ German Job Listings — Explore &amp; Ask</h1>
              <p>Filter live job postings on the map, then ask the assistant anything about them.</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            # ---------------- Left: filters + map ----------------
            with gr.Column(scale=3, elem_id="map_panel"):
                with gr.Row():
                    industry_dd = gr.Dropdown(
                        label="Industry",
                        choices=filter_options.get("industry", ["All"]),
                        value="All",
                    )
                    seniority_dd = gr.Dropdown(
                        label="Seniority",
                        choices=filter_options.get("seniority_level", ["All"]),
                        value="All",
                    )
                    employment_dd = gr.Dropdown(
                        label="Employment type",
                        choices=filter_options.get("employment_type", ["All"]),
                        value="All",
                    )
                    company_size_dd = gr.Dropdown(
                        label="Company size",
                        choices=filter_options.get("company_size_band", ["All"]),
                        value="All",
                    )
                with gr.Row():
                    office_policy_dd = gr.Dropdown(
                        label="Office policy",
                        choices=filter_options.get("office_policy", ["All"]),
                        value="All",
                    )
                    benefits_dd = gr.Dropdown(
                        label="Benefits",
                        choices=filter_options.get("benefits_rating", ["All"]),
                        value="All",
                    )
                    vibe_dd = gr.Dropdown(
                        label="Company vibe",
                        choices=filter_options.get("company_vibe", ["All"]),
                        value="All",
                    )
                with gr.Row():
                    skill_search = gr.Textbox(
                        label="Skill / keyword search",
                        placeholder="e.g. Python, SQL, data engineering…",
                        scale=4,
                    )
                    apply_button = gr.Button("Apply filters", variant="primary", scale=1)

                map_status = gr.Markdown(
                    "Click **Apply filters** to load listings onto the map.",
                    elem_classes=["map-status"],
                )
                map_plot = gr.Plot(label="Job listings map")
                results_table = gr.Dataframe(
                    headers=RESULTS_TABLE_COLUMNS,
                    label="Listings",
                    interactive=False,
                    wrap=True,
                    # Render the final "Listing" column as Markdown so its
                    # "[View job](url)" value becomes a clickable link to the
                    # original job posting; all other columns stay plain text.
                    datatype=["str", "str", "str", "str", "str", "str", "str", "str", "markdown"],
                )

            # ---------------- Right: chat ----------------
            with gr.Column(scale=2, elem_id="chat_panel"):
                gr.Markdown("### 💬 Ask about these jobs")
                chatbot = _make_chatbot()
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Which roles want Python? Who's hiring seniors in Berlin?",
                        show_label=False,
                        scale=4,
                    )
                    chat_send = gr.Button("Send", variant="primary", scale=1)
                gr.Markdown(
                    "_Answers are grounded on the listings currently shown on the map._"
                )

        # ---------------- Wiring ----------------
        filter_inputs = [
            industry_dd,
            seniority_dd,
            employment_dd,
            company_size_dd,
            office_policy_dd,
            benefits_dd,
            vibe_dd,
            skill_search,
        ]
        map_outputs = [map_plot, results_table, listings_state, map_status]

        apply_button.click(fn=refresh_map, inputs=filter_inputs, outputs=map_outputs)
        skill_search.submit(fn=refresh_map, inputs=filter_inputs, outputs=map_outputs)

        # Load the map once on page open. Use a smaller initial cap for a fast
        # first paint (clustering still conveys density); "Apply filters" then
        # pulls up to the full MAX_LISTINGS.
        def _initial_load(*filter_values):
            return refresh_map(*filter_values, limit=INITIAL_LOAD_LIMIT)

        demo.load(fn=_initial_load, inputs=filter_inputs, outputs=map_outputs)

        chat_send.click(
            fn=handle_chat,
            inputs=[chat_input, chatbot, listings_state],
            outputs=[chat_input, chatbot],
        )
        chat_input.submit(
            fn=handle_chat,
            inputs=[chat_input, chatbot, listings_state],
            outputs=[chat_input, chatbot],
        )

    return demo


demo = build_app()


if __name__ == "__main__":
    # Databricks Apps expose the port to bind via DATABRICKS_APP_PORT (default
    # 8000); bind to 0.0.0.0 so the platform can route to the app.
    _port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    demo.launch(server_name="0.0.0.0", server_port=_port)
