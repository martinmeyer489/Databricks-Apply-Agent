"""Unit tests for the map + chat Gradio app.

Exercises the pure/testable pieces of the redesigned frontend without a live
Databricks connection:

1. ``listings_loader`` — SQL query building (filters, skill search, escaping,
   limit cap) and row → dict coercion.
2. ``map_figure`` — Plotly figure construction from listing rows (empty and
   populated), version-robust across Plotly 5/6.
3. ``chat_handler`` — grounded-answer behaviour and graceful degradation.
4. ``main`` — ``refresh_map`` failure handling and ``handle_chat`` history
   management, without reaching the warehouse or the model.

Validates: Requirement 10 (frontend) for the map + chat redesign.
"""

from unittest.mock import patch

from src.app import chat_handler, listings_loader, main, map_figure


# ---------------------------------------------------------------------------
# 1. listings_loader — query building
# ---------------------------------------------------------------------------

def test_build_query_only_returns_mappable_enriched_rows():
    query = listings_loader.build_listings_query()
    assert "enrichment_state = 'enriched'" in query
    assert "latitude IS NOT NULL" in query
    assert "longitude IS NOT NULL" in query
    assert "LIMIT" in query


def test_build_query_applies_categorical_filters():
    query = listings_loader.build_listings_query(
        filters={"industry": "Tech", "seniority_level": "Senior"}
    )
    assert "industry = 'Tech'" in query
    assert "seniority_level = 'Senior'" in query


def test_build_query_ignores_all_sentinel_and_empty_filters():
    query = listings_loader.build_listings_query(
        filters={"industry": "All", "seniority_level": "", "employment_type": None}
    )
    assert "industry =" not in query
    assert "seniority_level =" not in query
    assert "employment_type =" not in query


def test_build_query_skill_search_matches_title_and_skills_text():
    query = listings_loader.build_listings_query(skill_query="Python")
    assert "LOWER(job_title) LIKE '%python%'" in query
    assert "required_skills_text" in query


def test_build_query_escapes_single_quotes():
    query = listings_loader.build_listings_query(filters={"industry": "O'Reilly"})
    assert "O''Reilly" in query


def test_build_query_caps_limit_at_max():
    query = listings_loader.build_listings_query(limit=100_000)
    assert f"LIMIT {listings_loader.MAX_LISTINGS}" in query


def test_build_query_applies_soft_vibe_filters():
    query = listings_loader.build_listings_query(
        filters={"office_policy": "hybrid", "benefits_rating": "good",
                 "company_vibe": "fast-paced startup"}
    )
    assert "office_policy = 'hybrid'" in query
    assert "benefits_rating = 'good'" in query
    assert "company_vibe = 'fast-paced startup'" in query


def test_rows_to_dicts_coerces_coordinates_and_drops_bad_rows():
    columns = list(listings_loader.LISTING_COLUMNS)
    # Order matches LISTING_COLUMNS: id, title, company, location, url, lat, lon,
    # seniority, employment, industry, size, vibe, office, benefits, skills.
    good = ["id1", "Dev", "ACME", "Berlin", "http://x", "52.52", "13.40",
            "Senior", "Full-time", "Tech", "Medium",
            "fast-paced", "hybrid", "good", "python, sql"]
    bad = ["id2", "Dev", "ACME", "Berlin", "http://x", "not-a-number", "13.40",
           "Senior", "Full-time", "Tech", "Medium",
           "corporate", "onsite", "basic", "python"]
    rows = listings_loader._rows_to_dicts(columns, [good, bad])
    assert len(rows) == 1
    assert rows[0]["latitude"] == 52.52
    assert isinstance(rows[0]["longitude"], float)
    assert rows[0]["office_policy"] == "hybrid"


def test_load_filter_options_degrades_to_all_without_warehouse(monkeypatch):
    monkeypatch.delenv("SQL_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_WAREHOUSE_ID", raising=False)
    options = listings_loader.load_filter_options()
    assert set(options.keys()) == set(listings_loader.FILTER_COLUMNS.keys())
    for values in options.values():
        assert values == ["All"]


# ---------------------------------------------------------------------------
# 2. map_figure
# ---------------------------------------------------------------------------

def test_build_map_figure_empty_has_no_points():
    fig = map_figure.build_map_figure([])
    assert len(fig.data) == 1
    assert len(fig.data[0].lat) == 0


def test_build_map_figure_plots_each_listing():
    listings = [
        {"job_title": "Data Engineer", "company_name": "ACME", "location_text": "Berlin",
         "latitude": 52.52, "longitude": 13.40, "industry": "Tech", "seniority_level": "Senior"},
        {"job_title": "ML Engineer", "company_name": "Beta", "location_text": "Munich",
         "latitude": 48.14, "longitude": 11.58, "industry": "Tech", "seniority_level": "Mid"},
    ]
    fig = map_figure.build_map_figure(listings)
    assert len(fig.data[0].lat) == 2
    assert "Data Engineer" in fig.data[0].text[0]


def test_build_map_figure_enables_clustering():
    fig = map_figure.build_map_figure([
        {"job_title": "J", "company_name": "C", "latitude": 52.5, "longitude": 13.4},
    ])
    assert fig.data[0].cluster.enabled is True


# ---------------------------------------------------------------------------
# 3. chat_handler
# ---------------------------------------------------------------------------

def test_answer_question_blank_prompts_for_input():
    assert "type a question" in chat_handler.answer_question("   ", [{"job_title": "x"}]).lower()


def test_answer_question_no_listings_returns_guidance():
    assert chat_handler.answer_question("any python jobs?", []) == chat_handler.NO_LISTINGS_MESSAGE


def test_answer_question_grounds_on_listings_and_returns_model_text():
    listings = [{"job_title": "Data Engineer", "company_name": "ACME",
                 "location_text": "Berlin", "required_skills_text": "python, sql"}]
    with patch.object(chat_handler, "_query_chat_llm", return_value="ACME is hiring a Data Engineer.") as m:
        answer = chat_handler.answer_question("who wants python?", listings)
    assert answer == "ACME is hiring a Data Engineer."
    # The context passed to the model mentions the listing.
    context_arg = m.call_args.args[1]
    assert "Data Engineer" in context_arg
    assert "ACME" in context_arg


def test_answer_question_degrades_gracefully_on_model_error():
    listings = [{"job_title": "Dev", "company_name": "ACME"}]
    with patch.object(chat_handler, "_query_chat_llm", side_effect=RuntimeError("boom")):
        answer = chat_handler.answer_question("q?", listings)
    assert "couldn't reach the language model" in answer


def test_build_listings_context_caps_at_max():
    many = [{"job_title": f"Role {i}", "company_name": "C"} for i in range(100)]
    context = chat_handler.build_listings_context(many)
    # Header reports the true total, body is capped.
    assert "100 listings" in context
    assert context.count("\n- ") == chat_handler.MAX_CONTEXT_LISTINGS


# ---------------------------------------------------------------------------
# 4. main handlers
# ---------------------------------------------------------------------------

def test_refresh_map_returns_error_status_without_warehouse(monkeypatch):
    monkeypatch.delenv("SQL_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_WAREHOUSE_ID", raising=False)
    fig, table, state, status = main.refresh_map(
        "All", "All", "All", "All", "All", "All", "All", ""
    )
    assert state == []
    assert table["data"] == []
    assert "Could not load listings" in status
    assert len(fig.data[0].lat) == 0


def test_refresh_map_populates_from_loaded_listings():
    listings = [
        {"job_title": "Data Engineer", "company_name": "ACME", "location_text": "Berlin",
         "latitude": 52.52, "longitude": 13.40, "seniority_level": "Senior", "industry": "Tech",
         "office_policy": "hybrid", "benefits_rating": "good", "company_vibe": "fast-paced",
         "source_url": "https://arbeitsagentur.de/jobs/1"},
    ]
    with patch.object(main, "load_listings", return_value=listings):
        fig, table, state, status = main.refresh_map(
            "All", "All", "All", "All", "All", "All", "All", ""
        )
    assert state == listings
    assert table["data"][0][0] == "Data Engineer"
    assert "hybrid" in table["data"][0]
    # Last column is a clickable markdown link to the job posting.
    assert table["data"][0][-1] == "[View job](https://arbeitsagentur.de/jobs/1)"
    assert "Showing" in status
    assert len(fig.data[0].lat) == 1


def test_handle_chat_ignores_blank_message():
    text, history = main.handle_chat("   ", [], [{"job_title": "x"}])
    assert text == ""
    assert history == []


def test_handle_chat_appends_user_and_assistant_turns():
    with patch.object(main, "answer_question", return_value="Here are the roles."):
        text, history = main.handle_chat("show me jobs", [], [{"job_title": "Dev"}])
    assert text == ""
    assert history == [
        {"role": "user", "content": "show me jobs"},
        {"role": "assistant", "content": "Here are the roles."},
    ]


def test_build_app_constructs():
    app = main.build_app()
    assert app is not None
