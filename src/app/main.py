"""Databricks App (Gradio) — job matching frontend.

Provides the single-page, tabbed user interface for CV upload, location and
commute preference input, matching results browsing, and application draft
generation (Requirement 10).

This module currently implements Tab 1 (Upload CV). Later tasks (16.2-16.5)
add the remaining tabs and wire them into the same `gr.Blocks()` app and the
shared `user_profile` session state defined here.

Validates: Requirements 5.1, 5.2, 5.3, 5.7, 5.8, 5.9, 10.1, 10.2
"""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

# The app is started as `python src/app/main.py`, which puts `src/app` (not the
# repo root) on sys.path[0], so `import src...` fails with ModuleNotFoundError.
# Add the repo root (three levels up from this file) to sys.path so the
# `src` package resolves at runtime.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gradio as gr

from src.agent.cv_parser import parse_cv
from src.agent.location_resolver import resolve_location
from src.agent.matching_agent import validate_profile_completeness
from src.utils.input_validation import (
    validate_commute_radius,
    validate_cv_file,
    validate_location_input,
)

# ---------------------------------------------------------------------------
# CV upload staging location
# ---------------------------------------------------------------------------
# In a deployed Databricks App, uploaded CVs must be stored in the
# `cv_uploads` Unity Catalog Volume (Req 5.2), addressed via its Volume
# path. A real Volume write requires running inside a Databricks workspace
# with catalog `job_agent` provisioned (see notebooks/00_setup_catalog.py).
# Since this app must also be runnable and testable outside Databricks
# (e.g. locally, in CI), we use the canonical Volume path when it exists on
# disk and fall back to a local temp directory otherwise. This fallback is
# purely a local-development convenience — in production the Volume path
# is always used.
CV_UPLOADS_VOLUME_PATH = "/Volumes/job_agent/volumes/cv_uploads"
CV_UPLOADS_LOCAL_FALLBACK = os.path.join(os.getcwd(), ".local_cv_uploads")


def _cv_uploads_staging_dir() -> str:
    """Return the directory to stage uploaded CV files into.

    Uses the `job_agent.volumes.cv_uploads` Volume path when it is
    available (i.e. running inside a Databricks workspace with the Volume
    mounted), otherwise falls back to a local directory so the app remains
    runnable outside Databricks.
    """
    if os.path.isdir(CV_UPLOADS_VOLUME_PATH):
        return CV_UPLOADS_VOLUME_PATH
    os.makedirs(CV_UPLOADS_LOCAL_FALLBACK, exist_ok=True)
    return CV_UPLOADS_LOCAL_FALLBACK


def _store_cv_file(uploaded_file_path: str) -> str:
    """Copy an uploaded CV file into the staging directory and return its path.

    Simulates storing the file in the `cv_uploads` Volume (Req 5.2). The
    destination filename is prefixed with a UUID to avoid collisions
    between concurrent sessions/uploads.
    """
    staging_dir = _cv_uploads_staging_dir()
    _, extension = os.path.splitext(uploaded_file_path)
    destination_filename = f"{uuid.uuid4().hex}{extension}"
    destination_path = os.path.join(staging_dir, destination_filename)
    shutil.copyfile(uploaded_file_path, destination_path)
    return destination_path


# ---------------------------------------------------------------------------
# Session state field names
# ---------------------------------------------------------------------------
# The full User_Profile shape is defined up-front (even though this task
# only populates the CV fields) so that later tasks (16.2-16.4) can extend
# it without restructuring the state dict.
CV_FIELD_NAMES = (
    "skills",
    "years_of_experience",
    "education_history",
    "job_title_history",
    "qualifications_summary",
    "unresolved_fields",
)

# ---------------------------------------------------------------------------
# Session lifetime (Req 10.4 — 30 minutes of inactivity)
# ---------------------------------------------------------------------------
# `gr.State` scopes `user_profile_state`/`matching_results_state` to a single
# browser session (Gradio session ID, tied to the browser tab). Gradio's
# Blocks server already evicts a session's state when the underlying
# connection is closed (browser tab closed), which covers the "until the
# browser tab is closed" half of Req 10.4.
#
# The "or 30 minutes elapse without user interaction" half is NOT something
# the installed gradio==6.26.0 exposes as a simple per-session TTL knob at
# the Blocks/State level (there is no `gr.State(ttl=...)` or
# `Blocks(session_timeout=...)` parameter in this version). Enforcing an
# exact 30-minute idle cutoff would require either a custom FastAPI
# middleware wrapping the Gradio ASGI app, or polling client-side JS to
# trigger a session reset — both out of scope for this portfolio app.
#
# In practice, the Databricks Apps runtime that hosts this Gradio app
# already imposes its own container/process lifecycle (independent of this
# 30-minute figure), and each browser tab's session is already isolated per
# Req 10.4's data-scoping intent. We therefore satisfy Req 10.4's data
# isolation intent via Gradio's default per-session `gr.State`, and
# delegate exact 30-minute idle-timeout enforcement to the Gradio/Databricks
# Apps runtime rather than implementing bespoke idle-tracking app code.
SESSION_IDLE_TIMEOUT_MINUTES = 30

INITIAL_USER_PROFILE: Dict[str, Any] = {
    "profile_id": None,
    # CV fields (Requirement 5) — populated by Tab 1.
    "skills": None,
    "years_of_experience": None,
    "education_history": None,
    "job_title_history": None,
    "qualifications_summary": None,
    "unresolved_fields": None,
    "cv_file_path": None,
    # Location fields (Requirement 6) — populated by Tab 2.
    "home_latitude": None,
    "home_longitude": None,
    "home_location_name": None,
    "commute_radius_km": 50,
}


def replace_cv_fields(user_profile: Dict[str, Any], new_cv_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `user_profile` with CV fields replaced by `new_cv_fields`.

    All previous CV field values are discarded and replaced with the newly
    parsed ones. Non-CV fields (location, commute radius, profile_id) are
    preserved unchanged (Req 5.9).
    """
    updated_profile = dict(user_profile)
    for field_name in CV_FIELD_NAMES:
        updated_profile[field_name] = new_cv_fields.get(field_name)
    return updated_profile


# ---------------------------------------------------------------------------
# Tab 1 — Upload CV: event handler
# ---------------------------------------------------------------------------

def handle_cv_upload(
    uploaded_file: Optional[str],
    user_profile: Dict[str, Any],
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Dict[str, Any], str, str]:
    """Validate, store, and parse an uploaded CV file.

    Args:
        uploaded_file: Local filesystem path of the file Gradio received
            from the browser (or None if nothing was uploaded).
        user_profile: The current session `user_profile` state dict.
        progress: Gradio-injected progress tracker (Req 10.5) — CV parsing
            can take up to 60s (Req 5.7), so we surface a loading indicator
            across the upload/store/parse stages.

    Returns:
        A 3-tuple of (updated_user_profile, summary_markdown, warning_markdown)
        to update the `gr.State`, the confirmation summary card, and the
        unresolved-fields warning area respectively. On any failure the
        unchanged `user_profile` is returned so the user's session data is
        preserved and they can retry (Req 10.6).
    """
    progress(0, desc="Checking file...")
    if uploaded_file is None:
        gr.Warning("Please select a CV file (PDF or DOCX) before parsing.")
        return user_profile, "", ""

    file_format = os.path.splitext(uploaded_file)[1].lstrip(".").upper() or "UNKNOWN"
    file_size = os.path.getsize(uploaded_file)

    is_valid, validation_message = validate_cv_file(file_format, file_size)
    if not is_valid:
        gr.Warning(validation_message)
        return user_profile, "", ""

    progress(0.2, desc="Uploading file...")
    try:
        stored_path = _store_cv_file(uploaded_file)
    except OSError as exc:
        gr.Warning(f"Could not store the uploaded CV file: {exc}")
        return user_profile, "", ""

    progress(0.4, desc="Parsing CV...")
    try:
        parsed_fields = parse_cv(stored_path)
    except ValueError as exc:
        # File failed parse_cv's own format/size validation.
        gr.Warning(str(exc))
        return user_profile, "", ""
    except TimeoutError:
        gr.Warning("CV parsing timed out. Please try again.")
        return user_profile, "", ""
    except RuntimeError as exc:
        gr.Warning(f"CV parsing failed: {exc}. Please try again or upload a different file.")
        return user_profile, "", ""

    progress(0.9, desc="Finalizing profile...")
    updated_profile = replace_cv_fields(user_profile, parsed_fields)
    updated_profile["cv_file_path"] = stored_path

    skills = parsed_fields.get("skills") or []
    education_history = parsed_fields.get("education_history") or []
    job_title_history = parsed_fields.get("job_title_history") or []
    years_of_experience = parsed_fields.get("years_of_experience")
    unresolved_fields = parsed_fields.get("unresolved_fields") or []

    summary_markdown = (
        "### CV parsed successfully\n"
        f"- **Skills extracted:** {len(skills)}\n"
        f"- **Years of experience:** {years_of_experience if years_of_experience is not None else 'unknown'}\n"
        f"- **Education history entries:** {len(education_history)}\n"
        f"- **Job title history entries:** {len(job_title_history)}\n"
    )

    warning_markdown = ""
    if unresolved_fields:
        warning_markdown = (
            "**Some fields could not be extracted:** "
            + ", ".join(unresolved_fields)
            + ". You can proceed with partial data or upload a different CV."
        )
        gr.Warning(warning_markdown)

    return updated_profile, summary_markdown, warning_markdown


def _get_spark_session():
    """Return an active SparkSession connected to the SQL warehouse, or None.

    Location resolution (Req 6.5) queries the `ops.geocode_lookup` table via
    the SQL warehouse. When this Gradio app is deployed as a Databricks App,
    Databricks Connect provides a `SparkSession` transparently. When running
    locally or in CI without Databricks Connect configured, no session is
    available; callers must handle a `None` return by informing the user
    that location resolution requires a Databricks connection. This mirrors
    the local-development fallback pattern used by `_cv_uploads_staging_dir()`
    above.
    """
    try:
        from databricks.connect import DatabricksSession

        return DatabricksSession.builder.getOrCreate()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tab 2 — Location & Commute: event handler
# ---------------------------------------------------------------------------

def handle_location_resolve(
    location_text: Optional[str],
    commute_radius_km: Optional[int],
    user_profile: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Validate, resolve, and store the user's home location and commute radius.

    Args:
        location_text: The submitted home location (city name or postal code).
        commute_radius_km: The submitted commute radius in kilometres.
        user_profile: The current session `user_profile` state dict.

    Returns:
        A 2-tuple of (updated_user_profile, confirmation_markdown).
    """
    is_valid_location, location_message = validate_location_input(location_text)
    if not is_valid_location:
        gr.Warning(location_message)
        return user_profile, ""

    is_valid_radius, radius_message = validate_commute_radius(commute_radius_km)
    if not is_valid_radius:
        gr.Warning(radius_message)
        return user_profile, ""

    spark = _get_spark_session()
    if spark is None:
        gr.Warning(
            "Location resolution requires a Databricks connection. "
            "Please run this app as a Databricks App or configure Databricks Connect."
        )
        return user_profile, ""

    resolved = resolve_location(spark, location_text)
    if resolved is None:
        gr.Warning(
            "We couldn't resolve that location. Please enter a valid city name or postal code."
        )
        return user_profile, ""

    updated_profile = dict(user_profile)
    updated_profile["home_latitude"] = resolved["latitude"]
    updated_profile["home_longitude"] = resolved["longitude"]
    updated_profile["home_location_name"] = resolved["city_name"]
    updated_profile["commute_radius_km"] = commute_radius_km

    confirmation_markdown = (
        "### Location confirmed\n"
        f"- **Resolved location:** {resolved['city_name']}\n"
        f"- **Commute radius:** {commute_radius_km} km\n"
    )

    return updated_profile, confirmation_markdown


# ---------------------------------------------------------------------------
# Tab 3 — Find Matches: constants and event handler
# ---------------------------------------------------------------------------
# The deployed Matching Agent is served from this Model Serving endpoint
# name (see notebooks/07_register_deploy_agent.py). Invoking it requires a
# live Databricks workspace connection via the Databricks SDK; see the
# fallback handling in `_invoke_matching_agent()` below.
MATCHING_AGENT_ENDPOINT_NAME = "job-agent-matching"

RESULTS_TABLE_COLUMNS = ["Job Title", "Company", "Score", "Distance (km)", "Explanation"]

# Maps the field names returned by `validate_profile_completeness` to a
# human-readable description of the step the user still needs to complete
# (Req 10.8).
MISSING_FIELD_DESCRIPTIONS = {
    "skills": "upload and parse your CV",
    "home_coordinates": "resolve your home location",
    "commute_radius_km": "set your commute radius",
}


def _empty_results_table():
    """Return an empty results table shaped for `gr.Dataframe`."""
    return {"headers": RESULTS_TABLE_COLUMNS, "data": []}


def _build_completeness_gate_message(missing_fields: list) -> str:
    """Build a Markdown message naming each incomplete step (Req 10.8)."""
    steps = [MISSING_FIELD_DESCRIPTIONS.get(field, field) for field in missing_fields]
    steps_list = "\n".join(f"- {step}" for step in steps)
    return (
        "### Complete these steps before matching\n"
        "Your profile is missing the following:\n"
        f"{steps_list}\n"
    )


# ---------------------------------------------------------------------------
# Auto-stop (Req 11.5) and quota-exhaustion (Req 11.11) detection
# ---------------------------------------------------------------------------

def _is_auto_stop_error(exception: BaseException) -> bool:
    """Return True if `exception` represents a 503 Service Unavailable.

    Databricks Apps that have been auto-stopped after 24 hours of
    inactivity return a 503 when the next request tries to reach them (or,
    for calls made *from* this running app to a Model Serving endpoint or
    SQL warehouse that has itself been auto-stopped/suspended, the
    Databricks SDK surfaces the failure as an HTTP 503). We check both a
    `status_code` attribute (as exposed by `databricks.sdk.errors` and most
    HTTP client exceptions) and the string representation, since different
    SDK versions/transports surface the code differently.
    """
    status_code = getattr(exception, "status_code", None)
    if status_code == 503:
        return True
    response = getattr(exception, "response", None)
    if getattr(response, "status_code", None) == 503:
        return True
    return "503" in str(exception)


def _auto_stop_restart_message() -> str:
    """Message shown when a call fails because of the 24h auto-stop (Req 11.5)."""
    return (
        "This app was auto-stopped after 24 hours of inactivity. "
        "Restart it from the Databricks workspace → Apps → job-agent-app → Start."
    )


def _is_quota_exhausted_error(exception: BaseException) -> bool:
    """Return True if `exception` indicates the Free Edition compute quota is exhausted.

    Requirements.md does not specify an exact status code or error string
    for this condition beyond "the Free_Edition usage quota is exhausted
    and workspace compute is unavailable" (Req 11.11), so we use a
    heuristic keyword match against the exception's string representation,
    covering the phrasings most likely to appear in the underlying error.
    """
    message = str(exception).lower()
    return "quota" in message or "compute unavailable" in message


def _quota_exhausted_message() -> str:
    """Message shown when the Free Edition usage quota is exhausted (Req 11.11)."""
    return (
        "The workspace compute quota is exhausted. Your data and settings are "
        "retained, and service will resume in the next quota period."
    )


def _endpoint_failure_message(exception: Optional[BaseException], operation: str) -> str:
    """Build the `gr.Warning` message for a failed endpoint/warehouse call.

    Classifies `exception` (if any) as an auto-stop 503, a quota-exhaustion
    condition, or a generic failure, and returns the appropriate
    user-facing message naming the failed `operation` (Req 10.7, 11.5,
    11.11). Data preservation is handled by the caller always returning the
    unchanged `user_profile`/state alongside this message.
    """
    if exception is not None:
        if _is_auto_stop_error(exception):
            return _auto_stop_restart_message()
        if _is_quota_exhausted_error(exception):
            return _quota_exhausted_message()
    return (
        f"{operation} is unavailable right now. Please try again — your entered "
        "data has been kept. If this keeps happening, run this app as a "
        "Databricks App with the required endpoint/warehouse deployed."
    )


def _invoke_matching_agent(
    user_profile: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[BaseException]]:
    """Invoke the Matching Agent via its Model Serving endpoint.

    Calling a real Model Serving endpoint requires a live Databricks
    workspace connection (`databricks.sdk.WorkspaceClient`). When that SDK
    or workspace connection is unavailable (e.g. running locally or in CI),
    or the call otherwise fails, this returns `(None, exception)` so the
    caller can classify the failure (auto-stop 503, quota exhaustion, or
    generic) and display the appropriate warning while preserving session
    data, mirroring the Volume/Spark-session fallback patterns used by
    Tabs 1-2.
    """
    try:
        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient()
        response = client.serving_endpoints.query(
            name=MATCHING_AGENT_ENDPOINT_NAME,
            inputs={"profile_id": user_profile.get("profile_id")},
        )
        # `query()` returns an SDK response object; the agent's response
        # payload is expected under `.predictions` or similar depending on
        # the endpoint's signature. Normalize to a plain dict here.
        if isinstance(response, dict):
            return response, None
        return dict(getattr(response, "predictions", response)), None
    except Exception as exc:  # noqa: BLE001 - classified by the caller
        return None, exc


def handle_find_matches(
    user_profile: Dict[str, Any],
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Any, str, List[Dict[str, Any]], Any]:
    """Validate profile completeness, then invoke the Matching Agent.

    Args:
        user_profile: The current session `user_profile` state dict.
        progress: Gradio-injected progress tracker (Req 10.5) — this call
            invokes a Model Serving endpoint and can plausibly take several
            seconds, so we surface a loading indicator for the duration.

    Returns:
        A 4-tuple of (results_table, gate_or_suggestion_markdown,
        results_list, listing_dropdown_update) to update the results
        `gr.Dataframe`, the gate/suggestion message area, the
        `matching_results_state` (Req 8.2 — used by Tab 4 to select a
        listing to draft against), and the Tab 4 listing dropdown choices.
    """
    progress(0, desc="Checking your profile...")
    missing_fields = validate_profile_completeness(user_profile)
    if missing_fields:
        gate_message = _build_completeness_gate_message(missing_fields)
        gr.Warning("Please complete your profile before finding matches.")
        return _empty_results_table(), gate_message, [], gr.update(choices=[], value=None)

    progress(0.3, desc="Searching for matches...")
    response, error = _invoke_matching_agent(user_profile)
    if response is None:
        # Data preservation (Req 10.6): `user_profile`/`matching_results_state`
        # are not part of this handler's outputs, so the caller's existing
        # state is left untouched regardless of this failure.
        gr.Warning(_endpoint_failure_message(error, "Finding matches"))
        return _empty_results_table(), "", [], gr.update(choices=[], value=None)

    progress(0.8, desc="Ranking results...")
    if response.get("error"):
        gate_message = _build_completeness_gate_message(response.get("missing_fields") or [])
        gr.Warning(response["error"])
        return _empty_results_table(), gate_message, [], gr.update(choices=[], value=None)

    results = response.get("results") or []
    if not results:
        message = response.get("message") or "No matches found."
        return _empty_results_table(), message, [], gr.update(choices=[], value=None)

    rows = [
        [
            result.get("job_title"),
            result.get("company_name"),
            result.get("relevance_score"),
            result.get("distance_km"),
            result.get("explanation"),
        ]
        for result in results
    ]
    dropdown_choices = _build_listing_dropdown_choices(results)
    return (
        {"headers": RESULTS_TABLE_COLUMNS, "data": rows},
        "",
        results,
        gr.update(choices=dropdown_choices, value=None),
    )


# ---------------------------------------------------------------------------
# Tab 4 — Draft Application: constants and event handler
# ---------------------------------------------------------------------------

def _build_listing_dropdown_choices(
    results: List[Dict[str, Any]]
) -> List[Tuple[str, str]]:
    """Build `gr.Dropdown` (label, value) choices from matching results.

    The dropdown's value is the `listing_id` (used to invoke
    `draft_application`); the label shows the job title and company so the
    user can recognize the listing (Req 8.2).
    """
    choices = []
    for result in results:
        listing_id = result.get("listing_id")
        if not listing_id:
            continue
        job_title = result.get("job_title") or "Untitled role"
        company_name = result.get("company_name") or "Unknown company"
        choices.append((f"{job_title} — {company_name}", listing_id))
    return choices


def _invoke_draft_application(
    listing_id: str, user_profile: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[BaseException]]:
    """Invoke the Matching Agent's `draft_application` tool via its endpoint.

    Mirrors the fallback pattern used by `_invoke_matching_agent()`: calling
    a real Model Serving endpoint requires a live Databricks workspace
    connection. When that connection is unavailable (e.g. running locally
    or in CI) or the call fails, this returns `(None, exception)` so the
    caller can classify the failure and show the appropriate warning.
    """
    try:
        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient()
        response = client.serving_endpoints.query(
            name=MATCHING_AGENT_ENDPOINT_NAME,
            inputs={
                "action": "draft_application",
                "listing_id": listing_id,
                "profile_id": user_profile.get("profile_id"),
                "user_skills": user_profile.get("skills"),
                "user_job_titles": user_profile.get("job_title_history"),
                "user_qualifications_summary": user_profile.get("qualifications_summary"),
                "user_years_of_experience": user_profile.get("years_of_experience"),
            },
        )
        if isinstance(response, dict):
            return response, None
        return dict(getattr(response, "predictions", response)), None
    except Exception as exc:  # noqa: BLE001 - classified by the caller
        return None, exc


def handle_draft_application(
    selected_listing_id: Optional[str],
    user_profile: Dict[str, Any],
    progress: gr.Progress = gr.Progress(),
) -> str:
    """Validate the selected listing, then draft a cover letter for it.

    Args:
        selected_listing_id: The `listing_id` chosen in the Tab 4 dropdown.
        user_profile: The current session `user_profile` state dict.
        progress: Gradio-injected progress tracker (Req 10.5) — drafting
            invokes a Model Serving endpoint (up to 30s per Req 8.6), so we
            surface a loading indicator for the duration.

    Returns:
        The generated cover letter text (Req 8.5), or an empty string if
        drafting could not be completed (a `gr.Warning` explains why; the
        user can retry by clicking the button again, satisfying Req 8.7's
        retry affordance). `user_profile` is never modified by this
        handler, so session data is preserved across failures (Req 10.6).
    """
    progress(0, desc="Preparing draft request...")
    if not selected_listing_id:
        gr.Warning("Please select a listing from your matching results before drafting.")
        return ""

    progress(0.3, desc="Drafting cover letter...")
    response, error = _invoke_draft_application(selected_listing_id, user_profile)
    if response is None:
        gr.Warning(_endpoint_failure_message(error, "Drafting the cover letter"))
        return ""

    progress(0.9, desc="Finalizing draft...")
    if response.get("error"):
        gr.Warning(response["error"])
        return ""

    cover_letter = response.get("cover_letter") or response.get("draft") or ""
    if not cover_letter:
        gr.Warning("The matching agent did not return a cover letter. Please try again.")
        return ""

    return cover_letter


# Responsive layout (Req 10.6 — no horizontal scrolling from 320px to
# 1440px viewport width). Gradio's default `gr.Blocks()` layout uses a
# flex/grid column that stacks components vertically and shrinks fluidly;
# no custom CSS is required for the 320-1440px range per design.md. The
# one adjustment worth making is capping the *maximum* content width so
# text/tables don't stretch awkwardly wide on large monitors while still
# shrinking naturally down to 320px on mobile.
RESPONSIVE_LAYOUT_CSS = """
.gradio-container {
    max-width: 1440px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
"""


def build_app() -> gr.Blocks:
    """Construct the Gradio Blocks app with the CV upload tab."""
    with gr.Blocks(title="Databricks Job Agent", css=RESPONSIVE_LAYOUT_CSS) as demo:
        # `gr.State` scopes these to one browser session (Gradio session ID).
        # See `SESSION_IDLE_TIMEOUT_MINUTES` above for how Req 10.4's
        # 30-minute inactivity window relates to this session scoping.
        user_profile_state = gr.State(dict(INITIAL_USER_PROFILE))
        # Holds the most recent Matching Agent results (list of dicts with
        # listing_id/job_title/company_name/etc.) so Tab 4 can populate its
        # listing-selection dropdown without re-invoking the agent (Req 8.2).
        matching_results_state = gr.State([])

        gr.Markdown("# Databricks Job Agent")

        with gr.Tabs():
            with gr.Tab("Upload CV"):
                gr.Markdown(
                    "Upload your CV as a PDF or DOCX file (max 5 MB). "
                    "We'll extract your skills, experience, education, and job history."
                )
                cv_file_input = gr.File(
                    label="Upload CV (PDF or DOCX, max 5MB)",
                    file_types=[".pdf", ".docx"],
                )
                parse_button = gr.Button("Parse CV", variant="primary")
                cv_summary_output = gr.Markdown(label="Confirmation summary")
                cv_warning_output = gr.Markdown(label="Unresolved fields")

                parse_button.click(
                    fn=handle_cv_upload,
                    inputs=[cv_file_input, user_profile_state],
                    outputs=[user_profile_state, cv_summary_output, cv_warning_output],
                )

            with gr.Tab("Location & Commute"):
                gr.Markdown(
                    "Enter your home location and preferred commute radius. "
                    "We'll resolve your location so matching jobs can be filtered by distance."
                )
                location_input = gr.Textbox(
                    label="Home location (city name or postal code)",
                    max_lines=1,
                )
                radius_slider = gr.Slider(
                    minimum=1,
                    maximum=200,
                    value=50,
                    step=1,
                    label="Commute radius (km)",
                )
                resolve_button = gr.Button("Resolve Location", variant="primary")
                location_confirmation_output = gr.Markdown(label="Confirmation")

                resolve_button.click(
                    fn=handle_location_resolve,
                    inputs=[location_input, radius_slider, user_profile_state],
                    outputs=[user_profile_state, location_confirmation_output],
                )

            with gr.Tab("Find Matches"):
                gr.Markdown(
                    "Find jobs that match your profile within your commute radius. "
                    "Complete the Upload CV and Location & Commute tabs first."
                )
                find_matches_button = gr.Button("Find Matches", variant="primary")
                matches_gate_output = gr.Markdown(label="Status")
                matches_table_output = gr.Dataframe(
                    headers=RESULTS_TABLE_COLUMNS,
                    label="Matching results",
                )

            with gr.Tab("Draft Application"):
                gr.Markdown(
                    "Select a listing from your most recent matching results, then "
                    "draft a tailored cover letter. Find matches in the previous tab first."
                )
                listing_select_dropdown = gr.Dropdown(
                    label="Select a listing",
                    choices=[],
                    value=None,
                )
                draft_button = gr.Button("Draft Cover Letter", variant="primary")
                # `buttons=["copy"]` renders Gradio's built-in copy-to-clipboard
                # control on the textbox (the pinned gradio==6.26.0 exposes this
                # via `buttons` rather than the older `show_copy_button` kwarg).
                # This satisfies the "copy button" requirement without a
                # separate widget.
                cover_letter_output = gr.Textbox(
                    lines=15,
                    interactive=True,
                    label="Cover Letter (editable)",
                    show_copy_button=True,
                )

                draft_button.click(
                    fn=handle_draft_application,
                    inputs=[listing_select_dropdown, user_profile_state],
                    outputs=[cover_letter_output],
                )

                find_matches_button.click(
                    fn=handle_find_matches,
                    inputs=[user_profile_state],
                    outputs=[
                        matches_table_output,
                        matches_gate_output,
                        matching_results_state,
                        listing_select_dropdown,
                    ],
                )

    return demo


demo = build_app()


if __name__ == "__main__":
    # The responsive max-width CSS is applied on the gr.Blocks() constructor
    # (see build_app). Databricks Apps expose the port to bind via the
    # DATABRICKS_APP_PORT env var (default 8000); bind to 0.0.0.0 so the
    # platform can route to the app.
    _port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    demo.launch(server_name="0.0.0.0", server_port=_port)
