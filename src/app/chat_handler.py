"""Chat handler for the job-listings assistant.

The chat box lets a visitor ask free-text questions about the jobs currently
shown on the map. Answers are grounded on those visible listings: the question
and a compact summary of the filtered listings are sent to the workspace
Foundation Model (Llama 3.3 70B) via the OpenAI-compatible client — the same
client pattern the CV parser uses (``serving_endpoints.get_open_ai_client()``).

Kept independent of ``gradio`` so it can be unit-tested without a UI. The
Databricks SDK is imported lazily so importing this module never requires a
workspace connection.

Validates: conversational job Q&A for the redesigned app.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Foundation Model APIs chat endpoint (matches cv_parser / matching_agent).
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# Cap how many listings we describe to the model so the prompt stays bounded.
MAX_CONTEXT_LISTINGS = 40

SYSTEM_PROMPT = (
    "You are a helpful job-search assistant for a map of German job listings. "
    "Answer the user's question using ONLY the listings provided as context. "
    "Be concise, reference companies and job titles by name, and if the answer "
    "isn't in the listings, say so and suggest adjusting the filters. Do not "
    "invent listings that are not in the context."
)

NO_LISTINGS_MESSAGE = (
    "There are no listings on the map right now. Try widening your filters or "
    "clearing the skill search, then ask again."
)


def _listing_line(listing: Dict[str, Any]) -> str:
    """One compact context line describing a listing for the prompt."""
    title = listing.get("job_title") or "Untitled role"
    company = listing.get("company_name") or "Unknown company"
    location = listing.get("location_text") or "?"
    seniority = listing.get("seniority_level") or ""
    industry = listing.get("industry") or ""
    employment = listing.get("employment_type") or ""
    skills = listing.get("required_skills_text") or ""
    attrs = ", ".join(a for a in (seniority, employment, industry) if a)
    line = f"- {title} @ {company} ({location})"
    if attrs:
        line += f" [{attrs}]"
    if skills:
        line += f" — skills: {skills}"
    return line


def build_listings_context(listings: List[Dict[str, Any]]) -> str:
    """Render the visible listings into a bounded text block for the prompt."""
    subset = (listings or [])[:MAX_CONTEXT_LISTINGS]
    lines = [_listing_line(row) for row in subset]
    header = f"There are {len(listings or [])} listings currently shown on the map"
    if len(listings or []) > MAX_CONTEXT_LISTINGS:
        header += f" (showing the first {MAX_CONTEXT_LISTINGS})"
    return header + ":\n" + "\n".join(lines)


def _query_chat_llm(question: str, listings_context: str) -> str:
    """Call the Foundation Model with the grounded prompt and return its text.

    Mirrors ``cv_parser`` by using the OpenAI-compatible client, which accepts
    plain-dict messages and returns a standard response object.
    """
    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient()
    openai_client = client.serving_endpoints.get_open_ai_client()
    response = openai_client.chat.completions.create(
        model=LLM_ENDPOINT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{listings_context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


def answer_question(
    question: str,
    listings: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Answer a user question grounded on the currently-visible listings.

    Args:
        question: The user's free-text question.
        listings: The listing dicts currently shown on the map.

    Returns:
        The assistant's answer, or a friendly message when there are no
        listings to ground on / the model call fails (so the chat never shows
        a raw traceback to the user).
    """
    if not question or not question.strip():
        return "Please type a question about the jobs on the map."
    if not listings:
        return NO_LISTINGS_MESSAGE

    listings_context = build_listings_context(listings)
    try:
        return _query_chat_llm(question.strip(), listings_context)
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        return (
            "I couldn't reach the language model just now "
            f"({type(exc).__name__}). Please try again in a moment. "
            "If this persists, the app may need to run as a Databricks App "
            "with the Foundation Model endpoint available."
        )
