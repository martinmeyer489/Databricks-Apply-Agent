"""CV parsing module.

Extracts structured profile fields (skills, years of experience, education
history, job title history, qualifications summary) from an uploaded CV
file (PDF or DOCX) using Foundation Model APIs.

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
"""

from __future__ import annotations

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict

from src.utils.cv_bounds import enforce_cv_bounds
from src.utils.cv_warnings import get_unresolved_fields
from src.utils.input_validation import validate_cv_file
from src.utils.retry import retry_with_backoff

CV_PARSING_TIMEOUT_SECONDS = 60

LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

EXTRACTION_PROMPT_TEMPLATE = """Extract the following from this CV as JSON:
- skills: array of strings (max 50)
- years_of_experience: integer 0-99
- education_history: array of {{institution, degree, field, year}} (max 20)
- job_title_history: array of {{title, company, start_year, end_year}} (max 30)
- qualifications_summary: string (max 2000 chars)

Return ONLY a JSON object with exactly these keys. If a field cannot be
determined from the CV text, omit that key from the JSON object.

CV text:
{cv_text}
"""


def _determine_file_format(file_path: str) -> str:
    """Determine the CV file format ("PDF" or "DOCX") from its extension."""
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    if extension == ".pdf":
        return "PDF"
    if extension == ".docx":
        return "DOCX"
    # Unknown extension: return it uppercase (without the leading dot) so
    # `validate_cv_file` rejects it with the standard error message.
    return extension.lstrip(".").upper() or "UNKNOWN"


def _extract_text_from_pdf(file_path: str) -> str:
    """Extract concatenated text from every page of a PDF file."""
    import PyPDF2

    with open(file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)


def _extract_text_from_docx(file_path: str) -> str:
    """Extract concatenated text from every paragraph of a DOCX file."""
    from docx import Document

    document = Document(file_path)
    paragraphs_text = [paragraph.text for paragraph in document.paragraphs]
    return "\n".join(paragraphs_text)


def _extract_text(file_path: str, file_format: str) -> str:
    """Extract raw text from a CV file based on its determined format."""
    if file_format == "PDF":
        return _extract_text_from_pdf(file_path)
    if file_format == "DOCX":
        return _extract_text_from_docx(file_path)
    raise ValueError(f"Unsupported CV file format for text extraction: {file_format}")


@retry_with_backoff(max_attempts=3, backoff_base=2, retryable_codes=(429,))
def _query_llm_endpoint(cv_text: str) -> str:
    """Call Foundation Model APIs with the structured extraction prompt.

    Uses the OpenAI-compatible client (`serving_endpoints.get_open_ai_client()`)
    which accepts plain-dict messages and returns a standard response object;
    the SDK's native `serving_endpoints.query(messages=[{...}])` path raises
    ``'dict' object has no attribute 'as_dict'`` on recent SDK versions.

    Wrapped in `retry_with_backoff` so a rate-limited (HTTP 429) response
    is retried up to 3 times with exponential backoff starting at 2
    seconds (Req 11 AC10).
    """
    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient()
    openai_client = client.serving_endpoints.get_open_ai_client()
    response = openai_client.chat.completions.create(
        model=LLM_ENDPOINT,
        messages=[
            {
                "role": "user",
                "content": EXTRACTION_PROMPT_TEMPLATE.format(cv_text=cv_text),
            }
        ],
    )
    return response.choices[0].message.content


def _call_llm_with_timeout(cv_text: str, timeout_seconds: int = CV_PARSING_TIMEOUT_SECONDS) -> str:
    """Call the Foundation Model API bounded by a timeout.

    Raises `TimeoutError` if the call exceeds `timeout_seconds`, or the
    original exception (wrapped in a descriptive `RuntimeError`) if the
    call itself fails (Req 5.7).
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future: Future = executor.submit(_query_llm_endpoint, cv_text)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as exc:
            raise TimeoutError(
                f"CV parsing timed out after {timeout_seconds}s"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise RuntimeError(f"CV parsing failed: {exc}") from exc


def _parse_llm_json(raw_response: str) -> Dict[str, Any]:
    """Parse the LLM's raw response string into a dict of extracted fields.

    Tolerant of the markdown-fenced (```json ... ```) and prose-wrapped
    outputs that chat models commonly emit: strips a code fence if present,
    then falls back to extracting the first ``{...}`` object. Raises a
    descriptive `RuntimeError` only if no JSON object can be recovered
    (Req 5.7).
    """
    if not raw_response:
        raise RuntimeError("CV parsing failed: empty response from Foundation Model APIs")

    import re

    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_]*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()

    def _try(candidate: str):
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None

    parsed = _try(text)
    if parsed is None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = _try(match.group(0))

    if parsed is None:
        raise RuntimeError(
            f"CV parsing failed: invalid JSON response ({raw_response!r:.200})"
        )
    return parsed


def parse_cv(file_path: str) -> dict:
    """Parse an uploaded CV file into structured profile fields.

    Validates the file (format PDF/DOCX, >0 bytes, <=5 MB), extracts its
    raw text, calls Foundation Model APIs with a structured extraction
    prompt, enforces field bounds, and computes the list of unresolved
    fields for partial parses.

    Args:
        file_path: Path to the stored CV file (in a Volume, or locally
            for testing).

    Returns:
        A dict with keys `skills`, `years_of_experience`,
        `education_history`, `job_title_history`,
        `qualifications_summary`, and `unresolved_fields`.

    Raises:
        ValueError: If the file fails format/size validation (Req 5.1,
            5.5).
        TimeoutError: If CV parsing exceeds 60 seconds (Req 5.7).
        RuntimeError: If CV parsing raises any other error (Req 5.7).
    """
    file_format = _determine_file_format(file_path)
    file_size = os.path.getsize(file_path)

    is_valid, validation_message = validate_cv_file(file_format, file_size)
    if not is_valid:
        raise ValueError(validation_message)

    cv_text = _extract_text(file_path, file_format)

    raw_response = _call_llm_with_timeout(cv_text, timeout_seconds=CV_PARSING_TIMEOUT_SECONDS)
    parsed_fields = _parse_llm_json(raw_response)

    bounded_fields = enforce_cv_bounds(parsed_fields)
    unresolved_fields = get_unresolved_fields(bounded_fields)

    return {
        "skills": bounded_fields.get("skills"),
        "years_of_experience": bounded_fields.get("years_of_experience"),
        "education_history": bounded_fields.get("education_history"),
        "job_title_history": bounded_fields.get("job_title_history"),
        "qualifications_summary": bounded_fields.get("qualifications_summary"),
        "unresolved_fields": unresolved_fields,
    }
