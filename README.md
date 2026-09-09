# Apply-Agent

An end-to-end **AI job-matching agent** on Databricks. A candidate uploads their CV,
sets a home location and commute radius, and the agent parses the CV, semantically
searches a live corpus of German job listings, filters by real commute distance,
ranks the best fits, and drafts a tailored cover letter — all in under 60 seconds,
served from a Model Serving endpoint behind a Gradio app.

The whole stack — ingestion jobs, the Vector Search endpoint, Unity Catalog function
tools, and the app — is provisioned declaratively from a **Databricks Asset Bundle**
(`databricks bundle deploy`). Nothing is clicked together by hand.

---

## What the app does

From the candidate's point of view (`src/app/main.py`, a tabbed Gradio Blocks app):

1. **Upload a CV** (PDF or DOCX). The parser (`src/agent/cv_parser.py`) extracts
   structured fields — skills, years of experience, education history, job-title
   history, qualifications summary — via a Foundation Model (Llama 3.3 70B) with a
   strict JSON schema, bounded field sizes, a 60 s timeout, and retry/backoff.
2. **Set location + commute radius.** The home location (city or postal code) is
   resolved to coordinates against an **offline geocode lookup** table — no outbound
   network calls (`src/agent/location_resolver.py`).
3. **Get ranked matches.** The `MatchingAgent` calls the retrieval tools, computes
   great-circle commute distance to each listing, drops anything outside the radius,
   and returns the top candidates with match explanations.
4. **Draft an application.** For any match, the agent generates a tailored cover
   letter grounded in both the CV and that specific listing.

## How the data is ingested — a medallion pipeline

The listings corpus is built through Bronze → Silver → Gold Delta tables in Unity
Catalog, one ordered notebook per stage (`notebooks/02`–`07`):

**Bronze — ingest (`02_ingest_listings.py`).**
Fetches live postings from the **German Federal Employment Agency (Bundesagentur für
Arbeit) Jobsuche API**, paginating within the API's real constraints (page size ≤ 500,
10 000-record window). If the API is unreachable it falls back to a bundled dataset.
Records are validated, batched, and **MERGE-upserted** into `bronze.job_listings`
keyed on `source_url`, preserving the original `listing_id` on updates. Bad rows go to
`bronze.ingestion_errors`; a checkpoint is written after each batch so a restarted run
resumes cleanly.

**Silver — enrich (`03_enrich_listings.py`).**
Reads `unenriched` Bronze rows and:
- resolves `location_text` → lat/lon via an **offline LEFT JOIN** to `ops.geocode_lookup`,
- derives five structured attributes (`required_skills`, `seniority_level`,
  `employment_type`, `industry`, `company_size_band`) from the free-text description
  using a Foundation Model,
- **adapts LLM strategy to batch size**: server-side `ai_query` SQL for large batches
  (> 50 records), per-record SDK calls otherwise,
- builds and chunks an `embedding_text` field,
- runs an explicit enrichment **state machine**, checkpointing after every batch.

Output lands in `silver.enriched_listings` and `silver.enriched_listings_chunks`.

**Index — embed (`04_sync_vector_index.py`).**
Creates a **Mosaic AI Vector Search** endpoint and a Delta Sync index over the chunks,
embedding `embedding_text` with `databricks-gte-large-en`. Creation and sync are
idempotent; a failed sync retains the last good index rather than clearing it.

**Gold — expose retrieval as tools (`06_create_uc_functions.py`).**
Four **Unity Catalog functions** become the agent's tools:
- `search_listings` — semantic search over the vector index (capped at 200),
- `compute_commute_distance` — haversine great-circle distance in km,
- `get_user_profile` — reads the candidate profile,
- `draft_application` — generates the cover letter via Foundation Model APIs.

**Serve — deploy (`07_register_deploy_agent.py`).**
The `MatchingAgent` (`ResponsesAgent` interface) is logged with MLflow and deployed to
a Model Serving endpoint that the Gradio app calls.

```
Jobsuche API ─▶ bronze.job_listings ─▶ silver.enriched_listings(_chunks)
   (fallback)        (MERGE upsert)        (geo-join + LLM enrich + chunk)
                                                   │
                                                   ▼
                                        Vector Search index (gte-large-en)
                                                   │
     CV ─▶ parse ─▶ profile ──▶ MatchingAgent ◀── UC function tools
                                     │           (search · distance · profile · draft)
                                     ▼
                        ranked matches + tailored cover letter
```

## Engineering details worth calling out

- **Agentic tool use** over Unity Catalog functions, so retrieval, distance, and
  drafting are governed, reusable SQL/Python objects — not ad-hoc code in the agent.
- **Resilience**: MERGE-upsert idempotency, per-batch checkpoint/resume, retry with
  backoff, error side-tables, and idempotent (`CREATE OR REPLACE`) index/function setup.
- **Latency budget**: a 60 s cap enforced on both CV parsing and matching via
  `ThreadPoolExecutor` timeouts.
- **Privacy/offline geocoding**: location resolution never leaves the workspace.
- **Declarative infra**: jobs, app, vector search, and grants live in
  `databricks.yml` + `resources/*.yml`.

## Layout

```
notebooks/   Ordered pipeline (00 setup → 02 ingest → 03 enrich → 04 index → 06 tools → 07 deploy)
src/agent/   Matching agent, CV parser, location resolver
src/app/     Gradio frontend (job-agent-app)
src/models/  Dataclasses & Delta table schemas
src/utils/   Validation, retry/backoff, haversine, checkpoints, deployment gate
resources/   Bundle resource definitions (jobs, app, vector search, grants)
tests/       unit · property (Hypothesis) · integration (live Databricks)
data/        Bundled fallback listings + geocode lookup
```

## Deploy

Requires the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/) with a
configured workspace profile.

```bash
databricks bundle validate       # check the bundle
databricks bundle deploy         # provision jobs, app, vector search (dev target)
```

The `dev` target runs in development mode (source-linked, per-developer isolation),
so repeated deploys are safe.

## Testing

Three tiers. Put the repo root on `PYTHONPATH` when running locally:

```bash
pip install -e .                 # or: pip install -r requirements.txt
PYTHONPATH=. pytest              # unit + property; integration auto-skips locally
```

- **Unit** (`tests/unit/`) — CV parser, matching agent, pipelines, location logic.
- **Property** (`tests/property/`) — [Hypothesis](https://hypothesis.works/)-driven
  invariants: commute filtering, cover-letter generation, retry logic,
  checkpoint/resume, MERGE upserts, chunking, and more.
- **Integration** (`tests/integration/`) — exercise live Vector Search, Model Serving,
  the SQL warehouse, and Unity Catalog. A shared environment gate **auto-skips** them
  off-workspace, so they never fail on a laptop:

```bash
PYTHONPATH=. pytest -m integration     # skipped locally, runs in-workspace
```

> Targets **Python 3.10–3.12**. Some pinned runtime deps (e.g. `gradio==5.49.1`) do
> not build on newer interpreters; use a 3.10–3.12 virtualenv for a clean run.
