# Apply-Agent

An end-to-end **job-matching agent** built on Databricks Free Edition. Upload a CV,
set a location and commute radius, and the agent ranks the best-fitting job listings
and drafts a tailored application — powered by Unity Catalog functions, Mosaic AI
Vector Search, and Foundation Model APIs.

## What it does

The pipeline turns raw job listings and a candidate's CV into ranked matches:

1. **Ingest & enrich** listings into Unity Catalog Delta tables (`notebooks/02`–`03`).
2. **Embed & index** listings in a Mosaic AI Vector Search index (`notebooks/04`).
3. **Expose retrieval as tools** via Unity Catalog functions — semantic search,
   commute-distance computation, profile lookup, application drafting
   (`notebooks/06`, `src/agent/matching_agent.py`).
4. **Match**: the `MatchingAgent` (`ResponsesAgent` interface) parses the CV, calls
   the UC function tools, and returns the top candidates within a 60-second budget.
5. **Serve**: the agent is registered and deployed to a Model Serving endpoint
   (`notebooks/07`), fronted by a Gradio **Databricks App** (`src/app/main.py`).

Everything is provisioned declaratively through a **Databricks Asset Bundle**
(`databricks.yml` + `resources/*.yml`) — jobs, the Vector Search endpoint, and the
app are all created via `databricks bundle deploy`, nothing by hand in the UI.

## Layout

```
notebooks/   Ordered pipeline steps (00 setup → 07 deploy → 08 workflow)
src/agent/   Matching agent, CV parser, location resolver
src/app/     Gradio frontend (job-agent-app)
src/models/  Dataclasses & schemas
src/utils/   Validation, retry, haversine, checkpoints, deployment gate
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

The suite has three tiers. Sources are under `src/`, so put the repo root on
`PYTHONPATH` when running locally:

```bash
pip install -e .                 # or: pip install -r requirements.txt
PYTHONPATH=. pytest              # run everything
```

**Unit** (`tests/unit/`) — fast, isolated logic checks for the CV parser,
matching agent, pipelines, and location resolution.

**Property** (`tests/property/`) — [Hypothesis](https://hypothesis.works/)-driven
invariant tests (commute filtering, cover-letter generation, retry logic,
checkpoint/resume, upserts, and more):

```bash
PYTHONPATH=. pytest tests/property
```

**Integration** (`tests/integration/`) — exercise live workspace resources
(Vector Search, Model Serving, SQL warehouse, Unity Catalog). They **auto-skip**
outside Databricks via a shared environment gate, so they never fail on a laptop.
They run when `DATABRICKS_HOST` is set or inside a Databricks cluster/job:

```bash
PYTHONPATH=. pytest -m integration     # skipped locally, run in-workspace
```

> Note: the project targets **Python 3.10+**. Some pinned runtime dependencies
> (e.g. `gradio==5.49.1`) may not build on very new interpreters; use a 3.10–3.12
> virtual environment for a clean run.
