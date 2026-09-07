-- Unity Catalog grants for the job-agent-app Databricks App service principal.
--
-- The app's service principal is created only when the app is first deployed,
-- so these grants cannot be applied at catalog-creation time (chicken-and-egg)
-- and are therefore applied after `databricks bundle run job_agent_app`.
--
-- Find the app SP client id with:
--   databricks apps get job-agent-app -o json | jq -r .service_principal_client_id
-- then substitute it for :app_sp below (or run these via the SQL editor /
-- `databricks api post /api/2.0/sql/statements`).
--
-- These grants let the app (and the served agent, whose tools run on the same
-- warehouse) read the Gold/Silver/Ops tables and execute the gold.* UC
-- functions (search_listings, get_user_profile, compute_commute_distance,
-- draft_application) and resolve locations against ops.geocode_lookup.

-- Replace with the app SP client id (e.g. 8cf42ba0-5d80-48e0-b6bb-9db54a7e99a9)
-- :app_sp

GRANT USE CATALOG ON CATALOG job_agent TO `:app_sp`;
GRANT USE SCHEMA  ON SCHEMA  job_agent.ops    TO `:app_sp`;
GRANT USE SCHEMA  ON SCHEMA  job_agent.gold   TO `:app_sp`;
GRANT USE SCHEMA  ON SCHEMA  job_agent.silver TO `:app_sp`;
GRANT SELECT      ON SCHEMA  job_agent.ops    TO `:app_sp`;
GRANT SELECT      ON SCHEMA  job_agent.gold   TO `:app_sp`;
GRANT SELECT      ON SCHEMA  job_agent.silver TO `:app_sp`;
GRANT EXECUTE     ON SCHEMA  job_agent.gold   TO `:app_sp`;
