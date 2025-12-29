# MasterScraper — A→Z Build & Run Plan (single‑source reference)

Last updated: 2025-11-16

Purpose
-------
This document is the single canonical plan to build, secure, deploy and operate MasterScraper — a nationwide public-data scraping, enrichment and Person‑Graph system that writes structured data to SharePoint and supports analytics and downstream automation. Keep this file in source control (repo root), and publish a copy to your SharePoint site and an enterprise backup location so it survives accidental chat/PR/issue deletion.

How to use this file
- Follow the numbered A→Z steps in order for initial implementation.
- Use the checklists as gate criteria before moving to the next stage.
- Store copies in:
  - Repo: /docs/MasterScraper_A-to-Z_Plan.md (commit & tag releases)
  - SharePoint site: /sites/mountainjewelsdata/Shared Documents/Docs
  - Organizational wiki / Confluence (if available)
  - Offsite backup (encrypted archive)

High‑level summary (one line)
------------------------------
Design, build, test, secure and operate a containerized scraper system that runs on Azure Container Apps (scheduled job), uses Managed Identity to write to SharePoint Lists, and is deployed with GitHub Actions + OIDC, Terraform infra, automated tests, monitoring and strict legal/compliance controls.

A → Z step‑by‑step plan (numbered)
----------------------------------

Phase 0 — Preparation & approvals
1. Legal & policy sign‑off
   - Create a Source‑Permissions matrix: for every source (newspapers, social sites, government portals) list allowed access, ToS clauses, rate limits, copyright/copyright-holders, and retention constraints.
   - Get written legal approval for each source before scraping. Log approvals in the repo (docs/legal).
   - Decide retention & PII policy; document deletion/opt-out workflows.

2. Governance & accounts
   - Confirm accounts: Azure subscription, Microsoft 365 tenant & SharePoint site, GitHub org/repo, Cloudflare (if used), DockerHub (optional).
   - Create an IAM/RBAC plan and list of admin contacts (Azure Owner, Global Admin, GitHub repo admins).
   - Create an "ops" contact list and escalation path.

3. Design & architecture review
   - Review this plan with architects, security, legal and operations teams.
   - Approve runtime choice: Recommended = Azure Container Apps Job (server-side Python in containers).
   - Approve storage target: SharePoint Lists (as required), optionally also Cosmos/SQL for analytics.

Phase 1 — Repo and scaffolding
4. Create repository layout (if not present)
   - /src — scraper code (package)
   - /tests — unit & integration tests
   - /Dockerfile — container build
   - /infra — Terraform/Bicep
   - /.github/workflows — CI/CD workflows
   - /docs — this plan, runbooks, legal approvals
   - README.md with developer setup steps

5. Minimal scaffold & example
   - Add a simple scraper module that fetches a test page, parses and writes to a local JSON file.
   - Add a minimal Dockerfile that runs the scraper script.
   - Add unit tests for parser and a smoke integration test against a static HTML fixture.

Acceptance criteria:
- Local docker build passes.
- Unit tests run and pass locally.

Phase 2 — Secrets, identity & access model
6. Decide identity model for runtime writes to SharePoint
   Option A (recommended): Managed Identity on Container App
     - Pros: No client secrets stored in repo; integrated Azure RBAC.
     - Cons: Requires admin consent for Graph permissions to be granted to the identity or role assignment.
   Option B: AAD app registration with client credentials
     - Store SDK-auth JSON in GitHub Secrets (less preferred).

7. GitHub → Azure trust for CI/CD
   - Use GitHub OIDC + azure/login (no long-lived service principal).
   - Create a federated credential in Azure AD for GitHub Actions if needed; or use azure/login@v1 with creds secret if unavoidable.
   - Document steps for creating the federated credential (tenant admin must run once).

8. Secrets inventory
   - Create an inventory file docs/secrets-required.md listing:
     - AZURE_SUBSCRIPTION_ID (repo secret)
     - AZURE_RESOURCE_GROUP (optional)
     - ACR name & ACR login (if necessary)
     - AZURE_WEBAPP_NAME / CONTAINER_APP_NAME
     - AZURE_CREDENTIALS (only if using service principal)
     - SHAREPOINT_SITE_URL (repo secret or config)
     - (Cloudflare) CF_API_TOKEN etc.

Phase 3 — Infrastructure (IaC)
9. Write Terraform/Bicep to provision base infra
   - Resources: ACR, Container Apps Environment, Container App, Container App Job (scheduled), Managed Identity (system or user-assigned), App Insights, Log Analytics workspace, Key Vault (optional).
   - Outputs: registry login server, container app URL, managed identity principalId, resource IDs.

10. Admin steps (must be run by Tenant Admin)
   - Grant Managed Identity the required Graph permissions:
     - Preferred pattern: give the Managed Identity Azure AD app the minimal Graph role and grant admin consent for the permission (Sites.Selected or Sites.ReadWrite.All depending on need).
     - Alternative: create a service principal with delegated permissions and grant via consent.
   - Provide documented commands to run (az ad app permission grant / grant admin consent or portal steps).

11. Deploy the base infra to a staging environment
   - Terraform init/plan/apply in staging subscription/resource group.
   - Verify that the Container App environment is created and Managed Identity exists.

Acceptance criteria:
- Container App + Job exist in staging environment.
- Managed Identity principalId output is present.
- App Insights logs test entry possible.

Phase 4 — Build & CI
12. GitHub Actions — unit tests & linting
   - Workflow: .github/workflows/build-and-test.yml
     - Runs: checkout, setup-python, install deps, run unit tests, run linters, upload artifact (package or wheel).

13. GitHub Actions — build container & push to ACR using OIDC
   - Workflow: .github/workflows/build-and-push-acr.yml
     - Uses: azure/login with OIDC (no secrets)
     - Build image, tag with commit SHA and semver tags
     - Push to ACR
     - Upload image tag as workflow output / create a release note

14. GitHub Actions — deploy/update Container App job
   - Workflow: .github/workflows/deploy-containerapp.yml
     - Uses: az cli commands to update container image on the Container App and update scheduled job recurrence
     - Use environment inputs: stage=staging|prod, image:tag, run-time settings
   - Implement a reusable workflow-call pattern so other repos can call deploy.

Acceptance criteria:
- CI pipeline builds image and pushes to ACR successfully.
- CD workflow updates Container App image and restarts scheduled job.

Phase 5 — Runtime code & SharePoint integration
15. SharePoint helper module
   - Provide a Python module that:
     - Uses Managed Identity (IMDS) to request a Graph access token (MSAL or requests to IMDS endpoint)
     - Writes/upserts records into specified SharePoint lists with idempotency keys and provenance fields (source, run_id, module)
     - Handles rate limits / throttling / retries (exponential backoff)
     - Logs success/failure with structured JSON

16. Person Graph & deduplication
   - Implement a record normalizer, deduplicator and a Person Graph joiner:
     - Use hashed keys for fuzzy matching (name+DOB+address fingerprint)
     - Keep confidence scores and provenance for edges added
     - Unit tests for linking logic

17. Provenance & audit
   - For each SharePoint entry include:
     - source_url, scraped_at, run_id, module_name, confidence_score
   - Make sure SharePoint writes are idempotent (use a stable key to patch existing items rather than insert duplicates).

Acceptance criteria:
- Managed Identity token retrieval works from a running container.
- A sample upsert to a SharePoint list succeeds (staging).

Phase 6 — Scheduling & scaling
18. Job scheduling strategy
   - Do NOT run every source full-crawl every 10 minutes. Instead:
     - Per-source cadence config (config/sources.yaml) — e.g., newspapers: daily historical; announcements: every 10 minutes; public social: hourly.
     - Implement a job partitioning pattern: Container App Job triggers only a subset of sources per run or use multiple jobs with dedicated schedules.
     - Use a work-queue pattern (Azure Storage Queue / Service Bus) if you need parallel workers.

19. Concurrency & limits
   - Use concurrency controls and polite crawling:
     - Respect robots.txt, per-host rate limits, exponential backoff on 429/5xx
     - Use proxy/pool or cloud scraping gateway if needed (rotate IPs)
     - Implement request throttling within the scraper

Phase 7 — Testing & staging workflow
20. Staging & smoke tests
   - After deploy to staging:
     - Run end‑to‑end smoke test that scrapes a small public page, validates data shape, upserts to a staging SharePoint list, and asserts presence.
     - Have tests run automatically on PR for code changes.

21. Canary & production rollout
   - Deploy to canary/prod only after tests pass in staging and after a hold/approval step.
   - Use GitHub Environments with required reviewers for production deploy workflow.

Phase 8 — Observability & operations
22. Logging & telemetry
   - Integrate App Insights / Log Analytics for:
     - Run-level metrics (duration, items scraped, success %, errors)
     - Per-module metrics and histograms
     - Traces to link request → parse → upsert

23. Alerts & runbook
   - Define alerts:
     - Job fails to start
     - Error rate > threshold
     - SharePoint write failures > threshold
     - Too many duplicates or sudden spike/drop in items scraped
   - Create runbooks for each alert describing immediate remediation steps and escalation.

24. Backup & retention
   - Store all raw scraped payloads (or metadata) in a blob store for recovery and debugging.
   - Define retention rules (30/90/365 days as appropriate) and deletion workflows to comply with data minimization.

Phase 9 — Security, compliance & ops hardening
25. Least privilege & secrets
   - Verify managed identity permissions are minimal.
   - If any secrets remain (third-party API tokens), store in Key Vault and grant Container App access.

26. Access control & audits
   - Lock down who can trigger production runs (GitHub protected branches, required reviewers).
   - Enable auditing in Azure and Microsoft 365 for Graph writes access.

27. Pen-test & compliance review
   - Schedule a security review and penetration test for the scraper infra and the SharePoint integration.

Phase 10 — Automation & developer ergonomics
28. Developer local flow
   - Provide a docker-compose dev environment to run the scraper locally and mock SharePoint (or use a staging SharePoint site).
   - Provide make targets:
     - make test
     - make lint
     - make build
     - make run-local

29. Dependency management & CI hygiene
   - Use pinned dependencies in requirements.txt or Poetry lockfiles.
   - Add Dependabot/renovate to open dependency PRs and run tests automatically.

Phase 11 — Long‑term scaling & architecture evolution
30. Analytics layer
   - Replicate SharePoint data to a data lake / CosmosDB / SQL for analytics and heavy querying.
   - Build incremental ETL jobs that read SharePoint Lists and populate analytics store.

31. Person Graph scale & offline compute
   - If Person Graph grows large, move to a graph database or an optimized linking engine with incremental reindexing (e.g., Neo4j, CosmosDB Gremlin, or a columnar store for blocking/indexing).

32. Multi‑tenant & multi‑site
   - If supporting multiple tenants/sites, introduce tenant isolation: per-tenant managed identities or separate resource groups and secrets.

Phase 12 — Handoff & documentation
33. Final docs & runbook
   - Finalize:
     - Oncall runbook
     - Architecture diagrams
     - Admin steps (granting Graph permissions, rotating tokens)
     - Recovery procedures

34. Training
   - Run a handoff session with operations team and record it (store video link).
   - Verify operations team can run a restore and handle alerts.

35. Continuous improvement
   - Schedule quarterly reviews for legal compliance, performance tuning, and data quality audits.

Appendices
----------

A. Quick checklist — pre‑deploy (must be green)
- Legal sign-off for each source (Y/N)
- Azure admin approved Graph permissions (Y/N)
- Repo has workflows for build/push/deploy (Y/N)
- Managed Identity exists and principalId is known (Y/N)
- Staging SharePoint lists exist and are writable (Y/N)
- Observability pipeline (App Insights) created (Y/N)
- Secrets in vault or repo created (list) (Y/N)

B. Minimal commands & snippets (examples)
- Login interactively and set subscription:
  az login --use-device-code
  az account set --subscription <SUBSCRIPTION_ID>

- Create federated credential (portal recommended). Example az CLI steps documented in infra docs.

- Build & push image locally (example):
  docker build -t myacr.azurecr.io/master-scraper:sha-$(git rev-parse --short HEAD) .
  az acr login --name myacr
  docker push myacr.azurecr.io/master-scraper:sha-...

- Update Container App image (az CLI):
  az containerapp update --name <containerapp> --resource-group <rg> --image myacr.azurecr.io/master-scraper:sha-... 

- Retrieve MSI token inside container (Python example):
  import requests, os
  endpoint = "http://169.254.169.254/metadata/identity/oauth2/token"
  params = {"api-version": "2018-02-01", "resource": "https://graph.microsoft.com"}
  headers = {"Metadata": "true"}
  r = requests.get(endpoint, params=params, headers=headers)
  token = r.json()["access_token"]

C. SharePoint upsert pattern (pseudo)
- Key = hash(source_url + normalized_name + event_date)
- If item with key exists: PATCH item with merged fields and appended provenance
- Else: POST new item with provenance fields
- Always keep raw_payload in a "RawPayload" field or blob storage reference

D. Sources & ToS checklist (template)
- Source name
- URL / API endpoint
- ToS link snapshot (save copy)
- Allowed usage? (Y/N)
- Rate limits
- Auth needed?
- Legal sign-off (initials/date)

E. Where to store this document
- primary: repo /docs/MasterScraper_A-to-Z_Plan.md
- backup: SharePoint site library + personal encrypted archive + company wiki

Acceptance criteria for project completion
- System runs scheduled scraping in production (jobs executed on schedule)
- Data appears in production SharePoint Lists with correct provenance and idempotency
- CI builds images, pushes to ACR, deploys to Container App jobs automatically
- Alerts fire and operators can remediate following runbooks
- Legal/compliance approvals are completed and documented

If this conversation is deleted — recovery steps
1. Retrieve this file from the repo (/docs). If the repo is lost, retrieve from SharePoint copy or the offsite backup mentioned earlier.
2. Use the Terraform code in /infra to re-provision resources.
3. Reset GitHub secrets and federated credentials per the documented admin steps in /docs/admin-steps.md.

Next recommended immediate actions (first 7 days)
1. Get legal sign-off for top 10 highest‑risk sources.
2. Provision a staging Azure resource group and run the Terraform module to create ACR + Container App skeleton.
3. Implement the SharePoint helper module and test it from a local container using a staging managed identity or app credentials.
4. Add the GitHub Actions build-and-push-acr and deploy workflows and verify end‑to‑end in staging.
5. Set up App Insights and 3 initial alerts (job failure, high error rate, SharePoint write failure).
6. Run a full smoke test and document results in /docs/deployment-log.md.

Contact / ownership
- Owner: colin-coates
- Primary operator: [fill name / email]
- Azure admin: [fill name]
- Legal contact: [fill name]

End of file.