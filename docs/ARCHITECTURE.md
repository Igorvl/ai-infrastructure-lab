

AI Design Infrastructure Lab: Architecture Manifest


1. Concept & Business Objective

This project is an Enterprise-grade AI pipeline management system tailored for a high-end designer (macOS/iMac client) and administered by an MLOps Engineer (ESXi + Docker backend).

Core Challenge: Standard AI chat interfaces fail to maintain long-term context, visual logic (Chain of Thought), style matrices, and source assets during multi-month design projects, especially when juggling multiple accounts and hard token limits.

Objective: To architect a self-hosted, scalable ecosystem with "infinite memory" (Long-Term Memory RAG), automated LLM rate-limit handling, and a seamless UI, completely abstracting the underlying infrastructure complexity from the end-user.


2. Generative Stack Specifics (Nano Banana + Gemini 3 Pro)

The architecture is tightly coupled with state management for the Nano Banana model (accessed via Gemini 3 Pro). The database layer is strictly required to log:

Exact prompts and generation seeds.

Style matrices for multi-image-to-image composition and style transfer.

Source masks for inpainting/editing workflows.

Hardcoded typography constraints to ensure high-fidelity text rendering across iterations.


3. System Topology (Multi-tier RAG Architecture)

The system is decoupled into 5 isolated deployment tiers within a Dockerized ESXi environment:

Tier 1: Frontend (User Experience)
Component: Open WebUI.

Role: PWA for the macOS client. Provides a native, highly polished interface.

Features: Supports project workspaces, tagging, and drag-and-drop for multimodal inputs (references). Acts purely as a stateless presentation layer (dumb terminal).

Tier 2: Orchestration & Logic (RAG Engine)
Component: Dify (API + Worker nodes).

Role: The brain of the system. Ingests requests from the frontend.

Features: Orchestrates RAG pipelines. For historical context queries, Dify retrieves past decisions from the Vector DB, routes them through a Context Compression node (LLM Summarization via a lightweight model like Flash), and injects the consolidated payload into the final system prompt.

Tier 3: Storage Network (State & Persistence)
Qdrant: Vector DB. Stores embeddings of all historical sessions enabling semantic search (e.g., "Retrieve the rationale for rejecting gradients in April").

MongoDB / PostgreSQL: Document/Relational DB. Stores raw chat logs, session IDs, project metadata, and precise Nano Banana configuration payloads.

MinIO: S3-compatible Object Storage. Mission-critical for multimodal workflows. Hosts heavy references and final artifacts. Models receive presigned URLs instead of raw Base64 strings, drastically reducing API latency from ~15s down to 2-3s.

Tier 4: LLM Gateway & Routing
Component: LiteLLM Proxy (Custom Python Router).

Role: Traffic shaping and failover management.

Business Logic (antigravity.json policy): 1.  Default traffic is routed to Anthropic (Claude).
2.  Upon hitting a 429 Too Many Requests limit, the Circuit Breaker pattern is triggered.
3.  The router intercepts the exception and reads the fallback policy from routing/config/antigravity.json.
4.  It applies the hardcoded constraint: maxContextTokens: 32000.
5.  Dynamically truncates the chat history array (tail clipping) to strictly fit the 32k token window.
6.  Seamlessly reroutes the payload to the Google API (Gemini 3 Pro) ensuring zero downtime for the user.

Tier 5: Observability Stack
Prometheus: Scrapes metrics across all nodes.

Grafana: Visualization. Key dashboards track token spend, API latency, Fallback trigger frequency (Claude rate limits), and memory utilization.

cAdvisor: Container resource analytics (monitors the memory footprint of Qdrant and Dify on ESXi).

Loki + Promtail: Centralized log aggregation for streamlined debugging.


4. Security & Disaster Recovery (DR) Strategy

To ensure data integrity and IP protection, the environment implements robust, automated backup pipelines:

Secrets Management: All API keys are strictly injected via .env (excluded via .gitignore); only .env.example is committed to the repo.

DB Backups (Postgres/Mongo/Qdrant): Scheduled cron dumps and utilization of native Snapshots APIs.

Heavy Assets (MinIO): Incremental backups utilizing Restic / Borg for data deduplication to optimize ESXi disk space.

Disaster Recovery Protocol: A restore.sh bash script is provisioned for single-command cluster recovery. DR workflows are tested monthly in an isolated VLAN, featuring automated API health checks and Telegram alert notifications upon successful restoration.


5. Repository Structure (ai-infrastructure-lab)

Plaintext
├── .github/                  # CI/CD pipelines (planned)
├── deploy/                   # Infrastructure as Code (IaC)
│   ├── docker-compose.yml    # Main orchestrator for all tiers
│   ├── .env.example          # Template for secrets and environment variables
│   └── scripts/              # Ops bash scripts (backup.sh / restore.sh)
├── routing/                  # LLM Gateway
│   ├── Dockerfile            # LiteLLM Proxy build context
│   ├── router.py             # Custom Python logic for rate-limit interception
│   └── config/
│       └── antigravity.json  # Fallback policy (maxContextTokens: 32000)
├── monitoring/               # Observability configs
│   ├── prometheus.yml
│   └── grafana-dashboards/
├── docs/                     # Architecture documentation
├── .gitignore                # Prevents credentials & data leaks
└── README.md                 # Project storefront


MLOps Summary: This architecture covers the full lifecycle of LLM operations in a Production-like environment: from intelligent context window management and dynamic API routing, to comprehensive observability and disaster recovery, all while delivering a frictionless UX for the end-user.

