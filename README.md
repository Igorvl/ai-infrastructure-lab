AI Design Workspace: Enterprise MLOps Architecture

## Project Overview
A multi-layered RAG system and workspace designed for AI designers working with heavy graphics (Nano Banana) and LLMs. This project addresses multi-account management, strict context limits, and secure long-term storage of "Chain of Thought" (CoT) reasoning.

## Architecture
The project is divided into isolated circuits:
* **Frontend: Open WebUI (macOS PWA) for a native user experience.
* **Orchestration: Dify (RAG pipelines, agents).
* **Storage: Qdrant (vector database), MongoDB (metadata), MinIO (S3-compatible storage for images).
* **Routing Layer: A custom proxy based on LiteLLM. It implements the "Circuit Breaker / Fallback" pattern: if Claude fails, traffic is automatically routed to Gemini 3 Pro, truncating dialogue history to a strict limit (maxContextTokens: 32000) defined in the antigravity.json config.
* **Observability: Prometheus + Grafana + cAdvisor for monitoring token usage, latency, and resource allocation.

## Infrastructure & Disaster Recovery
Deployed via ESXi + Docker. Includes an automated backup system (Restic/Borg) with regular restore.sh script testing within an isolated VLAN.

## Quick Start
1. cp deploy/.env.example deploy/.env (Fill in your API keys)
2. docker-compose -f deploy/docker-compose.yml up -d
