AI Design Workspace: Enterprise MLOps Architecture

## Project Overview
A multi-layered RAG system and workspace designed for AI designers working with heavy graphics (Nano Banana) and LLMs. This project addresses multi-account management, strict context limits, and secure long-term storage of "Chain of Thought" (CoT) reasoning.

## Architecture/ LLM Gateway

### 🧠 Cognitive Reasoning Grid (LLM Gateway)

The system implements a **Tiered Reasoning Strategy**, optimizing the balance between model cognitive capabilities and inference costs.

**Routing Logic & Roles:**

1.  **Primary Reasoning (Tier 1):** `Gemini 3 Pro Preview` (Google).
    * *Config:* `thinking_level: high`.
    * *Role:* Handles complex algorithmic logic, architectural planning, and RAG synthesis. Acts as the flagship model with an extended context window (65k output).
2.  **Secondary Reasoning (Tier 2):** `DeepSeek-V3`.
    * *Role:* Specialized in code analysis, refactoring, and unit test generation. Serves as a high-speed, cost-effective alternative for medium-complexity tasks.
3.  **Infrastructure Expert (Tier 3):** `Qwen-2.5-72B` (via SiliconFlow).
    * *Role:* The DevOps "workhorse." Optimized for Bash scripting, Dockerfile management, and system log parsing.
4.  **Legacy Fallback (Tier 4):** `GLM-5` (Zhipu AI).
    * *Config:* `enable_thinking: true`.
    * *Role:* Backup channel with Chain of Thought (CoT) enabled. Ensures system fault tolerance and redundancy if primary providers fail.

**Technologies:** LiteLLM, Custom Circuit Breaker (Python/FastAPI), Docker.

## Infrastructure & Disaster Recovery
Deployed via ESXi + Docker. Includes an automated backup system (Restic/Borg) with regular restore.sh script testing within an isolated VLAN.

## Quick Start
1. cp deploy/.env.example deploy/.env (Fill in your API keys)
2. docker-compose -f deploy/docker-compose.yml up -d
