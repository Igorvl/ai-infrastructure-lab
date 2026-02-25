AI Design Workspace: Enterprise MLOps Architecture

🧬 AI Design Infrastructure Lab (Project DNA)
Enterprise MLOps Architecture for Multimodal R&D

🧠 Intelligent LLM Gateway (The Brain)
The heart of the system is a highly available, asynchronous API gateway built using FastAPI and LiteLLM. It implements a Tiered Reasoning Strategy, balancing state-of-the-art cognitive capabilities with inference efficiency.

🛠 Key Engineering Features:
Asynchronous Processing: Full transition to acompletion and async/await event loops for non-blocking handling of heavy Vision payloads.

Custom SSE Streaming: Implemented a custom Server-Sent Events (SSE) generator. This addressed complex Pydantic object serialization issues (AttributeError) and enabled real-time token streaming to Open WebUI.

Vision-Aware Routing: The gateway automatically detects multimodal content (images) within the request payload and dynamically reconfigures the fallback chain to prioritize Vision-capable models.

Region-Specific Split Tunneling: Network traffic to Google AI Studio (v1beta API) is isolated within Docker and routed via VLESS/Shadowsocks to bypass regional restrictions.

🚀 Routing & Failover Logic:
Primary Reasoning (Tier 1): Gemini 3 Flash (Google API v1beta).

Role: Core reasoning engine for UI/UX analysis and architectural planning.

Multimodal Fallback: Qwen2-VL (via SiliconFlow).

Role: Redundant vision channel for screenshot analysis and UI element detection if primary quotas are exhausted.

Secondary Reasoning (Tier 2): DeepSeek-V3.

Role: Deep code analysis and logical refactoring.

DevOps Workhorse (Tier 3): Qwen-3-Coder (480B).

Role: System log parsing, Bash scripting, and Docker configuration generation.

🏗 Infrastructure Stack
Virtualization: VMware ESXi 7.0 (On-Premise R&D Lab).

Compute: Intel Xeon E5-2680 v3, 64GB RAM.

Storage: 90TB+ LVM storage with hot-resize capabilities.

Networking: Split tunneling, Docker-native routing, isolated VLANs.

Core Stack: Docker, Python 3.12, FastAPI, LiteLLM, Qdrant, MongoDB/Postgres, MinIO.
