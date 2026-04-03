"""
Project DNA — Database Module
Async PostgreSQL operations via asyncpg.
"""

import os
import json
import logging
import asyncpg
from typing import Optional, List, Dict, Any

logger = logging.getLogger("AI-Router.DB")


class DatabaseManager:
    """Async PostgreSQL connection pool and query methods."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.host = os.getenv("PG_HOST", "ai-postgres")
        self.port = int(os.getenv("PG_PORT", "5432"))
        self.user = os.getenv("PG_USER", "igorvl")
        self.database = os.getenv("PG_DATABASE", "project_dna")
        self.password = os.getenv("PG_PASSWORD", "")

    async def connect(self):
        """Create connection pool. Call on app startup."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host, port=self.port,
                user=self.user, database=self.database,
                password=self.password or None,
                min_size=2, max_size=10,
            )
            logger.info(f"PostgreSQL connected: {self.host}:{self.port}/{self.database}")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            self.pool = None

    async def disconnect(self):
        """Close connection pool. Call on app shutdown."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL disconnected")

    # ==================== PROJECTS ====================

    async def list_projects(self, status: str = None) -> List[dict]:
        if status:
            rows = await self.pool.fetch(
                "SELECT v.*, p.slug FROM v_project_stats v JOIN projects p ON p.id = v.id WHERE v.status = $1 ORDER BY v.name", status)
        else:
            rows = await self.pool.fetch("SELECT v.*, p.slug FROM v_project_stats v JOIN projects p ON p.id = v.id ORDER BY v.name")
        return [dict(r) for r in rows]

    async def get_project(self, slug: str) -> Optional[dict]:
        row = await self.pool.fetchrow("SELECT * FROM projects WHERE slug = $1", slug)
        return dict(row) if row else None

    async def create_project(self, name: str, slug: str, dna_document: str = "",
                             style_matrix: dict = None, tags: list = None) -> dict:
        row = await self.pool.fetchrow(
            """INSERT INTO projects (name, slug, dna_document, style_matrix, tags)
               VALUES ($1, $2, $3, $4::jsonb, $5) RETURNING *""",
            name, slug, dna_document, json.dumps(style_matrix or {}), tags or [])
        return dict(row)

    async def update_dna(self, slug: str, dna_document: str) -> Optional[dict]:
        row = await self.pool.fetchrow(
            """UPDATE projects SET dna_document = $1
               WHERE slug = $2 RETURNING id, name, slug, dna_document""",
            dna_document, slug)
        return dict(row) if row else None

    async def delete_project(self, slug: str) -> bool:
        if self.pool is None: return False
        
        try:
            async with self.pool.acquire() as conn:
                # 1. Забираем ID надёжным прямым запросом
                proj = await conn.fetchrow("SELECT id FROM projects WHERE slug = $1 OR name = $1 LIMIT 1", slug)
                if not proj:
                    print(f"[DNA DELETE] Ошибка: проект {slug} не найден в БД!")
                    return False
                    
                pid = proj['id']
                
                # 2. Вычищаем ВСЕ возможные хвосты по всем таблицам из скриншота
                await conn.execute("DELETE FROM context_summaries WHERE project_id = $1", pid)
                await conn.execute("DELETE FROM project_accounts WHERE project_id = $1", pid)
                await conn.execute("DELETE FROM sessions WHERE project_id = $1", pid)
                await conn.execute("DELETE FROM generations WHERE project_id = $1", pid)
                
                # 3. Сносим сам проект
                await conn.execute("DELETE FROM projects WHERE id = $1", pid)
                
                print(f"[DNA DELETE] ✅ Ядерная зачистка проекта {slug} завершена!")
                return True
        except Exception as e:
            print(f"[DNA DELETE] 💥 КРАШ БАЗЫ ПРИ УДАЛЕНИИ: {e}")
            return False

    async def delete_generation(self, generation_id: str) -> bool:
        if self.pool is None: return False
        try:
            async with self.pool.acquire() as conn:
                # В отличие от id проекта, тут UUID напрямую используется в таблице
                res = await conn.execute("DELETE FROM generations WHERE id = $1::uuid", generation_id)
                return res != "DELETE 0"
        except Exception as e:
            print(f"[DNA DELETE] Ошибка удаления генерации {generation_id}: {e}")
            return False

    async def move_generation(self, generation_id: str, target_slug: str) -> bool:
        if self.pool is None: return False
        try:
            async with self.pool.acquire() as conn:
                # 1. Находим настоящий UUID целевого проекта по его slug
                project_row = await conn.fetchrow("SELECT id FROM projects WHERE slug = $1", target_slug)
                if not project_row:
                    return False

                # 2. Перекидываем генерацию в него
                res = await conn.execute(
                    "UPDATE generations SET project_id = $1 WHERE id = $2::uuid",
                    project_row["id"], generation_id
                )
                return res != "UPDATE 0"
        except Exception as e:
            print(f"[DNA MOVE] Ошибка переноса генерации {generation_id}: {e}")
            return False


# ==================== ACCOUNTS ====================

    async def list_accounts(self, active_only: bool = True) -> List[dict]:
        if active_only:
            rows = await self.pool.fetch(
                "SELECT * FROM accounts WHERE is_active = TRUE ORDER BY name")
        else:
            rows = await self.pool.fetch("SELECT * FROM accounts ORDER BY name")
        return [dict(r) for r in rows]

    # ==================== GENERATIONS ====================

    async def capture_generation(self, project_id: str, prompt: str,
                                 account_id: str = None, negative_prompt: str = "",
                                 response_text: str = "", seed: int = None,
                                 model_params: dict = None, typography: dict = None,
                                 mask_source_url: str = None, result_urls: list = None,
                                 reference_urls: list = None, status: str = "generated") -> dict:
        """Capture a single generation from Safari Extension."""
        row = await self.pool.fetchrow(
            """INSERT INTO generations
               (project_id, account_id, prompt, negative_prompt, response_text,
                seed, model_params, typography, mask_source_url,
                result_urls, reference_urls, status)
               VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9, $10, $11, $12)
               RETURNING *""",
            project_id, account_id, prompt, negative_prompt, response_text,
            seed, json.dumps(model_params or {}), json.dumps(typography or {}),
            mask_source_url, result_urls or [], reference_urls or [], status)
        return dict(row)

    async def list_generations(self, project_slug: str, limit: int = 50,
                               offset: int = 0) -> List[dict]:
        rows = await self.pool.fetch(
            """SELECT g.* FROM generations g
               JOIN projects p ON g.project_id = p.id
               WHERE p.slug = $1 ORDER BY g.seq_num DESC LIMIT $2 OFFSET $3""",
            project_slug, limit, offset)
        return [dict(r) for r in rows]

    async def get_generation_count(self, project_id: str) -> int:
        return await self.pool.fetchval(
            "SELECT COUNT(*) FROM generations WHERE project_id = $1", project_id) or 0

    # ==================== CONTEXT SUMMARIES ====================

    async def get_latest_contexts(self, project_id: str) -> Dict[str, Optional[dict]]:
        rows = await self.pool.fetch(
            "SELECT * FROM v_latest_contexts WHERE project_id = $1", project_id)
        result = {"strategic": None, "tactical": None}
        for row in rows:
            result[row["context_type"]] = dict(row)
        return result

    async def save_context_summary(self, project_id: str, context_type: str,
                                   summary_text: str, gen_from_seq: int,
                                   gen_to_seq: int, gen_count: int,
                                   model_used: str = "gemini-flash",
                                   tokens_input: int = 0, tokens_output: int = 0) -> dict:
        row = await self.pool.fetchrow(
            """INSERT INTO context_summaries
               (project_id, context_type, summary_text, gen_from_seq, gen_to_seq,
                gen_count, model_used, tokens_input, tokens_output)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING *""",
            project_id, context_type, summary_text, gen_from_seq, gen_to_seq,
            gen_count, model_used, tokens_input, tokens_output)
        return dict(row)

    async def get_recent_prompts(self, project_id: str, limit: int = 10) -> List[dict]:
        rows = await self.pool.fetch(
            """SELECT seq_num, prompt, response_text, status, feedback_note, seed, created_at
               FROM generations WHERE project_id = $1
               ORDER BY seq_num DESC LIMIT $2""",
            project_id, limit)
        return [dict(r) for r in rows]

    # ==================== DNA CONTEXT ASSEMBLY ====================

    async def assemble_full_context(self, project_slug: str) -> Optional[dict]:
        """
        Assemble full 3-level DNA context for LLM injection:
          Level 1: core_dna (project constitution)
          Level 2a: strategic (last 50 gens summary)
          Level 2b: tactical (last 10 gens summary)
        """
        project = await self.get_project(project_slug)
        if not project:
            return None
        contexts = await self.get_latest_contexts(project["id"])
        return {
            "project_name": project["name"],
            "project_tags": project["tags"],
            "core_dna": project["dna_document"],
            "style_matrix": project["style_matrix"],
            "strategic_context": (contexts["strategic"]["summary_text"]
                                  if contexts["strategic"] else ""),
            "tactical_context": (contexts["tactical"]["summary_text"]
                                 if contexts["tactical"] else ""),
        }


# Singleton — import in router.py
db = DatabaseManager()
