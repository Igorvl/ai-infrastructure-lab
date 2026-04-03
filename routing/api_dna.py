import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
# Import our database singleton
from db import db
from qdrant_db import vector_db
from minio_storage import storage
from auto_summarize import maybe_summarize
from semantic_router import route_text_capture
import asyncio

logger = logging.getLogger("AI-Router.DNA")

# Create sub-router with /v1/dna prefix
#   APIRouter — модульный роутер FastAPI
#   prefix — все endpoints начинаются с /v1/dna
#   tags — группировка в Swagger UI документации
router = APIRouter(prefix="/v1/dna", tags=["Project DNA"])


# ==================== Pydantic Models ====================
# Pydantic — валидация входных данных (что прислал клиент)

class ProjectCreate(BaseModel):
    name: str
    slug: str
    dna_document: str = ""
    style_matrix: dict = {}
    tags: list = []

class DNAUpdate(BaseModel):
    dna_document: str

class DNADocBody(BaseModel):
    dna_document: Optional[str] = None

class ProjectPatch(BaseModel):
    name: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    project_slug: str = ""
    limit: int = 5

class RouteRequest(BaseModel):
    """
    Что Extension шлёт когда не знает, в какой проект писать.
    Сервер сам определяет проект через Semantic Router.
    """
    prompt_text: str
    output_text: str = ""
    model_name: str = "unknown"
    source: str = "ai-studio-extension"
    result_urls: list = []
    parameters: dict = {}
    system_instruction: str = ""
    metadata: dict = {}

class CaptureRequest(BaseModel):
    """What Safari Extension sends for each generation."""
    project_slug: str            # which project
    prompt: str                  # the generation prompt
    account_id: Optional[str] = None
    negative_prompt: str = ""
    response_text: str = ""
    seed: Optional[int] = None
    model_params: dict = {}      # {"model": "nano-banana-pro", "steps": 30, ...}
    typography: dict = {}        # {"font": "Helvetica", "size": 72, ...}
    mask_source_url: Optional[str] = None
    result_urls: list = []       # ["https://...image.png"]
    reference_urls: list = []    # style transfer sources
    status: str = "generated"    # generated/approved/rejected


# ==================== Generation Patch ====================

@router.patch("/generations/{generation_id}")
async def patch_generation(generation_id: str, data: dict):
    """Update generation fields after the fact — e.g. result_urls after MinIO upload."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    try:
        async with db.pool.acquire() as conn:
            await conn.execute(
                "UPDATE generations SET result_urls = $1 WHERE id = $2",
                data.get('result_urls', []), generation_id
            )
        logger.info(f"[PATCH gen] Updated {generation_id}: {len(data.get('result_urls', []))} URL(s)")
        return {"ok": True, "generation_id": generation_id}
    except Exception as e:
        logger.error(f"[PATCH gen] Error: {e}")
        raise HTTPException(500, str(e))


# ==================== Project Endpoints ====================

@router.get("/projects")
async def list_projects(status: Optional[str] = None):
    """List all projects with stats, including archived flag."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    try:
        async with db.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT slug, name, tags,
                       COALESCE(archived, FALSE) AS archived
                FROM projects
                ORDER BY created_at DESC
                """
            )
            projects = [dict(r) for r in rows]
        return {"projects": projects, "count": len(projects)}
    except Exception as e:
        logger.error(f"[list_projects] {e}")
        raise HTTPException(500, str(e))


@router.post("/projects")
async def create_project(data: ProjectCreate):
    """Create a new design project."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    try:
        project = await db.create_project(
            name=data.name, slug=data.slug,
            dna_document=data.dna_document,
            style_matrix=data.style_matrix,
            tags=data.tags,
        )
        logger.info(f"Project created: {data.slug}")
        return {"project": project}
    except Exception as e:
        if "unique" in str(e).lower():
            raise HTTPException(409, f"Project with slug '{data.slug}' already exists")
        raise HTTPException(500, str(e))


@router.get("/projects/{slug}")
async def get_project(slug: str):
    """Get project details + generation stats."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    project = await db.get_project(slug)
    if not project:
        raise HTTPException(404, f"Project '{slug}' not found")
    count = await db.get_generation_count(project["id"])
    contexts = await db.get_latest_contexts(project["id"])
    return {
        "project": project,
        "generation_count": count,
        "has_strategic_context": contexts["strategic"] is not None,
        "has_tactical_context": contexts["tactical"] is not None,
    }


@router.patch("/projects/{slug}")
async def patch_project(slug: str, body: ProjectPatch):
    """Rename a project (updates display name, slug stays the same)."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    if not body.name or not body.name.strip():
        raise HTTPException(400, "Name cannot be empty")
    try:
        async with db.pool.acquire() as conn:
            res = await conn.execute(
                "UPDATE projects SET name = $1 WHERE slug = $2",
                body.name.strip(), slug
            )
        if res == "UPDATE 0":
            raise HTTPException(404, f"Project '{slug}' not found")
        logger.info(f"[RENAME] Project '{slug}' renamed to '{body.name}'")
        return {"success": True, "slug": slug, "name": body.name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[patch_project] {e}")
        raise HTTPException(500, str(e))


# ── DNA Document (Level 1 Context — fixed, manually written) ─────

@router.get("/projects/{slug}/dna-document")
async def get_dna_document(slug: str):
    """Get the DNA Document (project constitution) for a project."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    try:
        async with db.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT dna_document FROM projects WHERE slug = $1", slug
            )
        if not row:
            raise HTTPException(404, f"Project '{slug}' not found")
        return {"slug": slug, "dna_document": row["dna_document"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[get_dna_document] {e}")
        raise HTTPException(500, str(e))


@router.put("/projects/{slug}/dna-document")
async def update_dna_document(slug: str, body: DNADocBody):
    """Save (overwrite) the DNA Document for a project."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    try:
        async with db.pool.acquire() as conn:
            res = await conn.execute(
                "UPDATE projects SET dna_document = $1 WHERE slug = $2",
                body.dna_document, slug
            )
        if res == "UPDATE 0":
            raise HTTPException(404, f"Project '{slug}' not found")
        logger.info(f"[DNA-DOC] Updated for '{slug}' ({len(body.dna_document or '')} chars)")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[update_dna_document] {e}")
        raise HTTPException(500, str(e))


@router.put("/projects/{slug}/dna")
async def update_dna(slug: str, data: DNAUpdate):
    """Update the DNA document via legacy endpoint (project constitution)."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    result = await db.update_dna(slug, data.dna_document)
    if not result:
        raise HTTPException(404, f"Project '{slug}' not found")
    logger.info(f"DNA updated for project: {slug}")
    return {"project": result}


# ── Archive / Unarchive ───────────────────────────────────────────

@router.post("/projects/{slug}/archive")
async def archive_project(slug: str):
    """Move a project to the archive (soft-hide)."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    try:
        async with db.pool.acquire() as conn:
            res = await conn.execute(
                "UPDATE projects SET archived = TRUE WHERE slug = $1", slug
            )
        if res == "UPDATE 0":
            raise HTTPException(404, f"Project '{slug}' not found")
        logger.info(f"[ARCHIVE] Project '{slug}' archived")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[archive_project] {e}")
        raise HTTPException(500, str(e))


@router.post("/projects/{slug}/unarchive")
async def unarchive_project(slug: str):
    """Restore a project from the archive."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    try:
        async with db.pool.acquire() as conn:
            res = await conn.execute(
                "UPDATE projects SET archived = FALSE WHERE slug = $1", slug
            )
        if res == "UPDATE 0":
            raise HTTPException(404, f"Project '{slug}' not found")
        logger.info(f"[UNARCHIVE] Project '{slug}' restored")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[unarchive_project] {e}")
        raise HTTPException(500, str(e))


# ── Delete Project ────────────────────────────────────────────────

@router.delete("/projects/{slug}")
async def delete_project(slug: str):
    """Permanently delete a project and all its generations."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    try:
        result = await db.delete_project(slug)
        if not result:
            raise HTTPException(404, f"Project '{slug}' not found")
        logger.info(f"[DELETE] Project '{slug}' deleted")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[delete_project] {e}")
        raise HTTPException(500, str(e))


# ==================== Capture Endpoint ====================

@router.post("/capture")
async def capture_generation(data: CaptureRequest):
    """
    Capture a generation from Safari Extension.
    This is the main ingestion endpoint — every prompt/result goes here.
    """
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    # Resolve project by slug
    project = await db.get_project(data.project_slug)
    if not project:
        raise HTTPException(404, f"Project '{data.project_slug}' not found")

    gen = await db.capture_generation(
        project_id=str(project["id"]),
        prompt=data.prompt,
        account_id=data.account_id,
        negative_prompt=data.negative_prompt,
        response_text=data.response_text,
        seed=data.seed,
        model_params=data.model_params,
        typography=data.typography,
        mask_source_url=data.mask_source_url,
        result_urls=data.result_urls,
        reference_urls=data.reference_urls,
        status=data.status,
    )

    # Векторизуем и сохраняем текст в Qdrant
    vector_db.add_prompt(
        project_slug=data.project_slug,
        prompt_text=data.prompt,
        generation_id=str(gen["id"])
    )

    # Auto-summarize in background (non-blocking)
    asyncio.create_task(maybe_summarize(
        project_id=str(gen["project_id"]),
        project_slug=data.project_slug,
        current_seq=gen["seq_num"]
    ))

    logger.info(f"Captured gen #{gen['seq_num']} for {data.project_slug}")
    return {
        "generation_id": str(gen["id"]),
        "seq_num": gen["seq_num"],
        "project": data.project_slug,
    }


# ==================== Generations ====================

@router.get("/generations/{slug}")
async def list_generations(slug: str, limit: int = 50, offset: int = 0):
    """List generations for a project (newest first)."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    gens = await db.list_generations(slug, limit, offset)
    return {"generations": gens, "count": len(gens)}


@router.delete("/generations/{generation_id}")
async def delete_generation(generation_id: str):
    """Delete a single generation by ID."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    try:
        async with db.pool.acquire() as conn:
            res = await conn.execute(
                "DELETE FROM generations WHERE id = $1", generation_id
            )
        if res == "DELETE 0":
            raise HTTPException(404, "Generation not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[delete_generation] {e}")
        raise HTTPException(500, str(e))


@router.put("/generations/{generation_id}")
async def move_generation(generation_id: str, target_project: str):
    """Move a generation to another project (param: ?target_project=slug)."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    try:
        async with db.pool.acquire() as conn:
            project = await conn.fetchrow(
                "SELECT id FROM projects WHERE slug = $1", target_project
            )
            if not project:
                raise HTTPException(404, f"Target project '{target_project}' not found")
            res = await conn.execute(
                "UPDATE generations SET project_id = $1 WHERE id = $2",
                project["id"], generation_id
            )
        if res == "UPDATE 0":
            raise HTTPException(404, "Generation not found")
        return {"success": True, "target_project": target_project}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[move_generation] {e}")
        raise HTTPException(500, str(e))


# ==================== Context Assembly ====================

@router.get("/context/{slug}")
async def get_context(slug: str):
    """
    Get the full assembled DNA context (all 3 levels):
      Level 1: dna_document  — fixed, manually written constitution
      Level 2: tactical      — last 5 generations auto-summary
      Level 3: strategic     — last 20 generations auto-summary
    This is what gets injected into LLM prompts.
    """
    if not db.pool:
        raise HTTPException(503, "Database not connected")

    context = await db.assemble_full_context(slug)
    if not context:
        raise HTTPException(404, f"Project '{slug}' not found")

    # Ensure dna_document is included (Level 1 context).
    # assemble_full_context may not return it yet — we fetch manually.
    if "dna_document" not in context:
        try:
            async with db.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT dna_document FROM projects WHERE slug = $1", slug
                )
            context["dna_document"] = row["dna_document"] if row else None
        except Exception as e:
            logger.warning(f"[get_context] Could not fetch dna_document: {e}")
            context["dna_document"] = None

    return context


# ==================== Accounts ====================

@router.get("/accounts")
async def list_accounts():
    """List all AI service accounts."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    accounts = await db.list_accounts()
    return {"accounts": accounts}


# ==================== Files & Upload ====================

@router.post("/upload/{slug}")
async def upload_image(slug: str, file: UploadFile):
    """Upload an image to project storage."""
    data = await file.read()
    content_type = file.content_type or "application/octet-stream"
    result = storage.upload_image(
        project_slug=slug,
        filename=file.filename,
        data=data,
        content_type=content_type
    )
    return result


@router.get("/files/{slug}")
async def list_project_files(slug: str):
    """List all files for a project."""
    images = storage.list_objects("project-images", prefix=f"{slug}/")
    return {"project": slug, "files": images, "count": len(images)}


# ==================== Semantic Search ====================

@router.post("/search")
async def search_prompts(data: SearchRequest):
    """Semantic search: find similar prompts using Qdrant vectors."""
    results = vector_db.search_similar_prompts(
        project_slug=data.project_slug,
        query_text=data.query,
        limit=data.limit
    )
    return {"query": data.query, "results": results, "count": len(results)}


# ==================== Semantic Auto-Router ====================

@router.post("/route")
async def route_and_capture(data: RouteRequest):
    """
    Semantic Auto-Routing для Chrome Extension.

    Extension шлёт захват БЕЗ project_slug.
    Мы сами определяем проект через Semantic Router и сразу сохраняем.

    Если проект не определён (UNKNOWN) — возвращаем null,
    Extension сама поставит в очередь.
    """
    if not db.pool:
        raise HTTPException(503, "Database not connected")

    # 1. Запускаем Semantic Router
    from router import call_with_key_rotation
    detected_slug = await route_text_capture(data.prompt_text, db, call_with_key_rotation)

    if detected_slug == "UNKNOWN":
        logger.info(f"[ROUTE] UNKNOWN — Extension поставит в очередь")
        return {
            "project_slug": None,
            "method": "unknown",
            "generation_id": None,
            "seq_num": None,
        }

    # 2. Проверяем что проект существует в БД
    project = await db.get_project(detected_slug)
    if not project:
        logger.warning(f"[ROUTE] Router вернул '{detected_slug}', но проект не найден в БД")
        return {
            "project_slug": None,
            "method": "unknown",
            "generation_id": None,
            "seq_num": None,
        }

    # 3. Сохраняем генерацию
    gen = await db.capture_generation(
        project_id=str(project["id"]),
        prompt=data.prompt_text,
        negative_prompt="",
        response_text=data.output_text,
        seed=None,
        model_params={"model": data.model_name, **data.parameters},
        typography={},
        mask_source_url=None,
        result_urls=data.result_urls,
        reference_urls=[],
        status="generated",
    )

    # 4. Векторизуем в Qdrant
    vector_db.add_prompt(
        project_slug=detected_slug,
        prompt_text=data.prompt_text,
        generation_id=str(gen["id"])
    )

    # 5. Авто-суммаризация в фоне (не блокирует ответ)
    asyncio.create_task(maybe_summarize(
        project_id=str(gen["project_id"]),
        project_slug=detected_slug,
        current_seq=gen["seq_num"]
    ))

    logger.info(f"[ROUTE] ✅ Auto-captured gen #{gen['seq_num']} → {detected_slug}")

    return {
        "project_slug": detected_slug,
        "method": "semantic",
        "generation_id": str(gen["id"]),
        "seq_num": gen["seq_num"],
    }


# ==================== Health Check ====================

@router.get("/health")
async def dna_health():
    """Check database connectivity."""
    if not db.pool:
        return {"status": "disconnected", "database": "unavailable"}
    try:
        count = await db.pool.fetchval("SELECT COUNT(*) FROM projects")
        return {"status": "connected", "projects_count": count}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

