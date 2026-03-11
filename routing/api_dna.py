"""
Project DNA — API Endpoints
Modular FastAPI router for project management and generation capture.
Endpoints:
  GET  /v1/dna/projects          — list all projects
  POST /v1/dna/projects          — create project
  GET  /v1/dna/projects/{slug}   — get project + stats
  PUT  /v1/dna/projects/{slug}/dna — update DNA document
  POST /v1/dna/capture           — capture generation (from Safari Extension)
  GET  /v1/dna/generations/{slug} — list generations for project
  GET  /v1/dna/context/{slug}    — get assembled DNA context (3 levels)
  GET  /v1/dna/accounts          — list accounts
  GET  /v1/dna/health            — check DB connection
"""
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
# Import our database singleton
from db import db
from qdrant_db import vector_db
from minio_storage import storage
from auto_summarize import maybe_summarize
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
class SearchRequest(BaseModel):
    query: str
    project_slug: str = ""
    limit: int = 5

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
# ==================== Project Endpoints ====================
@router.get("/projects")
async def list_projects(status: Optional[str] = None):
    """List all projects with stats."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    projects = await db.list_projects(status)
    return {"projects": projects, "count": len(projects)}
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
@router.put("/projects/{slug}/dna")
async def update_dna(slug: str, data: DNAUpdate):
    """Update the DNA document (project constitution)."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    result = await db.update_dna(slug, data.dna_document)
    if not result:
        raise HTTPException(404, f"Project '{slug}' not found")
    logger.info(f"DNA updated for project: {slug}")
    return {"project": result}
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
# ==================== Context Assembly ====================
@router.get("/context/{slug}")
async def get_context(slug: str):
    """
    Get the full assembled DNA context (3 levels).
    This is what gets injected into LLM prompts.
    """
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    context = await db.assemble_full_context(slug)
    if not context:
        raise HTTPException(404, f"Project '{slug}' not found")
    return context
# ==================== Accounts ====================
@router.get("/accounts")
async def list_accounts():
    """List all AI service accounts."""
    if not db.pool:
        raise HTTPException(503, "Database not connected")
    accounts = await db.list_accounts()
    return {"accounts": accounts}
# ==================== Health Check ====================


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

@router.post("/search")
async def search_prompts(data: SearchRequest):
    """Semantic search: find similar prompts using Qdrant vectors."""
    results = vector_db.search_similar_prompts(
        project_slug=data.project_slug,
        query_text=data.query,
        limit=data.limit
    )
    return {"query": data.query, "results": results, "count": len(results)}

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
