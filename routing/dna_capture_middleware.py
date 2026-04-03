import re
import asyncio
import logging
import json
from db import db

logger = logging.getLogger(__name__)

def extract_project_slug(model: str, messages: list) -> str:
    """Извлекает project_slug в приоритете: Model Prefix > System Prompt > Default"""
    # 1. Model prefix: "dna:project-slug/real-model"
    if model.startswith("dna:"):
        slug = model.split("/")[0].replace("dna:", "").strip()
        if slug:
            return slug

    # 2. System prompt tag: [DNA:project-slug]
    for msg in messages:
        if msg.get("role") == "system":
            match = re.search(r'\[DNA:([\w-]+)\]', str(msg.get("content", "")))
            if match:
                return match.group(1)

    return "open-webui"  # дефолт

def extract_real_model(model: str) -> str:
    """Очищает префикс DNA для реального запроса к LLM"""
    if model.startswith("dna:") and "/" in model:
        return model.split("/", 1)[1]
    if model.startswith("dna:"):
        return model.replace("dna:", "").strip()
    return model

def extract_user_prompt(messages: list) -> str:
    """Достает последнее сообщение пользователя"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Мультимодальность (картинки + текст)
                return " ".join(part.get("text", "") for part in content if part.get("type", "") == "text")
            return str(content)
    return ""

def build_conversation_context(messages: list) -> str:
    """Собирает контекст многошагового диалога в markdown-вид для RAG памяти"""
    context_parts = []
    for msg in messages[:-1]:  # Все кроме последнего (он уже лежит в prompt)
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if p.get("type", "") == "text")
        content = str(content)
        if role != "system" and content:
            prefix = "👤" if role == "user" else "🤖"
            context_parts.append(f"{prefix} **{role.capitalize()}:** {content[:500]}")
    return "\n\n".join(context_parts) if context_parts else ""


async def capture_to_dna(
    project_slug: str,
    model_name: str,
    prompt_text: str,
    response_text: str,
    conversation_context: str
):
    """Фоновая задача прямой вставки в базу Postgres"""
    if not prompt_text.strip() or not response_text.strip():
        return

    try:
        from db import db
        import json
        
        if db.pool is None:
            logger.warning("[DNA Capture] DB pool is None, cannot save generation.")
            return

        # 1. Извлекаем реальный UUID проекта по его названию (slug)
        project_record = await db.get_project(project_slug)
        if not project_record:
            logger.error(f"[DNA Capture] ❌ Project slug '{project_slug}' not found in database!")
            return
            
        real_project_id = project_record["id"]

        # 2. Правильные колонки из твоего скриншота! (id и created_at база создаст сама)
        query = """
            INSERT INTO generations (project_id, prompt, response_text, source, metadata)
            VALUES ($1, $2, $3, $4, $5::jsonb)
        """
        
        # 3. Сохраняем имя модели и историю переписки в JSON
        meta_str = json.dumps({
            "model_name": model_name,
            "conversation_context": conversation_context,
            "turn_count": conversation_context.count("👤")
        })
        
        # 4. Отправляем в пул!
        await db.pool.execute(
            query,
            real_project_id,
            prompt_text,
            response_text,
            "open-webui",
            meta_str
        )
        logger.info(f"[DNA Capture] ✅ Saved {project_slug} | {model_name} | text: {prompt_text[:30]}...")
    except Exception as e:
        logger.error(f"[DNA Capture] ❌ DB error: {e}")

