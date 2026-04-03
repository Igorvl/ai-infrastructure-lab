import hashlib
import json
from litellm import acompletion

# Кэш-словарь: хранит привязку чатов (по хэшу первого сообщения) к конкретным проектам
# Формат: {'e99a18c4...': 'ksar-me'}
ROUTING_CACHE = {}

async def get_active_projects_dna(db_instance):
    """Выгружает все активные проекты и их Strategic контекст из базы."""
    if db_instance.pool is None: return []
    async with db_instance.pool.acquire() as conn:
        # У нас в таблице projects есть slug, а системные настройки можно взять из context_summaries
        rows = await conn.fetch("SELECT slug, name FROM projects")
        return [dict(r) for r in rows]

async def semantic_auto_detect(messages: list, db_instance, call_llm_func=None) -> str:
    """
    Магическая функция: сама определяет, о каком проекте говорит юзер.
    Возвращает slug проекта или 'UNKNOWN'.
    """
    if not messages: return "UNKNOWN"
    
    # 1. Берем КОРНЕВОЕ (самое первое) сообщение от человека в этом чате. 
    # Open WebUI всегда шлет полную историю, поэтому первый запрос = уникальный ID чата.
    root_usr_msg = next((m for m in messages if m.get("role") == "user"), None)
    if not root_usr_msg: return "UNKNOWN"
    
    user_text = str(root_usr_msg.get("content", ""))
    msg_hash = hashlib.md5(user_text.encode('utf-8')).hexdigest()
    
    # 2. Отработка КЭША за 0 мс (если мы уже сортировали этот чат ранее)
    if msg_hash in ROUTING_CACHE:
        return ROUTING_CACHE[msg_hash]
        
    # 3. Выгружаем проекты (кандидаты для Сортировочной Шляпы)
    projects = await get_active_projects_dna(db_instance)
    if not projects: return "UNKNOWN"
    
    valid_slugs = [p["slug"] for p in projects]
    projects_list_txt = "\n".join([f"- {p['slug']} (Название: {p['name']})" for p in projects])

    system_prompt = f"""You are a strict Semantic Router. Categorize the user's message into EXACTLY ONE of the project slugs based on relevance. 
Rules:
1. Reply ONLY with the exact slug name.
2. If it doesn't clearly match any project, reply 'UNKNOWN'.
Available projects:
{projects_list_txt}"""

    try:
        print(f"[SEMANTIC ROUTER] Анализирую новый контекст ({user_text[:20]}...)...")
        # 🔥 Кидаем запрос в самую быструю и дешевую модель:
        routing_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text[:800]} 
        ]
        
        if call_llm_func:
            res = await call_llm_func(
                mid="openrouter/qwen/qwen3.6-plus:free",
                messages=routing_messages,
                api_key_env="OPENROUTER_API_KEY",
                api_base="https://openrouter.ai/api/v1",
                stream=False
            )
        else:
            from litellm import acompletion
            import os
            res = await acompletion(
                model="openrouter/qwen/qwen3.6-plus:free",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                api_base="https://openrouter.ai/api/v1",
                messages=routing_messages,
                temperature=0.0
            )
        
        detected_slug = res.choices[0].message.content.strip().lower()
        # DEBUG: покажем что LLM вернул и с чем сравниваем
        print(f"[SEMANTIC ROUTER] LLM ответил: '{detected_slug}' ")

        valid_slugs_lower = [p["slug"].lower() for p in projects]
        if detected_slug in valid_slugs_lower:
            # Возвращаем оригинальный slug из БД (MLOps или mlops), чтобы не сломать Foreign Keys базы
            actual_slug = next(p["slug"] for p in projects if p["slug"].lower() == detected_slug)
            ROUTING_CACHE[msg_hash] = actual_slug
            print(f"✅ [SEMANTIC ROUTER] Авто-маршрутизация: [{actual_slug}]")
            return actual_slug

    except Exception as e:
        print(f"❌ [SEMANTIC ERROR] Ошибка авто-маршрутизации: {e}")
        
    return "UNKNOWN"


async def route_text_capture(prompt_text: str, db_instance, call_llm_func=None) -> str:
    """
    Маршрутизация захвата из Chrome Extension.
    Оборачивает текст в messages-формат и вызывает semantic_auto_detect.
    """
    if not prompt_text or not prompt_text.strip():
        return "UNKNOWN"
    messages = [{"role": "user", "content": prompt_text[:800]}]
    return await semantic_auto_detect(messages, db_instance, call_llm_func)

