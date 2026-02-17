import json
import logging
import os
import time
from typing import List, Dict, Any, Optional

import tiktoken
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from litellm import completion, embedding
from dotenv import load_dotenv

# Загрузка переменных окружения (API ключи)
load_dotenv()

# Настройка логирования (Enterprise style)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("router.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AI_Router")

app = FastAPI(title="AI Design Lab Gateway", version="1.0.0")

# --- ЗАГРУЗКА КОНФИГУРАЦИИ ---
CONFIG_PATH = "../deploy/antigravity.json"

def load_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {CONFIG_PATH}")
        raise RuntimeError("Critical: antigravity.json missing!")

config = load_config()
MAX_TOKENS = config["constraints"]["max_context_tokens"]
PRIMARY_MODEL = config["routing_rules"]["primary_model"]
FALLBACK_CHAIN = config["routing_rules"]["fallback_chain"]

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def count_tokens(messages: List[Dict[str, str]]) -> int:
    """Считает токены в истории сообщений (приблизительно для разных моделей)"""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # message start/end tokens
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
    return num_tokens

def truncate_context(messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
    """Жестко обрезает контекст, удаляя старые сообщения, чтобы влезть в лимит"""
    current_tokens = count_tokens(messages)
    
    if current_tokens <= max_tokens:
        return messages

    logger.warning(f"Context overflow ({current_tokens} > {max_tokens}). Truncating...")
    
    # Оставляем системный промпт (обычно первое сообщение)
    system_msg = []
    if messages and messages[0].get("role") == "system":
        system_msg = [messages[0]]
        messages = messages[1:]
    
    truncated_messages = []
    # Идем с конца (самые новые сообщения важнее)
    tokens_so_far = count_tokens(system_msg)
    
    for msg in reversed(messages):
        msg_tokens = count_tokens([msg])
        if tokens_so_far + msg_tokens > max_tokens:
            break
        truncated_messages.insert(0, msg)
        tokens_so_far += msg_tokens
        
    final_context = system_msg + truncated_messages
    logger.info(f"Context truncated to {tokens_so_far} tokens.")
    return final_context

# --- ОСНОВНАЯ ЛОГИКА (CIRCUIT BREAKER) ---

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    user_model = data.get("model", PRIMARY_MODEL) # Игнорируем запрос UI, форсируем нашу логику
    
    # Попытка 1: Основная модель (Gemini)
    try:
        logger.info(f"Attempting Primary Model: {PRIMARY_MODEL}")
        response = completion(
            model=PRIMARY_MODEL,
            messages=messages,
            api_key=os.getenv("GEMINI_API_KEY") # Берется из .env
        )
        return response
        
    except Exception as e:
        logger.error(f"Primary model {PRIMARY_MODEL} failed: {str(e)}")
        logger.info("Initiating FALLBACK PROTOCOL...")
        
        # Подготовка к Fallback: Обрезка контекста
        safe_messages = truncate_context(messages, MAX_TOKENS)
        
        # Цикл по запасным моделям
        for fallback_model in FALLBACK_CHAIN:
            try:
                logger.info(f"Attempting Fallback Model: {fallback_model}")
                # Выбираем ключ API в зависимости от модели
                api_key = os.getenv("GLM_API_KEY") if "glm" in fallback_model else os.getenv("ERNIE_API_KEY")
                
                response = completion(
                    model=fallback_model,
                    messages=safe_messages,
                    api_key=api_key
                )
                logger.info(f"Recovered successfully with {fallback_model}")
                return response
            except Exception as fallback_error:
                logger.error(f"Fallback {fallback_model} failed: {str(fallback_error)}")
                continue # Пробуем следующую
                
        # Если все умерли
        logger.critical("ALL MODELS FAILED. System is down.")
        raise HTTPException(status_code=503, detail="All AI models are currently unavailable.")

# Запуск: uvicorn router:app --host 0.0.0.0 --port 8000
