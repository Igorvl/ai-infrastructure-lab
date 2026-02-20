import os
import json
import logging
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from litellm import completion, exceptions

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(levelname)s - %(message)s"
)
logger = logging.getLogger("AI-Gateway")

app = FastAPI(title="AI Design Infrastructure Gateway")

# Загрузка конфигурации
CONFIG_PATH = os.getenv("CONFIG_PATH", "deploy/routing_config.json")
try:
    with open(CONFIG_PATH, "r") as f:
        ROUTING_CONFIG = json.load(f)
    logger.info(f"✅ Configuration loaded from {CONFIG_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to load config: {e}")
    ROUTING_CONFIG = {}

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0 
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

def get_api_key(env_var_name: str) -> str:
    """Безопасное получение ключа из переменных окружения"""
    key = os.getenv(env_var_name)
    if not key:
        logger.warning(f"⚠️ API Key variable '{env_var_name}' is not set!")
        return ""
    return key

@app.get("/v1/models")
async def list_models():
    """Возвращаем список доступных моделей для совместимости с клиентами"""
    return {
        "object": "list",
        "data": [
            {"id": "primary_reasoning", "object": "model", "owned_by": "system"},
            {"id": "gemini-3-flash", "object": "model", "owned_by": "google"},
            {"id": "deepseek-v3", "object": "model", "owned_by": "alibaba"},
            {"id": "qwen-max", "object": "model", "owned_by": "alibaba"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # 1. Определяем конфигурацию (по умолчанию primary_reasoning)
    target_config = ROUTING_CONFIG.get("primary_reasoning")
    if not target_config:
        raise HTTPException(status_code=500, detail="Configuration for primary_reasoning not found")

    messages = [msg.model_dump() for msg in request.messages]
    
    # === ПОПЫТКА №1: ОСНОВНАЯ МОДЕЛЬ ===
    try:
        logger.info(f"🚀 Routing: primary_reasoning -> {target_config['provider']}/{target_config['model_name']}")
        
        response = completion(
            model=f"{target_config['provider']}/{target_config['model_name']}",
            messages=messages,
            api_key=get_api_key(target_config.get("api_key_env")),
            base_url=target_config.get("api_base"), # Важно для совместимых API
            temperature=request.temperature,
            max_tokens=request.max_tokens or target_config.get("max_tokens"),
            timeout=target_config.get("timeout", 30)
        )
        return response

    except Exception as e:
        logger.error(f"🔥 Primary model failed: {str(e)}")
        
        # === CIRCUIT BREAKER: ЗАПУСК РЕЗЕРВНЫХ МОДЕЛЕЙ ===
        fallbacks = ROUTING_CONFIG.get("fallbacks", [])
        
        if not fallbacks:
            logger.error("❌ No fallbacks configured!")
            raise HTTPException(status_code=502, detail=f"Primary model failed and no fallbacks available. Error: {str(e)}")

        logger.info("⚠️ Initiating Fallback Sequence...")

        for i, fallback_cfg in enumerate(fallbacks, 1):
            try:
                model_full_name = f"{fallback_cfg['provider']}/{fallback_cfg['model_name']}"
                logger.info(f"🛡️ Attempting Fallback #{i}: {model_full_name}")

                response = completion(
                    model=model_full_name,
                    messages=messages,
                    api_key=get_api_key(fallback_cfg.get("api_key_env")),
                    base_url=fallback_cfg.get("api_base"),
                    temperature=request.temperature,
                    timeout=fallback_cfg.get("timeout", 45) # Даем больше времени резерву
                )
                logger.info(f"✅ Fallback #{i} ({model_full_name}) succeeded!")
                return response

            except Exception as fallback_error:
                logger.warning(f"⚠️ Fallback #{i} failed: {str(fallback_error)}")
                continue # Пробуем следующую модель в списке

        # Если все резервы исчерпаны
        logger.critical("💀 All systems down. Routing failed.")
        raise HTTPException(status_code=503, detail="Service Unavailable: All AI models (primary and fallbacks) are unreachable.")
