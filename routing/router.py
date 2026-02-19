import os
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from litellm import completion

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI-Gateway")

app = FastAPI(title="AI Infrastructure Gateway v3.0")

# --- КОНФИГУРАЦИЯ ---
CONFIG_PATH = os.getenv("CONFIG_PATH", "deploy/antigravity.json")
try:
    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
    logger.info(f"✅ Configuration loaded from {CONFIG_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to load config: {e}")
    CONFIG = {"models": {}}

# --- МОДЕЛИ ДАННЫХ ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

# --- ЭНДПОИНТЫ ---
@app.get("/health")
async def health_check():
    return {"status": "operational", "models_loaded": list(CONFIG.get("models", {}).keys())}

@app.get("/v1/models")
async def list_models():
    data = []
    for model_id, params in CONFIG.get("models", {}).items():
        data.append({
            "id": model_id,
            "object": "model",
            "created": 1677610602,
            "owned_by": params.get("provider", "unknown"),
            "name": f"{model_id} ({params.get('model_name')})"
        })
    return {"object": "list", "data": data}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    requested_model = request.model
    
    # 1. Fallback логика выбора модели
    if requested_model not in CONFIG["models"]:
        logger.warning(f"Requested model '{requested_model}' not found. Defaulting to Primary.")
        target_role = CONFIG["fallback_order"][0]
    else:
        target_role = requested_model

    model_cfg = CONFIG["models"][target_role]
    
    # --- ИСПРАВЛЕНИЕ: НОРМАЛИЗАЦИЯ ИМЕНИ МОДЕЛИ ---
    provider = model_cfg["provider"]
    real_model_name = model_cfg["model_name"]
    
    # LiteLLM требует специфичные префиксы
    if provider == "google":
        # Меняем 'google' на 'gemini'
        litellm_model = f"gemini/{real_model_name}"
    elif provider == "openai":
        # Для OpenAI-compatible (DeepSeek/Qwen) префикс часто не нужен или openai/
        litellm_model = real_model_name
    elif provider == "zhipu":
        # Zhipu AI
        litellm_model = f"zhipu/{real_model_name}"
    else:
        # Default: provider/model
        litellm_model = f"{provider}/{real_model_name}"

    # 3. Собираем аргументы
    kwargs = {
        "model": litellm_model,
        "messages": [m.dict() for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens or model_cfg.get("max_tokens", 4096),
        "api_key": os.getenv(model_cfg.get("api_key_env")),
    }

    if "api_base" in model_cfg:
        kwargs["api_base"] = model_cfg["api_base"]
    
    if "extra_body" in model_cfg:
        kwargs["extra_body"] = model_cfg["extra_body"]

    logger.info(f"🚀 Routing: {target_role} -> {litellm_model}")

    try:
        # Вызов LiteLLM
        response = completion(**kwargs)
        return response
        
    except Exception as e:
        logger.error(f"🔥 Error calling {target_role}: {str(e)}")
        # Возвращаем 500, чтобы было видно в логах клиента, но можно сделать Fallback
        raise HTTPException(status_code=500, detail=f"Provider Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
