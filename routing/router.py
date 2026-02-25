import os
import json
import logging
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from litellm import acompletion

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(levelname)s - %(message)s"
)
logger = logging.getLogger("AI-Gateway")

app = FastAPI(title="AI Design Infrastructure Gateway")

# Загрузка конфигурации
CONFIG_PATH = os.getenv("CONFIG_PATH", "deploy/antigravity.json")
try:
    with open(CONFIG_PATH, "r") as f:
        ROUTING_CONFIG = json.load(f)
    logger.info(f"✅ Configuration loaded from {CONFIG_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to load config: {e}")
    ROUTING_CONFIG = {}

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

def get_api_key(env_var_name: str) -> str:
    return os.getenv(env_var_name, "")

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "primary_reasoning", "object": "model", "owned_by": "system"},
            {"id": "deepseek-v3", "object": "model", "owned_by": "siliconflow"},
            {"id": "qwen-max", "object": "model", "owned_by": "siliconflow"},
            {"id": "GLM_5", "object": "model", "owned_by": "siliconflow"},
            {"id": "qwen-vision", "object": "model", "owned_by": "siliconflow"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    requested_model = request.model
    messages = [msg.model_dump() for msg in request.messages]

    has_images = any(
        isinstance(msg.get("content"), list) and
        any(part.get("type") == "image_url" for part in msg["content"] if isinstance(part, dict))
        for msg in messages
    )

    def get_model_params(model_name):
        for item in ROUTING_CONFIG.get("model_list", []):
            if item.get("model_name") == model_name:
                return item.get("litellm_params")
        return None

    target_params = get_model_params(requested_model) or get_model_params("primary_reasoning")
    
    models_queue = [target_params]
    if has_images:
        vision_fb = get_model_params("qwen-vision")
        if vision_fb: models_queue.append(vision_fb)
    else:
        for fb_name in target_params.get("fallbacks", []):
            fb_params = get_model_params(fb_name)
            if fb_params: models_queue.append(fb_params)

    last_error = None
    for current_params in models_queue:
        actual_model = current_params.get("model")
        try:
            logger.info(f"🚀 Routing to: {actual_model}")
            
            response = await acompletion(
                model=actual_model,
                messages=messages,
                api_key=get_api_key(current_params.get("api_key_env")),
                api_base=current_params.get("api_base"),
                api_version=current_params.get("api_version"),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=request.stream,
                timeout=180 if has_images else 45
            )

            if request.stream:
                async def stream_generator():
                    async for chunk in response:
                        data = chunk.model_dump_json() if hasattr(chunk, "model_dump_json") else json.dumps(chunk)
                        yield f"data: {data}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(stream_generator(), media_type='text/event-stream')

            return response

        except Exception as e:
            logger.error(f"🔥 Fail: {actual_model} | Error: {str(e)}")
            last_error = e
            continue

    raise HTTPException(status_code=502, detail=f"All models failed. Last error: {str(last_error)}")
