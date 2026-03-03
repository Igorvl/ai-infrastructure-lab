import os, json, logging, httpx, uuid, re, asyncio, time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from litellm import acompletion

# Edge TTS для русского
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("AI-Router")

app = FastAPI(title="Project DNA Router")
AUDIO_OUTPUT_PATH = "/app/audio_out"
AUDIO_PUBLIC_URL = os.getenv("AUDIO_PUBLIC_URL", "http://172.25.9.33:8090")
os.makedirs(AUDIO_OUTPUT_PATH, exist_ok=True)

CONFIG_PATH = os.getenv("CONFIG_PATH", "deploy/antigravity.json")
logger.info(f"Edge TTS: {'available' if EDGE_TTS_AVAILABLE else 'NOT available (pip install edge-tts)'}")

# ========== РОТАЦИЯ КЛЮЧЕЙ ==========
def load_api_keys(prefix: str) -> list[str]:
    keys = []
    base = os.getenv(prefix)
    if base:
        keys.append(base)
    for i in range(2, 20):
        k = os.getenv(f"{prefix}_{i}")
        if k:
            keys.append(k)
    return keys

KEY_POOLS: dict[str, list[str]] = {}
KEY_INDEX: dict[str, int] = {}

def init_key_pools():
    for env_name in ["GEMINI_API_KEY", "SILICONFLOW_API_KEY"]:
        keys = load_api_keys(env_name)
        KEY_POOLS[env_name] = keys
        KEY_INDEX[env_name] = 0
        logger.info(f"Key pool [{env_name}]: {len(keys)} key(s) loaded")

init_key_pools()

def get_all_keys(env_name: str) -> list[str]:
    keys = KEY_POOLS.get(env_name, [])
    if not keys:
        single = os.getenv(env_name)
        return [single] if single else []
    return keys

# ========== ОПРЕДЕЛЕНИЕ ЯЗЫКА ==========
def has_cyrillic(text: str) -> bool:
    """Проверяет наличие кириллицы в тексте"""
    return bool(re.search('[а-яА-ЯёЁ]', text))

def detect_language(text: str) -> str:
    """Определяет язык: ru или en"""
    # Считаем долю кириллических символов
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    cyrillic = sum(1 for c in letters if re.match('[а-яА-ЯёЁ]', c))
    ratio = cyrillic / len(letters)
    return "ru" if ratio > 0.3 else "en"

# ========== EDGE TTS ГОЛОСА ==========
EDGE_VOICES = {
    "ru-RU-DariyaNeural": "Дарья (жен, тёплый)",
    "ru-RU-SvetlanaNeural": "Светлана (жен, чёткий)",
    "ru-RU-DmitryNeural": "Дмитрий (муж)",
}

KOKORO_VOICES = {
    "af_heart": "Heart (жен, мягкий)",
    "af_bella": "Bella (жен, яркий)",
    "af_sky": "Sky (жен, лёгкий)",
    "af_nicole": "Nicole (жен)",
    "af_sarah": "Sarah (жен)",
    "am_adam": "Adam (муж)",
    "am_michael": "Michael (муж)",
    "bf_emma": "Emma (жен, британ)",
    "bf_isabella": "Isabella (жен, британ)",
    "bm_george": "George (муж, британ)",
    "bm_lewis": "Lewis (муж, британ)",
}

# ========== ОЧИСТКА ТЕКСТА ДЛЯ TTS ==========
def clean_for_tts(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r'```\w*\n?', '', stripped).strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            for key in ["content", "text", "response", "answer", "summary",
                        "output", "result", "message", "translation"]:
                if key in data and isinstance(data[key], str):
                    stripped = data[key]
                    break
            else:
                texts = [v for v in data.values() if isinstance(v, str) and len(v) > 20]
                if texts:
                    stripped = " ".join(texts)
    except (json.JSONDecodeError, TypeError):
        pass
    result = stripped
    result = re.sub(r'```\w*\n?.*?```', '', result, flags=re.DOTALL)
    result = re.sub(r'`[^`]+`', '', result)
    result = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', result)
    result = re.sub(r'^#{1,6}\s+', '', result, flags=re.MULTILINE)
    result = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', result)
    result = re.sub(r'[_]{1,2}([^_]+)[_]{1,2}', r'\1', result)
    result = re.sub(r'^\s*[-*]\s+', '', result, flags=re.MULTILINE)
    result = re.sub(r'^\s*\d+\.\s+', '', result, flags=re.MULTILINE)
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = result.strip()
    logger.info(f"TTS cleanup: {len(text)} -> {len(result)} chars")
    return result

# ========== СИСТЕМНЫЙ ПРОМПТ ДЛЯ ОЗВУЧКИ ==========
VOICE_SYSTEM_PROMPT = """The user wants you to READ ALOUD the provided text.
You MUST:
1. Output ONLY the original text content exactly as written
2. Keep the ORIGINAL language of the text (do NOT translate)
3. Do NOT add commentary, summaries, or explanations
4. Do NOT wrap output in JSON, markdown, or code blocks
5. Just output the plain text ready to be spoken

If the text is in English - output in English.
If the text is in Russian - output in Russian.
Output ONLY the text to be read."""

# ========== TTS ДВИЖКИ ==========

async def tts_edge(text: str, voice: str = "ru-RU-DariyaNeural") -> bytes | None:
    """Озвучить текст через Edge TTS (Microsoft)"""
    if not EDGE_TTS_AVAILABLE:
        logger.error("Edge TTS not installed")
        return None
    try:
        communicate = edge_tts.Communicate(text, voice)
        tmp_path = os.path.join(AUDIO_OUTPUT_PATH, f"_edge_tmp_{uuid.uuid4().hex}.mp3")
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            audio = f.read()
        os.remove(tmp_path)
        if len(audio) > 0:
            logger.info(f"Edge TTS OK: {len(audio)} bytes, voice={voice}")
            return audio
    except Exception as e:
        logger.error(f"Edge TTS error: {e}")
    return None

async def tts_kokoro(text: str, voice: str = "af_heart") -> bytes | None:
    """Озвучить текст через Kokoro"""
    payload = {
        "input": text, "model": "kokoro", "voice": voice,
        "response_format": "mp3", "speed": 1.0, "lang_code": "a"
    }
    try:
        async with httpx.AsyncClient(timeout=300.0, trust_env=False) as client:
            r = await client.post("http://tts-kokoro:8880/v1/audio/speech", json=payload)
            if r.status_code == 200 and len(r.content) > 0:
                logger.info(f"Kokoro TTS OK: {len(r.content)} bytes, voice={voice}")
                return r.content
            logger.error(f"Kokoro TTS: status {r.status_code}")
    except Exception as e:
        logger.error(f"Kokoro TTS error: {e}")
    return None

def is_edge_voice(voice: str) -> bool:
    """Проверяет, является ли голос Edge TTS голосом"""
    return voice.startswith("ru-") or voice.startswith("en-") or "Neural" in voice

async def tts_chunk(text: str, chunk_idx: int, voice: str = "af_heart") -> bytes | None:
    """Универсальный TTS: выбирает движок по голосу и языку"""
    lang = detect_language(text)

    # Если выбран Edge-голос — используем Edge
    if is_edge_voice(voice):
        logger.info(f"  Chunk {chunk_idx}: Edge TTS ({voice}, {len(text)} chars)")
        audio = await tts_edge(text, voice)
        if audio:
            return audio
        # Фоллбэк на Kokoro если Edge упал
        logger.warning(f"  Chunk {chunk_idx}: Edge failed, trying Kokoro")
        return await tts_kokoro(text, "af_heart")

    # Если выбран Kokoro-голос
    if lang == "ru" and EDGE_TTS_AVAILABLE:
        # Русский текст + Kokoro-голос → принудительно Edge (Kokoro не понимает русский)
        logger.info(f"  Chunk {chunk_idx}: Russian text → Edge TTS (ru-RU-DariyaNeural, {len(text)} chars)")
        audio = await tts_edge(text, "ru-RU-DariyaNeural")
        if audio:
            return audio

    # Английский текст или Edge недоступен → Kokoro
    logger.info(f"  Chunk {chunk_idx}: Kokoro ({voice}, {len(text)} chars)")
    return await tts_kokoro(text, voice)

# ========== АУДИОКНИГА: ФОНОВЫЕ ЗАДАЧИ ==========
book_jobs: dict[str, dict] = {}

def split_text_to_chunks(text: str, max_chars: int = 3000) -> list[str]:
    """Разбивает текст на части, уважая границы абзацев и предложений"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                current_chunk += sentence + " "
        elif len(current_chunk) + len(para) + 2 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
        else:
            current_chunk += para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

async def process_book(job_id: str, text: str, voice: str = "af_heart"):
    """Фоновая задача: разбить текст → озвучить → склеить"""
    job = book_jobs[job_id]
    try:
        chunks = split_text_to_chunks(text)
        job["total_chunks"] = len(chunks)
        job["status"] = "processing"
        total_chars = sum(len(c) for c in chunks)
        lang = detect_language(text)
        logger.info(f"Book [{job_id[:8]}]: {len(chunks)} chunks, {total_chars} chars, lang={lang}, voice={voice}")

        audio_parts: list[bytes] = []
        for i, chunk in enumerate(chunks):
            job["current_chunk"] = i + 1
            job["progress"] = round((i / len(chunks)) * 100)

            audio = await tts_chunk(chunk, i + 1, voice=voice)
            if audio:
                audio_parts.append(audio)
            else:
                job["errors"] = job.get("errors", 0) + 1
                logger.warning(f"  Chunk {i+1} failed, skipping")

            await asyncio.sleep(0.3)

        if not audio_parts:
            job["status"] = "failed"
            job["error"] = "All chunks failed"
            return

        fname = f"book_{job_id}.mp3"
        fpath = os.path.join(AUDIO_OUTPUT_PATH, fname)
        with open(fpath, "wb") as f:
            for part in audio_parts:
                f.write(part)

        total_size = os.path.getsize(fpath)
        job["status"] = "done"
        job["progress"] = 100
        job["filename"] = fname
        job["url"] = f"{AUDIO_PUBLIC_URL}/{fname}"
        job["size_mb"] = round(total_size / 1024 / 1024, 1)
        job["chunks_ok"] = len(audio_parts)
        job["finished_at"] = time.time()

        logger.info(f"Book [{job_id[:8]}]: DONE! {fname} ({job['size_mb']} MB)")

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        logger.error(f"Book [{job_id[:8]}]: FAILED: {e}")

# ========== КОНФИГ И МОДЕЛИ ==========
def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Config fail: {e}")
        return {"model_list": []}

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False

@app.get("/v1/models")
async def list_models():
    config = load_config()
    return {"object": "list", "data": [{"id": m["model_name"], "object": "model"} for m in config.get("model_list", [])]}

# ========== ЭНДПОИНТ: АУДИОКНИГА ==========
@app.post("/v1/audio/book")
async def create_audiobook(
    file: UploadFile = File(...),
    voice: str = Form(default="auto")
):
    """Загрузить текстовый файл → получить аудиокнигу"""
    content = await file.read()

    for encoding in ["utf-8", "utf-8-sig", "cp1251", "latin-1"]:
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise HTTPException(status_code=400, detail="Cannot decode file")

    if len(text.strip()) < 10:
        raise HTTPException(status_code=400, detail="File too short")

    # Auto-detect: русский текст → Edge TTS, английский → Kokoro
    if voice == "auto":
        lang = detect_language(text)
        voice = "ru-RU-DariyaNeural" if lang == "ru" else "af_heart"
        logger.info(f"Auto-detected language: {lang}, voice: {voice}")

    job_id = uuid.uuid4().hex
    book_jobs[job_id] = {
        "status": "queued",
        "filename_original": file.filename,
        "total_chars": len(text),
        "total_chunks": 0,
        "current_chunk": 0,
        "progress": 0,
        "voice": voice,
        "created_at": time.time(),
    }

    asyncio.create_task(process_book(job_id, text, voice))

    logger.info(f"Book job created: {job_id[:8]} ({file.filename}, {len(text)} chars, voice={voice})")

    return {
        "job_id": job_id,
        "status": "queued",
        "filename": file.filename,
        "total_chars": len(text),
        "voice": voice,
        "check_progress": f"/v1/audio/book/{job_id}"
    }

@app.get("/v1/audio/book/{job_id}")
async def get_audiobook_status(job_id: str):
    if job_id not in book_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return book_jobs[job_id]

@app.get("/v1/audio/books")
async def list_audiobook_jobs():
    return {
        "jobs": [
            {"job_id": jid, "status": j["status"], "progress": j.get("progress", 0),
             "filename": j.get("filename_original", ""), "url": j.get("url")}
            for jid, j in book_jobs.items()
        ]
    }

@app.get("/v1/audio/voices")
async def list_voices():
    """Комбинированный список голосов: Kokoro + Edge TTS"""
    voices = {}

    # Kokoro голоса
    try:
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            r = await client.get("http://tts-kokoro:8880/v1/audio/voices")
            if r.status_code == 200:
                data = r.json()
                kokoro_list = data if isinstance(data, list) else data.get("voices", [])
                for v in kokoro_list:
                    vid = v if isinstance(v, str) else v.get("id", v.get("name", ""))
                    if vid:
                        voices[vid] = {"id": vid, "label": KOKORO_VOICES.get(vid, vid), "engine": "kokoro", "lang": "en"}
    except Exception:
        for vid, label in KOKORO_VOICES.items():
            voices[vid] = {"id": vid, "label": label, "engine": "kokoro", "lang": "en"}

    # Edge TTS русские голоса
    if EDGE_TTS_AVAILABLE:
        for vid, label in EDGE_VOICES.items():
            voices[vid] = {"id": vid, "label": label, "engine": "edge", "lang": "ru"}

    # Автовыбор
    voices["auto"] = {"id": "auto", "label": "Авто (определяет язык)", "engine": "auto", "lang": "auto"}

    return {"voices": voices}

# ========== TTS ДЛЯ ЧАТА ==========
async def generate_audio(text: str) -> str:
    if not text.strip():
        return ""
    clean_text = clean_for_tts(text)
    if not clean_text:
        logger.error("TTS: text empty after cleanup")
        return ""

    lang = detect_language(clean_text)

    # Русский → Edge TTS, Английский → Kokoro
    if lang == "ru" and EDGE_TTS_AVAILABLE:
        logger.info(f"Chat TTS: Edge (russian, {len(clean_text)} chars)")
        audio = await tts_edge(clean_text, "ru-RU-DariyaNeural")
        if audio:
            fname = f"voice_{uuid.uuid4().hex}.mp3"
            with open(os.path.join(AUDIO_OUTPUT_PATH, fname), "wb") as f:
                f.write(audio)
            return f"{AUDIO_PUBLIC_URL}/{fname}"

    # Английский или Edge недоступен → Kokoro
    logger.info(f"Chat TTS: Kokoro (english, {len(clean_text)} chars)")
    kokoro_payload = {
        "input": clean_text, "model": "kokoro", "voice": "af_heart",
        "response_format": "mp3", "speed": 1.0, "lang_code": "a"
    }
    try:
        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            r = await client.post("http://tts-kokoro:8880/v1/audio/speech", json=kokoro_payload)
            if r.status_code == 200 and len(r.content) > 0:
                fname = f"voice_{uuid.uuid4().hex}.mp3"
                with open(os.path.join(AUDIO_OUTPUT_PATH, fname), "wb") as f:
                    f.write(r.content)
                return f"{AUDIO_PUBLIC_URL}/{fname}"
    except Exception as e:
        logger.error(f"Kokoro chat TTS error: {e}")

    logger.error("TTS: all backends failed")
    return ""

# ========== LLM С РОТАЦИЕЙ КЛЮЧЕЙ ==========
async def call_with_key_rotation(mid, messages, api_key_env, api_base, stream):
    keys = get_all_keys(api_key_env)
    if not keys:
        raise ValueError(f"No API keys for {api_key_env}")
    start_idx = KEY_INDEX.get(api_key_env, 0) % len(keys) if keys else 0
    last_error = None
    for attempt in range(len(keys)):
        idx = (start_idx + attempt) % len(keys)
        key = keys[idx]
        key_label = f"key_{idx+1}/{len(keys)}"
        try:
            logger.info(f"Calling {mid} [{key_label}]")
            resp = await acompletion(
                model=mid, messages=messages,
                api_key=key, api_base=api_base, stream=stream
            )
            KEY_INDEX[api_key_env] = idx + 1
            return resp
        except Exception as e:
            last_error = e
            is_rate_limit = "429" in str(e) or "RateLimitError" in type(e).__name__
            if is_rate_limit and attempt < len(keys) - 1:
                logger.warning(f"429 on {mid} [{key_label}], rotating...")
                continue
            else:
                raise
    raise last_error

# ========== ЧАТ ==========
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    config = load_config()
    last_msg = str(request.messages[-1].get("content", ""))
    voice_intent = "озвучь" in last_msg.lower()
    logger.info(f"REQUEST model={request.model} voice={voice_intent} msg={last_msg[:80]}")

    def get_params(name):
        for item in config.get("model_list", []):
            if item.get("model_name") == name: return item.get("litellm_params")
        return None

    target = get_params(request.model) or get_params("primary_reasoning")
    if not target:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' not found")

    queue = [target] + [get_params(fb) for fb in target.get("fallbacks", []) if get_params(fb)]

    messages = list(request.messages)
    if voice_intent:
        messages.insert(0, {"role": "system", "content": VOICE_SYSTEM_PROMPT})
        logger.info("Voice mode: injected system prompt for raw text output")

    for current in queue:
        mid = current.get("model")
        try:
            resp = await call_with_key_rotation(
                mid=mid, messages=messages,
                api_key_env=current.get("api_key_env", ""),
                api_base=current.get("api_base"),
                stream=request.stream and not voice_intent
            )
            if voice_intent:
                txt = resp.choices[0].message.content
                logger.info(f"Voice mode: model returned {len(txt)} chars")
                logger.info(f"Voice preview: {txt[:100]}")
                url = await generate_audio(txt)
                if url:
                    resp.choices[0].message.content += f"\n\n🎧 **Озвучка:** [Скачать аудио]({url})"
                else:
                    resp.choices[0].message.content += "\n\n⚠️ Не удалось сгенерировать аудио"
                    logger.error("No audio generated")
                return resp
            if request.stream:
                async def gen():
                    async for chunk in resp: yield f"data: {chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(gen(), media_type='text/event-stream')
            return resp
        except Exception as e:
            logger.error(f"Fail on {mid}: {e}")
            continue
    raise HTTPException(status_code=502)


