import os, json, logging, httpx, uuid, re, asyncio, time, struct, subprocess
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from litellm import acompletion
from fastapi.middleware.cors import CORSMiddleware
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import db
from api_dna import router as dna_router

# Silero TTS: set cache dir BEFORE importing torch (fix Permission denied)
os.environ['TORCH_HOME'] = '/tmp/torch_cache'

try:
    import torch
    import numpy as np
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("AI-Router")

app = FastAPI(title="Project DNA Router")

# CORS: allow Dashboard (port 8090) to call API (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(dna_router)
AUDIO_OUTPUT_PATH = "/app/audio_out"
AUDIO_PUBLIC_URL = os.getenv("AUDIO_PUBLIC_URL", "http://172.25.9.33:8090")
os.makedirs(AUDIO_OUTPUT_PATH, exist_ok=True)

# ========== DATABASE LIFECYCLE ==========
@app.on_event("startup")
async def startup_db():
    await db.connect()

@app.on_event("shutdown")
async def shutdown_db():
    await db.disconnect()

CONFIG_PATH = os.getenv("CONFIG_PATH", "deploy/antigravity.json")

# ========== SILERO TTS MODEL ==========
silero_model = None

def init_silero():
    global silero_model
    if not SILERO_AVAILABLE:
        logger.warning("Silero TTS: torch not available")
        return
    try:
        silero_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts', language='ru', speaker='v5_ru'
        )
        logger.info(f"Silero TTS loaded! Voices: {silero_model.speakers}")
    except Exception as e:
        logger.error(f"Silero TTS init error: {e}")

init_silero()
logger.info(f"Silero TTS: {'loaded' if silero_model else 'NOT available (pip install torch numpy)'}")

# ========== SALUTESPEECH TTS (SBER) ==========
SALUTE_AUTH_KEY = os.getenv("SALUTE_SPEECH_KEY", "")
SALUTE_TOKEN = {"access_token": "", "expires_at": 0}

async def salute_get_token() -> str:
    """Get or refresh SaluteSpeech access token (valid 30 min)"""
    global SALUTE_TOKEN
    now_ms = int(time.time() * 1000)
    # Refresh if less than 60 seconds remaining
    if SALUTE_TOKEN["access_token"] and SALUTE_TOKEN["expires_at"] - now_ms > 60000:
        return SALUTE_TOKEN["access_token"]

    if not SALUTE_AUTH_KEY:
        logger.error("SaluteSpeech: SALUTE_SPEECH_KEY not set")
        return ""

    try:
        async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
            r = await client.post(
                "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "RqUID": str(uuid.uuid4()),
                    "Authorization": f"Basic {SALUTE_AUTH_KEY}",
                },
                data={"scope": "SALUTE_SPEECH_PERS"},
            )
            if r.status_code == 200:
                data = r.json()
                SALUTE_TOKEN["access_token"] = data["access_token"]
                SALUTE_TOKEN["expires_at"] = data["expires_at"]
                logger.info("SaluteSpeech: token refreshed OK")
                return data["access_token"]
            else:
                logger.error(f"SaluteSpeech token error: {r.status_code} {r.text[:200]}")
    except Exception as e:
        logger.error(f"SaluteSpeech token error: {e}")
    return ""

SALUTE_AVAILABLE = bool(SALUTE_AUTH_KEY)
if SALUTE_AVAILABLE:
    logger.info("SaluteSpeech: configured (key found)")
else:
    logger.info("SaluteSpeech: NOT configured (set SALUTE_SPEECH_KEY)")

# ========== CHATTERBOX TTS (PREMIUM ENGLISH) ==========
CHATTERBOX_URL = os.getenv("CHATTERBOX_URL", "http://tts-chatterbox:4123")
CHATTERBOX_AVAILABLE = False

async def check_chatterbox():
    """Check if Chatterbox TTS is available"""
    global CHATTERBOX_AVAILABLE
    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            r = await client.get(f"{CHATTERBOX_URL}/health")
            if r.status_code == 200:
                data = r.json()
                CHATTERBOX_AVAILABLE = data.get("model_loaded", False)
                logger.info(f"Chatterbox TTS: {'ready' if CHATTERBOX_AVAILABLE else 'model loading...'}")
            else:
                logger.warning(f"Chatterbox TTS: health check failed ({r.status_code})")
    except Exception as e:
        logger.warning(f"Chatterbox TTS: not available ({e})")

# Schedule startup check
@app.on_event("startup")
async def startup_check_chatterbox():
    await check_chatterbox()

CHATTERBOX_VOICES = {
    "chatterbox": {"label": "Chatterbox Default (expressive)", "exaggeration": 0.6, "cfg_weight": 0.5, "temperature": 0.8, "speed": 0.85},
    "chatterbox-expressive": {"label": "Chatterbox Expressive (dramatic)", "exaggeration": 1.2, "cfg_weight": 0.3, "temperature": 0.9, "speed": 0.85},
    "chatterbox-calm": {"label": "Chatterbox Calm (narration)", "exaggeration": 0.3, "cfg_weight": 0.7, "temperature": 0.7, "speed": 0.82},
    "chatterbox-storyteller": {"label": "Chatterbox Storyteller (audiobooks)", "exaggeration": 0.8, "cfg_weight": 0.5, "temperature": 0.85, "speed": 0.85},
}

# ========== KEY ROTATION ==========
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

# ========== LANGUAGE DETECTION ==========
def has_cyrillic(text: str) -> bool:
    """Check if text contains Cyrillic characters"""
    return bool(re.search('[\u0430-\u044f\u0410-\u042f\u0451\u0401]', text))

def detect_language(text: str) -> str:
    """Detect language: ru or en"""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    cyrillic = sum(1 for c in letters if re.match('[\u0430-\u044f\u0410-\u042f\u0451\u0401]', c))
    ratio = cyrillic / len(letters)
    return "ru" if ratio > 0.3 else "en"

# ========== VOICE DEFINITIONS ==========
SALUTE_VOICES = {
    "salute-may": {"label": "Марфа (female, warm)", "model": "May_24000"},
    "salute-nec": {"label": "Наталья (female, clear)", "model": "Nec_24000"},
    "salute-bys": {"label": "Борис (male, confident)", "model": "Bys_24000"},
    "salute-tur": {"label": "Тур (male, deep)", "model": "Tur_24000"},
    "salute-ost": {"label": "Александра (female, neutral)", "model": "Ost_24000"},
    "salute-pon": {"label": "Сергей (male, friendly)", "model": "Pon_24000"},
    "salute-kin": {"label": "Кира (female, bright)", "model": "Kin_24000"},
}

SILERO_VOICES = {
    "silero-xenia": "Xenia (female, clear)",
    "silero-baya": "Baya (female, soft)",
    "silero-kseniya": "Kseniya-2 (female)",
    "silero-aidar": "Aidar (male)",
    "silero-eugene": "Eugene (male)",
}

KOKORO_VOICES = {
    "af_heart": "Heart (female, soft)",
    "af_bella": "Bella (female, bright)",
    "af_sky": "Sky (female, light)",
    "af_nicole": "Nicole (female)",
    "af_sarah": "Sarah (female)",
    "am_adam": "Adam (male)",
    "am_michael": "Michael (male)",
    "bf_emma": "Emma (female, British)",
    "bf_isabella": "Isabella (female, British)",
    "bm_george": "George (male, British)",
    "bm_lewis": "Lewis (male, British)",
}

# ========== TEXT CLEANUP FOR TTS ==========
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

# ========== VOICE SYSTEM PROMPT ==========
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

# ========== TTS ENGINES ==========

def audio_adjust_speed(audio_bytes: bytes, speed: float = 1.0) -> bytes:
    """Adjust audio playback speed using ffmpeg atempo filter.
    Speed 0.85 = 15% slower (more comfortable for audiobooks).
    Processes in memory via pipes — adds <0.1s overhead.
    """
    if speed == 1.0 or not audio_bytes:
        return audio_bytes
    try:
        proc = subprocess.run(
            ['ffmpeg', '-i', 'pipe:0', '-filter:a', f'atempo={speed}',
             '-f', 'wav', '-y', 'pipe:1'],
            input=audio_bytes, capture_output=True, timeout=30
        )
        if proc.returncode == 0 and len(proc.stdout) > 1000:
            logger.info(f"Speed adjust: {speed}x, {len(audio_bytes)} -> {len(proc.stdout)} bytes")
            return proc.stdout
        else:
            logger.warning(f"ffmpeg speed adjust failed (rc={proc.returncode}), using original")
    except FileNotFoundError:
        logger.warning("ffmpeg not found, skipping speed adjustment (apt-get install ffmpeg)")
    except Exception as e:
        logger.warning(f"Speed adjust error: {e}, using original")
    return audio_bytes

def wav_to_mp3_raw(wav_data: bytes) -> bytes:
    """Simple WAV wrapper - returns as is (MP3 conversion optional)"""
    return wav_data

def tensor_to_wav(audio_tensor, sample_rate: int = 48000) -> bytes:
    """Convert torch tensor to WAV bytes"""
    audio_np = audio_tensor.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    # WAV header
    num_samples = len(audio_int16)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size
    )
    return header + audio_int16.tobytes()

async def tts_salute(text: str, voice: str = "salute-may") -> bytes | None:
    """Generate speech using SaluteSpeech (Sber Cloud API)"""
    if not SALUTE_AVAILABLE:
        return None

    token = await salute_get_token()
    if not token:
        return None

    # Get model name from voice id
    voice_info = SALUTE_VOICES.get(voice)
    model_name = voice_info["model"] if voice_info else "May_24000"

    try:
        async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
            r = await client.post(
                f"https://smartspeech.sber.ru/rest/v1/text:synthesize?format=wav16&voice={model_name}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/text",
                },
                content=text.encode("utf-8"),
            )
            if r.status_code == 200 and len(r.content) > 0:
                logger.info(f"SaluteSpeech TTS OK: {len(r.content)} bytes, voice={model_name}")
                return r.content
            else:
                logger.error(f"SaluteSpeech TTS error: {r.status_code} {r.text[:200]}")
    except Exception as e:
        logger.error(f"SaluteSpeech TTS error: {e}")
    return None

async def tts_silero(text: str, voice: str = "xenia") -> bytes | None:
    """Generate speech using Silero TTS (local model, fallback)"""
    if silero_model is None:
        logger.error("Silero TTS not loaded")
        return None
    # Extract speaker name from voice id (silero-xenia -> xenia)
    speaker = voice.replace("silero-", "") if voice.startswith("silero-") else voice
    if speaker not in silero_model.speakers:
        speaker = "xenia"  # default Russian voice
    try:
        audio = silero_model.apply_tts(
            text=text, speaker=speaker, sample_rate=48000
        )
        wav_bytes = tensor_to_wav(audio, 48000)
        logger.info(f"Silero TTS OK: {len(wav_bytes)} bytes, speaker={speaker}")
        return wav_bytes
    except Exception as e:
        logger.error(f"Silero TTS error: {e}")
    return None

async def tts_chatterbox(text: str, voice: str = "chatterbox", max_retries: int = 3) -> bytes | None:
    """Generate speech using Chatterbox TTS (premium English, OpenAI-compatible API)
    Applies speed adjustment via ffmpeg atempo post-processing.
    Includes retry logic with health checks for resilience.
    """
    global CHATTERBOX_AVAILABLE
    if not CHATTERBOX_AVAILABLE:
        await check_chatterbox()
        if not CHATTERBOX_AVAILABLE:
            return None

    voice_settings = CHATTERBOX_VOICES.get(voice, CHATTERBOX_VOICES["chatterbox"])
    payload = {
        "input": text,
        "exaggeration": voice_settings["exaggeration"],
        "cfg_weight": voice_settings["cfg_weight"],
        "temperature": voice_settings["temperature"],
    }

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=1200.0, trust_env=False) as client:
                r = await client.post(f"{CHATTERBOX_URL}/v1/audio/speech", json=payload)
                if r.status_code == 200 and len(r.content) > 1000:
                    logger.info(f"Chatterbox TTS OK: {len(r.content)} bytes, voice={voice}, attempt={attempt+1}")
                    speed = voice_settings.get("speed", 1.0)
                    audio = audio_adjust_speed(r.content, speed)
                    return audio
                else:
                    error_msg = r.text[:200] if r.status_code != 200 else f"content too small ({len(r.content)} bytes)"
                    logger.error(f"Chatterbox TTS error: {r.status_code} {error_msg}")
        except Exception as e:
            logger.error(f"Chatterbox TTS error (attempt {attempt+1}/{max_retries}): {repr(e)}")

        # Retry logic: wait, re-check health, try again
        if attempt < max_retries - 1:
            wait = 30 * (attempt + 1)  # 30s, 60s
            logger.warning(f"Chatterbox: retrying in {wait}s (attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(wait)
            # Re-check if Chatterbox is still alive
            await check_chatterbox()
            if not CHATTERBOX_AVAILABLE:
                logger.warning(f"Chatterbox is DOWN, waiting extra 60s for container recovery...")
                await asyncio.sleep(60)
                await check_chatterbox()
                if not CHATTERBOX_AVAILABLE:
                    logger.error(f"Chatterbox still DOWN after recovery wait, giving up")
                    return None

    return None

async def tts_kokoro(text: str, voice: str = "af_heart") -> bytes | None:
    """Generate speech using Kokoro"""
    payload = {
        "input": text, "model": "kokoro", "voice": voice,
        "response_format": "mp3", "speed": 1.0, "lang_code": "a"
    }
    try:
        async with httpx.AsyncClient(timeout=1200.0, trust_env=False) as client:
            r = await client.post("http://tts-kokoro:8880/v1/audio/speech", json=payload)
            if r.status_code == 200 and len(r.content) > 0:
                logger.info(f"Kokoro TTS OK: {len(r.content)} bytes, voice={voice}")
                return r.content
            logger.error(f"Kokoro TTS: status {r.status_code}")
    except Exception as e:
        logger.error(f"Kokoro TTS error: {e}")
    return None

def is_salute_voice(voice: str) -> bool:
    return voice.startswith("salute-")

def is_silero_voice(voice: str) -> bool:
    return voice.startswith("silero-")

def is_chatterbox_voice(voice: str) -> bool:
    return voice.startswith("chatterbox")

async def tts_chunk(text: str, chunk_idx: int, voice: str = "af_heart") -> bytes | None:
    """Universal TTS: selects engine by voice and language.
    Priority for Russian: SaluteSpeech -> Silero -> Kokoro (fallback)
    Priority for English: Chatterbox -> only Chatterbox
    """
    lang = detect_language(text)

    # If SaluteSpeech voice explicitly selected
    if is_salute_voice(voice):
        logger.info(f"  Chunk {chunk_idx}: SaluteSpeech ({voice}, {len(text)} chars)")
        audio = await tts_salute(text, voice)
        if audio:
            return audio
        logger.warning(f"  Chunk {chunk_idx}: SaluteSpeech failed, trying Silero")
        audio = await tts_silero(text, "xenia")
        if audio:
            return audio
        return await tts_kokoro(text, "af_heart")

    # If Silero voice explicitly selected
    if is_silero_voice(voice):
        logger.info(f"  Chunk {chunk_idx}: Silero TTS ({voice}, {len(text)} chars)")
        audio = await tts_silero(text, voice)
        if audio:
            return audio
        logger.warning(f"  Chunk {chunk_idx}: Silero failed, trying Kokoro")
        return await tts_kokoro(text, "af_heart")

    # If Chatterbox voice explicitly selected
    if is_chatterbox_voice(voice):
        logger.info(f"  Chunk {chunk_idx}: Chatterbox ({voice}, {len(text)} chars)")
        audio = await tts_chatterbox(text, voice)
        if audio:
            return audio
        logger.error(f"  Chunk {chunk_idx}: Chatterbox returned no audio, skipping chunk")
        # Возвращаем None (чанк фейлится, никакого Kokoro)
        return None

    # Auto: Russian text -> SaluteSpeech (best quality) -> Silero (fallback)
    if lang == "ru":
        if SALUTE_AVAILABLE:
            logger.info(f"  Chunk {chunk_idx}: Russian -> SaluteSpeech (May_24000, {len(text)} chars)")
            audio = await tts_salute(text, "salute-may")
            if audio:
                return audio
            logger.warning(f"  Chunk {chunk_idx}: SaluteSpeech failed, trying Silero")

        if silero_model is not None:
            logger.info(f"  Chunk {chunk_idx}: Russian -> Silero (xenia, {len(text)} chars)")
            audio = await tts_silero(text, "xenia")
            if audio:
                return audio

    # English text -> Chatterbox (premium) -> Kokoro (fallback)
    if lang == "en" and CHATTERBOX_AVAILABLE:
        cb_voice = voice if is_chatterbox_voice(voice) else "chatterbox-storyteller"
        logger.info(f"  Chunk {chunk_idx}: English -> Chatterbox ({cb_voice}, {len(text)} chars)")
        audio = await tts_chatterbox(text, cb_voice)
        if audio:
            return audio
        logger.error(f"  Chunk {chunk_idx}: Chatterbox failed, skipping chunk")
        return None

    # Kokoro final fallback
    kokoro_voice = voice if voice in KOKORO_VOICES else "af_heart"
    logger.info(f"  Chunk {chunk_idx}: Kokoro fallback ({kokoro_voice}, {len(text)} chars)")
    return await tts_kokoro(text, kokoro_voice)

# ========== AUDIOBOOK: BACKGROUND TASKS ==========
book_jobs: dict[str, dict] = {}

def split_text_to_chunks(text: str, max_chars: int = 3000) -> list[str]:
    """Split text into chunks respecting paragraph and sentence boundaries"""
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
    """Background task: split text -> TTS each chunk -> concatenate"""
    job = book_jobs[job_id]
    try:
        chunks = split_text_to_chunks(text)
        job["total_chunks"] = len(chunks)
        job["status"] = "processing"
        total_chars = sum(len(c) for c in chunks)
        lang = detect_language(text)
        logger.info(f"Book [{job_id[:8]}]: {len(chunks)} chunks, {total_chars} chars, lang={lang}, voice={voice}")

        audio_parts: list[bytes] = []
        consecutive_failures = 0
        is_heavy_tts = is_chatterbox_voice(voice)  # Chatterbox needs more cooldown

        for i, chunk in enumerate(chunks):
            job["current_chunk"] = i + 1
            job["progress"] = round((i / len(chunks)) * 100)

            audio = await tts_chunk(chunk, i + 1, voice=voice)
            if audio:
                audio_parts.append(audio)
                consecutive_failures = 0  # Reset on success
            else:
                job["errors"] = job.get("errors", 0) + 1
                consecutive_failures += 1
                logger.warning(f"  Chunk {i+1} failed (consecutive: {consecutive_failures})")

                # If 5+ consecutive failures -> TTS container is probably dead, abort
                if consecutive_failures >= 5:
                    logger.error(f"Book [{job_id[:8]}]: {consecutive_failures} consecutive failures! Aborting remaining chunks.")
                    break
                # If 3+ consecutive failures -> wait for recovery
                elif consecutive_failures >= 3:
                    logger.warning(f"Book [{job_id[:8]}]: {consecutive_failures} failures in a row, waiting 120s for TTS recovery...")
                    await asyncio.sleep(120)

            # Cooldown between chunks:
            # Chatterbox on CPU needs 5s to free memory between generations
            # Other TTS engines are lighter and need only 0.5s
            cooldown = 5.0 if is_heavy_tts else 0.5
            await asyncio.sleep(cooldown)

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

# ========== CONFIG AND MODELS ==========
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

# ========== ENDPOINT: AUDIOBOOK ==========
@app.post("/v1/audio/book")
async def create_audiobook(
    file: UploadFile = File(...),
    voice: str = Form(default="auto")
):
    """Upload text file -> get audiobook"""
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

    # Auto-detect: Russian -> SaluteSpeech, English -> Chatterbox
    if voice == "auto":
        lang = detect_language(text)
        if lang == "ru":
            voice = "salute-may" if SALUTE_AVAILABLE else "silero-xenia"
        else:
            voice = "chatterbox-storyteller" if CHATTERBOX_AVAILABLE else "af_heart"
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
    """Combined voice list: Chatterbox + SaluteSpeech + Silero + Kokoro"""
    voices = {}

    # Chatterbox voices (premium English)
    if CHATTERBOX_AVAILABLE:
        for vid, info in CHATTERBOX_VOICES.items():
            voices[vid] = {"id": vid, "label": info["label"], "engine": "chatterbox", "lang": "en"}

    # Kokoro voices (English)
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
    except Exception as e:
        for vid, label in KOKORO_VOICES.items():
            voices[vid] = {"id": vid, "label": label, "engine": "kokoro", "lang": "en"}

    # SaluteSpeech Russian voices (primary)
    if SALUTE_AVAILABLE:
        for vid, info in SALUTE_VOICES.items():
            voices[vid] = {"id": vid, "label": info["label"], "engine": "salutespeech", "lang": "ru"}

    # Silero Russian voices (fallback)
    if silero_model is not None:
        for vid, label in SILERO_VOICES.items():
            voices[vid] = {"id": vid, "label": label, "engine": "silero", "lang": "ru"}

    # Auto selection
    voices["auto"] = {"id": "auto", "label": "Auto (detect language)", "engine": "auto", "lang": "auto"}

    return {"voices": voices}

# ========== TTS FOR CHAT ==========
async def generate_audio(text: str) -> str:
    if not text.strip():
        return ""
    clean_text = clean_for_tts(text)
    if not clean_text:
        logger.error("TTS: text empty after cleanup")
        return ""

    lang = detect_language(clean_text)

    # Russian -> SaluteSpeech (best) -> Silero (fallback)
    if lang == "ru":
        if SALUTE_AVAILABLE:
            logger.info(f"Chat TTS: SaluteSpeech (russian, {len(clean_text)} chars)")
            audio = await tts_salute(clean_text, "salute-may")
            if audio:
                fname = f"voice_{uuid.uuid4().hex}.wav"
                with open(os.path.join(AUDIO_OUTPUT_PATH, fname), "wb") as f:
                    f.write(audio)
                return f"{AUDIO_PUBLIC_URL}/{fname}"

        if silero_model is not None:
            logger.info(f"Chat TTS: Silero fallback (russian, {len(clean_text)} chars)")
            audio = await tts_silero(clean_text, "xenia")
            if audio:
                fname = f"voice_{uuid.uuid4().hex}.wav"
                with open(os.path.join(AUDIO_OUTPUT_PATH, fname), "wb") as f:
                    f.write(audio)
                return f"{AUDIO_PUBLIC_URL}/{fname}"

    # English -> Chatterbox (premium) -> Kokoro (fallback)
    if lang == "en" and CHATTERBOX_AVAILABLE:
        logger.info(f"Chat TTS: Chatterbox (english, {len(clean_text)} chars)")
        audio = await tts_chatterbox(clean_text, "chatterbox")
        if audio:
            fname = f"voice_{uuid.uuid4().hex}.wav"
            with open(os.path.join(AUDIO_OUTPUT_PATH, fname), "wb") as f:
                f.write(audio)
            return f"{AUDIO_PUBLIC_URL}/{fname}"

    # Kokoro final fallback
    logger.info(f"Chat TTS: Kokoro fallback ({len(clean_text)} chars)")
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

# ========== LLM WITH KEY ROTATION ==========
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

# ========== CHAT ==========
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    config = load_config()
    last_msg = str(request.messages[-1].get("content", ""))
    voice_intent = any(w in last_msg.lower() for w in ["voice", "read aloud", "tts"])
    # Also check Russian voice command
    if not voice_intent:
        voice_intent = "\u043e\u0437\u0432\u0443\u0447\u044c" in last_msg.lower()
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
                    resp.choices[0].message.content += f"\n\n\U0001f3a7 **Audio:** [Download audio]({url})"
                else:
                    resp.choices[0].message.content += "\n\n\u26a0\ufe0f Failed to generate audio"
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


