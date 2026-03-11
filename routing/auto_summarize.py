"""
Auto-Summarize Module
Automatically compresses project history into Strategic and Tactical contexts.

Strategic context: long-term project DNA (summarizes last 50 generations)
   "This project is about minimalist brand design with neon accents"

Tactical context: short-term working memory (summarizes last 10 generations)
   "Designer is currently working on product cards with glassmorphism style"
"""
import asyncio
import os
import logging
from litellm import acompletion
from db import db

logger = logging.getLogger("AI-Router.Summarize")

# How often to trigger summarization
TACTICAL_EVERY = 5     # Every 5 new generations
STRATEGIC_EVERY = 20   # Every 20 new generations
SUMMARY_MODEL = "gemini/gemini-2.0-flash"
FALLBACK_MODEL = "openai/Qwen/Qwen3-8B"
FALLBACK_API_BASE = "https://api.siliconflow.com/v1"

TACTICAL_PROMPT = """You are a design project analyst. Analyze the last few AI image generation prompts
and create a SHORT tactical summary (3-5 sentences) of what the designer is currently working on.

Focus on:
- Current visual style and mood
- Specific elements being designed (cards, headers, icons, etc.)
- Color palette and typography trends
- Any recurring patterns or preferences

Prompts (most recent first):
{prompts}

Write the tactical summary in English. Be concise and specific."""

STRATEGIC_PROMPT = """You are a design project analyst. Analyze the full history of AI image generation prompts
and create a COMPREHENSIVE strategic summary (5-10 sentences) of the project's design DNA.

Focus on:
- Overall brand identity and visual language
- Consistent color schemes and palettes
- Typography and layout patterns
- Target audience and mood
- Design evolution over time
- Key decisions and style pivots

Prompts history (most recent first):
{prompts}

Write the strategic summary in English. Be thorough but focused."""


async def _call_llm(system_prompt: str, prompts_text: str, retries: int = 3) -> dict:
    """Call LLM to generate a summary with retry logic for rate limits."""
    for attempt in range(retries):
        try:
            response = await acompletion(
                model=SUMMARY_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please analyze and summarize:\n{prompts_text}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            text = response.choices[0].message.content
            usage = response.usage
            return {
                "text": text,
                "tokens_input": usage.prompt_tokens if usage else 0,
                "tokens_output": usage.completion_tokens if usage else 0,
            }
        except Exception as e:
            if "RateLimit" in type(e).__name__:
                logger.warning(f"⏳ Gemini rate limit, trying SiliconFlow fallback...")
                try:
                    response = await acompletion(
                        model=FALLBACK_MODEL,
                        api_base=FALLBACK_API_BASE,
                        api_key=os.environ.get("SILICONFLOW_API_KEY"),
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Please analyze and summarize:\n{prompts_text}"}
                        ],
                        temperature=0.3,
                        max_tokens=500
                    )
                    text = response.choices[0].message.content
                    usage = response.usage
                    logger.info(f"✅ Fallback to SiliconFlow succeeded!")
                    return {
                        "text": text,
                        "tokens_input": usage.prompt_tokens if usage else 0,
                        "tokens_output": usage.completion_tokens if usage else 0,
                    }
                except Exception as e2:
                    logger.error(f"❌ Fallback also failed: {repr(e2)}")
                    return None
            else:
                logger.error(f"❌ LLM summarization failed: {repr(e)}")
                return None


def _format_prompts(prompts: list) -> str:
    """Format prompts list into readable text for LLM."""
    lines = []
    for p in prompts:
        seq = p.get("seq_num", "?")
        text = p.get("prompt", "")
        status = p.get("status", "")
        lines.append(f"[#{seq}] {text} (status: {status})")
    return "\n".join(lines)


async def maybe_summarize(project_id: str, project_slug: str, current_seq: int):
    """
    Check if it's time to auto-summarize and do it if needed.
    Called after each capture_generation.
    """
    try:
        # Tactical summary: every N generations
        if current_seq % TACTICAL_EVERY == 0:
            logger.info(f"🧠 Tactical summary triggered for {project_slug} (seq={current_seq})")
            prompts = await db.get_recent_prompts(project_id, limit=10)
            if prompts:
                prompts_text = _format_prompts(prompts)
                result = await _call_llm(
                    TACTICAL_PROMPT.format(prompts=prompts_text),
                    prompts_text
                )
                if result:
                    await db.save_context_summary(
                        project_id=project_id,
                        context_type="tactical",
                        summary_text=result["text"],
                        gen_from_seq=max(1, current_seq - 9),
                        gen_to_seq=current_seq,
                        gen_count=len(prompts),
                        model_used=SUMMARY_MODEL,
                        tokens_input=result["tokens_input"],
                        tokens_output=result["tokens_output"]
                    )
                    logger.info(f"✅ Tactical summary saved for {project_slug}")

        # Strategic summary: every M generations
        if current_seq % STRATEGIC_EVERY == 0:
            logger.info(f"🧠 Strategic summary triggered for {project_slug} (seq={current_seq})")
            prompts = await db.get_recent_prompts(project_id, limit=50)
            if prompts:
                prompts_text = _format_prompts(prompts)
                result = await _call_llm(
                    STRATEGIC_PROMPT.format(prompts=prompts_text),
                    prompts_text
                )
                if result:
                    await db.save_context_summary(
                        project_id=project_id,
                        context_type="strategic",
                        summary_text=result["text"],
                        gen_from_seq=max(1, current_seq - 49),
                        gen_to_seq=current_seq,
                        gen_count=len(prompts),
                        model_used=SUMMARY_MODEL,
                        tokens_input=result["tokens_input"],
                        tokens_output=result["tokens_output"]
                    )
                    logger.info(f"✅ Strategic summary saved for {project_slug}")

    except Exception as e:
        logger.error(f"❌ Auto-summarize error: {repr(e)}")
