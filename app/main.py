"""Application entry point — wires all components together."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import LLMProvider, Settings, get_settings
from app.api.http_routes import router as http_router
from app.api.ws_routes import router as ws_router
from app.middleware.rate_limiter import RateLimiter
from app.middleware.throttling import ThrottlingMiddleware
from app.middleware.validation import PayloadValidator
from app.services.audio_ingestion import AudioIngestionService
from app.services.context_compressor import ContextCompressor
from app.services.event_dispatcher import EventDispatcher
from app.services.llm_orchestrator import LLMOrchestrator
from app.services.session_manager import SessionManager
from app.services.stt_service import STTService
from app.services.tool_executor import ToolExecutor
from app.services.tts_service import TTSService
from app.state_machine import AgentPipeline

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Provider factory
# ------------------------------------------------------------------

def _build_llm_provider(s: Settings):
    if s.LLM_PROVIDER == LLMProvider.OPENAI:
        from app.providers.openai_provider import OpenAILLM
        return OpenAILLM(api_key=s.OPENAI_API_KEY or "", model=s.LLM_MODEL)
    if s.LLM_PROVIDER == LLMProvider.ANTHROPIC:
        from app.providers.anthropic_provider import AnthropicLLM
        return AnthropicLLM(api_key=s.ANTHROPIC_API_KEY or "", model=s.LLM_MODEL)
    if s.LLM_PROVIDER == LLMProvider.GEMINI:
        from app.providers.gemini_provider import GeminiLLM
        return GeminiLLM(api_key=s.GEMINI_API_KEY or "", model=s.LLM_MODEL)
    raise ValueError(f"Unsupported LLM provider: {s.LLM_PROVIDER}")


def _build_stt_provider(s: Settings):
    if s.STT_PROVIDER == LLMProvider.OPENAI:
        from app.providers.openai_provider import OpenAISTT
        return OpenAISTT(api_key=s.OPENAI_API_KEY or "", model=s.STT_MODEL)
    if s.STT_PROVIDER == LLMProvider.GEMINI:
        from app.providers.gemini_provider import GeminiSTT
        return GeminiSTT(api_key=s.GEMINI_API_KEY or "", model=s.STT_MODEL)
    # Anthropic has no STT — fallback to OpenAI
    from app.providers.openai_provider import OpenAISTT
    return OpenAISTT(api_key=s.OPENAI_API_KEY or "", model=s.STT_MODEL)


def _build_summary_provider(s: Settings):
    """Lightweight LLM provider used only for context compression summaries."""
    if s.MAX_CONTEXT_MESSAGES == 0:
        # Compression disabled — return a no-op provider
        from app.providers.base import BaseLLMProvider, LLMResponse

        class _NoOpLLM(BaseLLMProvider):
            async def chat(self, **_):
                return LLMResponse()
            async def chat_with_tool_results(self, **_):
                return LLMResponse()

        return _NoOpLLM()

    from app.providers.openai_provider import OpenAILLM
    return OpenAILLM(api_key=s.OPENAI_API_KEY or "", model=s.CONTEXT_SUMMARY_MODEL)


def _build_tts_provider(s: Settings):
    if s.TTS_PROVIDER == LLMProvider.OPENAI:
        from app.providers.openai_provider import OpenAITTS
        return OpenAITTS(api_key=s.OPENAI_API_KEY or "", model=s.TTS_MODEL, voice=s.TTS_VOICE)
    if s.TTS_PROVIDER == LLMProvider.GEMINI:
        from app.providers.gemini_provider import GeminiTTS
        return GeminiTTS(api_key=s.GEMINI_API_KEY or "", model=s.TTS_MODEL)
    # Anthropic has no TTS — fallback to OpenAI
    from app.providers.openai_provider import OpenAITTS
    return OpenAITTS(api_key=s.OPENAI_API_KEY or "", model=s.TTS_MODEL, voice=s.TTS_VOICE)


# ------------------------------------------------------------------
# Background tasks
# ------------------------------------------------------------------

async def _session_cleanup_loop(session_mgr: SessionManager) -> None:
    while True:
        await asyncio.sleep(60)
        try:
            await session_mgr.cleanup_expired()
        except Exception:
            logger.exception("Session cleanup error")


# ------------------------------------------------------------------
# Application factory
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    session_mgr: SessionManager = app.state.deps["session_manager"]

    cleanup_task = asyncio.create_task(_session_cleanup_loop(session_mgr))
    logger.info(
        "Agent started — provider=%s  model=%s  audio_support=%s",
        settings.LLM_PROVIDER.value,
        settings.LLM_MODEL,
        settings.LLM_SUPPORTS_AUDIO,
    )
    yield
    cleanup_task.cancel()
    logger.info("Agent shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="Multimodal Conversational Agent",
        version="1.0.0",
        lifespan=lifespan,
    )

    # -- Build all dependencies --
    dispatcher = EventDispatcher()
    session_mgr = SessionManager(settings)
    audio_svc = AudioIngestionService(settings)
    validator = PayloadValidator(settings)
    tool_executor = ToolExecutor(settings)

    stt_provider = _build_stt_provider(settings)
    stt_service = STTService(settings, stt_provider)

    llm_provider = _build_llm_provider(settings)
    llm_orchestrator = LLMOrchestrator(settings, llm_provider)

    tts_provider = _build_tts_provider(settings)
    tts_service = TTSService(settings, tts_provider)

    summary_provider = _build_summary_provider(settings)
    context_compressor = ContextCompressor(settings, summary_provider)

    pipeline = AgentPipeline(
        settings=settings,
        dispatcher=dispatcher,
        stt=stt_service,
        llm=llm_orchestrator,
        tool_executor=tool_executor,
        tts=tts_service,
        context_compressor=context_compressor,
    )

    app.state.settings = settings
    app.state.deps = {
        "session_manager": session_mgr,
        "dispatcher": dispatcher,
        "audio_ingestion": audio_svc,
        "validator": validator,
        "tool_executor": tool_executor,
        "pipeline": pipeline,
        "stt": stt_service,
        "llm": llm_orchestrator,
        "tts": tts_service,
    }

    # -- Middleware --
    rate_limiter = RateLimiter(
        requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
        burst=settings.RATE_LIMIT_BURST,
    )
    app.add_middleware(ThrottlingMiddleware, rate_limiter=rate_limiter)

    # -- Routes --
    app.include_router(http_router)
    app.include_router(ws_router)

    return app


# Default app instance for `uvicorn app.main:app`
app = create_app()
