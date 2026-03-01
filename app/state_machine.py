"""
Orchestration engine that drives a session through the agent state machine.

Flow:
  IDLE -> RECEIVING_INPUT -> [TRANSCRIBING] -> THINKING -> [CALLING_TOOL]
       -> RESPONDING_TEXT -> [GENERATING_AUDIO] -> DONE -> IDLE
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

from app.config import Settings
from app.models.events import (
    EventType,
    SttImpact,
    make_event,
)
from app.models.session import AgentState, Session
from app.providers.base import LLMResponse

if TYPE_CHECKING:
    from app.services.context_compressor import ContextCompressor
    from app.services.event_dispatcher import EventDispatcher
    from app.services.stt_service import STTService
    from app.services.llm_orchestrator import LLMOrchestrator
    from app.services.tool_executor import ToolExecutor
    from app.services.tts_service import TTSService

logger = logging.getLogger(__name__)


class AgentPipeline:
    """Drives a single turn through the agent state machine."""

    def __init__(
        self,
        settings: Settings,
        dispatcher: "EventDispatcher",
        stt: "STTService",
        llm: "LLMOrchestrator",
        tool_executor: "ToolExecutor",
        tts: "TTSService",
        context_compressor: "ContextCompressor",
    ) -> None:
        self._settings = settings
        self._dispatcher = dispatcher
        self._stt = stt
        self._llm = llm
        self._tools = tool_executor
        self._tts = tts
        self._compressor = context_compressor

    async def run(self, session: Session) -> None:
        """Execute one full conversational turn."""
        try:
            await self._receive_input(session)
            await self._maybe_transcribe(session)
            await self._compress_context(session)
            await self._think(session)
            await self._respond_text(session)
            await self._maybe_generate_audio(session)
            await self._done(session)
        except _SttBlockingFailure:
            session.state = AgentState.ERROR
        except Exception:
            logger.exception("Pipeline error for session %s", session.id)
            session.state = AgentState.ERROR
            await self._dispatcher.send(
                session.id,
                make_event(EventType.ERROR, session.id, message="Internal error"),
            )

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    async def _receive_input(self, session: Session) -> None:
        session.transition_to(AgentState.RECEIVING_INPUT)
        await self._dispatcher.send(
            session.id,
            make_event(EventType.STATE_CHANGED, session.id, state=AgentState.RECEIVING_INPUT.value),
        )

    async def _maybe_transcribe(self, session: Session) -> None:
        has_audio = session.current_input_audio is not None
        model_supports_audio = self._settings.LLM_SUPPORTS_AUDIO
        show_transcript = self._settings.SHOW_USER_TRANSCRIPT

        needs_cognitive_stt = has_audio and not model_supports_audio
        needs_ux_stt = has_audio and show_transcript

        if not (needs_cognitive_stt or needs_ux_stt):
            return

        session.transition_to(AgentState.TRANSCRIBING)
        await self._dispatcher.send(
            session.id,
            make_event(EventType.STATE_CHANGED, session.id, state=AgentState.TRANSCRIBING.value),
        )

        transcript, error = await self._stt.transcribe(session.current_input_audio)

        if error:
            impact = SttImpact.BLOCKING if needs_cognitive_stt else SttImpact.UX_ONLY
            await self._dispatcher.send(
                session.id,
                make_event(
                    EventType.STT_FAILED,
                    session.id,
                    impact=impact.value,
                    recoverable=impact == SttImpact.UX_ONLY,
                ),
            )
            if needs_cognitive_stt:
                await self._dispatcher.send(
                    session.id,
                    make_event(EventType.REQUEST_AUDIO_RETRY, session.id),
                )
                raise _SttBlockingFailure()
            return

        session.current_transcript = transcript

        if needs_cognitive_stt:
            session.current_input_text = transcript
            session.messages.append({"role": "user", "content": transcript})

        if show_transcript:
            await self._dispatcher.send(
                session.id,
                make_event(EventType.TRANSCRIPTION_FINAL, session.id, text=transcript),
            )

    async def _compress_context(self, session: Session) -> None:
        session.messages = await self._compressor.maybe_compress(session.messages)

    async def _think(self, session: Session) -> None:
        session.transition_to(AgentState.THINKING)
        await self._dispatcher.send(
            session.id,
            make_event(EventType.STATE_CHANGED, session.id, state=AgentState.THINKING.value),
        )

        # Run main LLM
        await self._run_main_llm(session)

    async def _run_main_llm(self, session: Session) -> None:
        """Execute the main LLM call and update session state."""
        input_audio = (
            session.current_input_audio
            if self._settings.LLM_SUPPORTS_AUDIO
            else None
        )

        response = await self._llm.generate(
            messages=session.messages,
            text=None,
            audio=input_audio,
        )

        if response.error:
            await self._emit_safe_response(session)
            return

        if response.tool_calls:
            await self._execute_tools(session, response)
            return

        session.current_response_text = response.text

    async def _emit_safe_response(self, session: Session) -> None:
        await self._dispatcher.send(
            session.id,
            make_event(
                EventType.SAFE_RESPONSE,
                session.id,
                message=self._settings.SAFE_RESPONSE_MESSAGE,
            ),
        )
        session.current_response_text = self._settings.SAFE_RESPONSE_MESSAGE

    async def _execute_tools(self, session: Session, llm_response: LLMResponse) -> None:
        session.transition_to(AgentState.CALLING_TOOL)
        await self._dispatcher.send(
            session.id,
            make_event(EventType.STATE_CHANGED, session.id, state=AgentState.CALLING_TOOL.value),
        )

        tool_results: list[dict[str, Any]] = []
        for call in llm_response.tool_calls:
            result, error = await self._tools.execute(call)
            if error:
                await self._dispatcher.send(
                    session.id,
                    make_event(
                        EventType.TOOL_FAILED,
                        session.id,
                        tool_name=call.get("name", ""),
                        recoverable=True,
                    ),
                )
                tool_results.append({
                    "tool_call_id": call.get("id", ""),
                    "name": call.get("name", ""),
                    "result": "",
                    "error": str(error),
                })
            else:
                tool_results.append({
                    "tool_call_id": call.get("id", ""),
                    "name": call.get("name", ""),
                    "result": result,
                    "error": None,
                })

        session.transition_to(AgentState.THINKING)
        followup = await self._llm.generate_with_tool_results(
            messages=session.messages,
            tool_results=tool_results,
            original_response=llm_response,
        )

        session.current_response_text = (
            followup.text
            if not followup.error
            else self._settings.SAFE_RESPONSE_MESSAGE
        )

    async def _respond_text(self, session: Session) -> None:
        if session.state == AgentState.ERROR:
            return

        session.transition_to(AgentState.RESPONDING_TEXT)
        await self._dispatcher.send(
            session.id,
            make_event(EventType.STATE_CHANGED, session.id, state=AgentState.RESPONDING_TEXT.value),
        )

        if session.current_response_text:
            session.messages.append(
                {"role": "assistant", "content": session.current_response_text}
            )

    async def _maybe_generate_audio(self, session: Session) -> None:
        if session.state == AgentState.ERROR:
            return
        if not session.current_response_text:
            return

        session.transition_to(AgentState.GENERATING_AUDIO)
        await self._dispatcher.send(
            session.id,
            make_event(EventType.STATE_CHANGED, session.id, state=AgentState.GENERATING_AUDIO.value),
        )

        audio_bytes, error = await self._tts.synthesize(session.current_response_text)

        if error:
            await self._dispatcher.send(
                session.id,
                make_event(EventType.TTS_FAILED, session.id, recoverable=False),
            )
        else:
            session.current_response_audio = audio_bytes

    async def _done(self, session: Session) -> None:
        if session.state == AgentState.ERROR:
            return

        session.transition_to(AgentState.DONE)

        audio_b64 = None
        if session.current_response_audio:
            audio_b64 = base64.b64encode(session.current_response_audio).decode()

        event_data: dict[str, Any] = {
            "text": session.current_response_text or "",
            "audio": audio_b64,
            "messages": session.messages,
        }

        await self._dispatcher.send(
            session.id,
            make_event(EventType.AGENT_RESPONSE, session.id, **event_data),
        )


class _SttBlockingFailure(Exception):
    """Raised when STT fails and the model cannot process audio directly."""
