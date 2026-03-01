"""WebSocket endpoint for real-time bidirectional communication."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.events import EventType, make_event
from app.models.session import AgentState
from app.middleware.validation import PayloadValidationError
from app.services.audio_ingestion import AudioIngestionError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    deps = websocket.app.state.deps
    session_mgr = deps["session_manager"]
    dispatcher = deps["dispatcher"]
    pipeline = deps["pipeline"]
    validator = deps["validator"]
    audio_svc = deps["audio_ingestion"]

    await websocket.accept()

    session = await session_mgr.get_or_create(session_id)
    await dispatcher.register(session.id, websocket)

    await dispatcher.send(
        session.id,
        make_event(EventType.SESSION_CREATED, session.id),
    )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await dispatcher.send(
                    session.id,
                    make_event(EventType.ERROR, session.id, message="Invalid JSON"),
                )
                continue

            await _handle_message(
                session, message, deps, dispatcher, pipeline, validator, audio_svc
            )

    except WebSocketDisconnect:
        logger.info("WS disconnected: session %s", session.id)
    except Exception:
        logger.exception("WS error: session %s", session.id)
    finally:
        await dispatcher.unregister(session.id)


async def _handle_message(
    session,
    message: dict[str, Any],
    deps: dict[str, Any],
    dispatcher,
    pipeline,
    validator,
    audio_svc,
) -> None:
    msg_type = message.get("type")

    if session.state not in (AgentState.IDLE, AgentState.DONE):
        await dispatcher.send(
            session.id,
            make_event(EventType.ERROR, session.id, message="Session is busy"),
        )
        return

    messages_raw = message.get("messages")
    if not messages_raw or not isinstance(messages_raw, list):
        await dispatcher.send(
            session.id,
            make_event(EventType.INPUT_REJECTED, session.id, reason="'messages' array is required"),
        )
        return

    if messages_raw[0].get("role") != "system":
        await dispatcher.send(
            session.id,
            make_event(EventType.INPUT_REJECTED, session.id, reason="First message must have role 'system'"),
        )
        return

    session.reset_turn()
    session.messages = messages_raw

    if msg_type == "text":
        last = messages_raw[-1] if messages_raw else {}
        text = last.get("content", "") if last.get("role") == "user" else message.get("text", "")
        if not text:
            text = message.get("text", "")
            if text:
                session.messages.append({"role": "user", "content": text})

        try:
            validator.validate_text(text)
        except PayloadValidationError as exc:
            await dispatcher.send(
                session.id,
                make_event(EventType.INPUT_REJECTED, session.id, reason=str(exc)),
            )
            return
        session.current_input_text = text

    elif msg_type == "audio":
        audio_b64 = message.get("audio_base64", "")
        mime_type = message.get("mime_type", "audio/webm")
        try:
            validator.validate_audio_base64_size(audio_b64)
            audio_bytes = audio_svc.validate_and_decode(audio_b64, mime_type)
        except (PayloadValidationError, AudioIngestionError) as exc:
            await dispatcher.send(
                session.id,
                make_event(EventType.INPUT_REJECTED, session.id, reason=str(exc)),
            )
            return
        session.current_input_audio = audio_bytes

    else:
        await dispatcher.send(
            session.id,
            make_event(EventType.ERROR, session.id, message=f"Unknown message type: {msg_type}"),
        )
        return

    await pipeline.run(session)
