"""REST API endpoints for text and audio input."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.models.requests import ChatAudioRequest, ChatTextRequest
from app.models.session import AgentState
from app.middleware.validation import PayloadValidationError
from app.services.audio_ingestion import AudioIngestionError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["agent"])


def _get_deps(request: Request) -> dict[str, Any]:
    return request.app.state.deps


@router.get("/health")
async def health(request: Request):
    deps = _get_deps(request)
    return {
        "status": "ok",
        "active_sessions": deps["session_manager"].active_count,
        "ws_connections": deps["dispatcher"].connected_count,
    }


@router.post("/session")
async def create_session(request: Request):
    deps = _get_deps(request)
    session = await deps["session_manager"].create()
    return {"session_id": session.id}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    deps = _get_deps(request)
    await deps["session_manager"].delete(session_id)
    return {"status": "deleted"}


@router.post("/chat/text")
async def chat_text(body: ChatTextRequest, request: Request):
    deps = _get_deps(request)
    validator = deps["validator"]
    session_mgr = deps["session_manager"]
    pipeline = deps["pipeline"]

    last_msg = body.messages[-1]
    try:
        validator.validate_text(last_msg.content)
    except PayloadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    session = await session_mgr.get_or_create(body.session_id)

    if session.state not in (AgentState.IDLE, AgentState.DONE):
        raise HTTPException(status_code=409, detail="Session is busy")

    session.reset_turn()
    session.messages = [m.model_dump() for m in body.messages]
    session.current_input_text = last_msg.content

    await pipeline.run(session)

    result: dict = {
        "session_id": session.id,
        "text": session.current_response_text or "",
        "messages": session.messages,
    }
    return result


@router.post("/chat/audio")
async def chat_audio(body: ChatAudioRequest, request: Request):
    deps = _get_deps(request)
    validator = deps["validator"]
    audio_svc = deps["audio_ingestion"]
    session_mgr = deps["session_manager"]
    pipeline = deps["pipeline"]

    try:
        validator.validate_audio_base64_size(body.audio_base64)
    except PayloadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        audio_bytes = audio_svc.validate_and_decode(body.audio_base64, body.mime_type)
    except AudioIngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    session = await session_mgr.get_or_create(body.session_id)

    if session.state not in (AgentState.IDLE, AgentState.DONE):
        raise HTTPException(status_code=409, detail="Session is busy")

    session.reset_turn()
    session.messages = [m.model_dump() for m in body.messages]
    session.current_input_audio = audio_bytes

    await pipeline.run(session)

    result: dict = {
        "session_id": session.id,
        "text": session.current_response_text or "",
        "messages": session.messages,
    }
    return result
