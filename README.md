# Agente Conversacional Multimodal Escalable

Sistema backend de agente conversacional multimodal event-driven, capaz de recibir audio y texto, razonar con LLM, invocar herramientas externas, y responder en texto y/o audio. Soporta OpenAI, Anthropic y Gemini.

## Arquitectura

```
Cliente (WS / HTTP)
        │
   ┌────▼─────┐
   │ API GW   │  rate limiting · throttling · validación
   └────┬─────┘
        │
   ┌────▼──────────┐
   │ Session Mgr   │  stateless · TTL · límite concurrencia
   └────┬──────────┘
        │
   ┌────▼──────────────────────────────────────────────┐
   │               Agent Pipeline (State Machine)       │
   │                                                    │
   │  IDLE → RECEIVING_INPUT → [TRANSCRIBING]           │
   │       → THINKING → [CALLING_TOOL]                  │
   │       → RESPONDING_TEXT → [GENERATING_AUDIO]       │
   │       → DONE → IDLE                                │
   └────┬──────────────────────────────────────────────┘
        │          │           │           │
   ┌────▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼────┐
   │  STT   │ │  LLM   │ │ Tools  │ │  TTS   │
   └────────┘ └────────┘ └────────┘ └────────┘
     retry     retry      retry      retry
     circuit   circuit               circuit
     breaker   breaker               breaker
```

## Estructura del proyecto

```
app/
├── main.py                 # Entry point, wiring, lifespan
├── config.py               # .env settings (pydantic-settings)
├── state_machine.py        # Agent pipeline / state machine
├── models/
│   ├── events.py           # Contratos de eventos al cliente
│   ├── requests.py         # Modelos de entrada (text, audio)
│   └── session.py          # Sesión y estados del agente
├── services/
│   ├── session_manager.py  # Gestión de sesiones in-memory
│   ├── audio_ingestion.py  # Validación de audio entrante
│   ├── stt_service.py      # Speech-to-Text con retry + CB
│   ├── llm_orchestrator.py # LLM routing con retry + CB
│   ├── tool_executor.py    # Ejecución de herramientas
│   ├── tts_service.py      # Text-to-Speech con retry + CB
│   ├── context_compressor.py # Compresión de contexto (resúmenes)
│   └── event_dispatcher.py # Dispatch de eventos por WebSocket
├── providers/
│   ├── base.py             # Interfaces abstractas (ABC)
│   ├── openai_provider.py  # OpenAI (GPT-4o, Whisper, TTS)
│   ├── anthropic_provider.py # Anthropic (Claude)
│   └── gemini_provider.py  # Google Gemini
├── resilience/
│   ├── retry_manager.py    # Retry con backoff exponencial
│   ├── circuit_breaker.py  # Circuit breaker por servicio
│   └── fault_handler.py    # Políticas de fallo por etapa
├── middleware/
│   ├── rate_limiter.py     # Token-bucket por usuario/IP
│   ├── throttling.py       # Middleware FastAPI de throttling
│   └── validation.py       # Validación estricta de payloads
└── api/
    ├── http_routes.py      # REST: /api/v1/chat/text, /chat/audio
    └── ws_routes.py        # WebSocket: /ws/{session_id}
```

## Requisitos

- Python 3.11+
- Al menos una API key de proveedor (OpenAI, Anthropic o Gemini)

## Instalación

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables
cp .env.example .env
# Editar .env con tus API keys y configuración
```

## Ejecución

```bash
# Desarrollo
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Producción (múltiples workers)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Pruebas

Comprobar que la aplicación arranca y responde:

```bash
# Verificar importación y configuración
python -c "from app.main import app; from app.config import get_settings; print('OK', get_settings().LLM_PROVIDER.value)"

# Con el servidor en marcha (en otra terminal): health y sesión
# curl http://localhost:8000/api/v1/health
# curl -X POST http://localhost:8000/api/v1/session
```

Prueba de integración vía cliente HTTP (requiere API key en `.env`):

```bash
python -c "
import asyncio
from httpx import ASGITransport, AsyncClient
from app.main import app

async def test():
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test', timeout=90.0) as c:
        r = await c.get('/api/v1/health')
        assert r.status_code == 200
        r2 = await c.post('/api/v1/session')
        assert r2.status_code == 200
        print('Health y sesión OK')
asyncio.run(test())
"
```

## API

### REST

| Método   | Ruta                          | Descripción                   |
|----------|-------------------------------|-------------------------------|
| `GET`    | `/api/v1/health`              | Health check                  |
| `POST`   | `/api/v1/session`             | Crear sesión                  |
| `DELETE` | `/api/v1/session/{id}`        | Eliminar sesión               |
| `POST`   | `/api/v1/chat/text`           | Enviar texto al agente        |
| `POST`   | `/api/v1/chat/audio`          | Enviar audio al agente        |

### WebSocket

Conectar a `ws://host:port/ws/{session_id}`

**Mensajes del cliente:**

```json
{"type": "text", "text": "Hola, ¿cómo estás?"}
```

```json
{"type": "audio", "audio_base64": "...", "mime_type": "audio/webm"}
```

**Eventos del servidor:**

```json
{"event": "state_changed", "session_id": "...", "data": {"state": "thinking"}}
{"event": "transcription_final", "session_id": "...", "data": {"text": "..."}}
{"event": "agent_response", "session_id": "...", "data": {"text": "...", "audio": "..."}}
{"event": "tool_failed", "session_id": "...", "data": {"tool_name": "...", "recoverable": true}}
{"event": "safe_response", "session_id": "...", "data": {"message": "..."}}
```

## Configuración

Toda la configuración se realiza vía variables de entorno (`.env`). Ver `.env.example` para la referencia completa.

### Proveedores soportados

| Proveedor | LLM | STT | TTS | Audio nativo |
|-----------|-----|-----|-----|--------------|
| OpenAI    | ✅  | ✅  | ✅  | ✅ (GPT-4o)  |
| Anthropic | ✅  | ❌  | ❌  | ❌           |
| Gemini    | ✅  | ✅  | ❌  | ✅           |

### Políticas de fallo

| Etapa | Acción por defecto         | Configurable |
|-------|----------------------------|--------------|
| STT   | Solicitar repetir audio    | ✅           |
| Tool  | Continuar sin herramienta  | ✅           |
| LLM   | Respuesta segura           | ✅           |
| TTS   | Responder solo texto       | ✅           |

## Escalabilidad

- Backend 100% stateless (sesiones en memoria reemplazable por Redis)
- Rate limiting por IP con token-bucket
- Circuit breakers por servicio externo
- Workers asíncronos (uvicorn + asyncio)
- Diseñado para ≥50.000 sesiones concurrentes
