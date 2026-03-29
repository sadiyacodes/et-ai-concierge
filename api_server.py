"""
ET AI Concierge — FastAPI REST + WebSocket API Server
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from orchestrator import run_turn, start_session, get_memory_store, get_knowledge_base
from audit_logger import AuditLogger

import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

CONFIG = _load_config()

# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    import os
    os.makedirs("./data", exist_ok=True)

    # Init components
    get_memory_store()
    get_knowledge_base()
    print("[ET AI Concierge] System initialised — ready to serve.")
    yield
    # Shutdown
    print("[ET AI Concierge] Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ET AI Concierge",
    description="Multi-agent AI concierge for the Economic Times ecosystem",
    version=CONFIG.get("app", {}).get("version", "1.0.0"),
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dev mode — restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
import os
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    recommendations: list
    journey_step: Optional[dict] = None
    journey_phase: str = "discover"
    profile_completeness: float = 0.0
    session_id: str
    audit_id: str = ""
    intents: list = []
    agents_called: list = []

class SessionStartRequest(BaseModel):
    user_id: Optional[str] = None
    returning_user_hint: bool = False

class SessionStartResponse(BaseModel):
    session_id: str
    user_id: str
    greeting: str
    user_type: str

class ScenarioRequest(BaseModel):
    scenario: str  # "cold_start" | "lapsed_subscriber" | "cross_sell_home_loan"


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Process a single chat turn."""
    if not req.message or not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    result = await run_turn(req.message.strip(), req.session_id, req.user_id)

    return ChatResponse(
        response=result.get("response", ""),
        recommendations=result.get("recommendations", []),
        journey_step=result.get("journey_step"),
        journey_phase=result.get("journey_phase", "discover"),
        profile_completeness=result.get("profile_completeness", 0.0),
        session_id=result.get("session_id", req.session_id),
        audit_id=result.get("audit_id", ""),
        intents=result.get("intents", []),
        agents_called=result.get("agents_called", []),
    )


@app.post("/api/session/start", response_model=SessionStartResponse)
async def session_start(req: SessionStartRequest):
    """Start a new session."""
    result = await start_session(req.user_id, req.returning_user_hint)
    return SessionStartResponse(**result)


@app.get("/api/session/{session_id}/profile")
async def session_profile(session_id: str):
    """Get the current session's user profile."""
    store = get_memory_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    profile = session.get("user_profile", {})
    journey = session.get("journey_state", {})

    return {
        "profile": profile,
        "completeness": profile.get("profile_completeness", 0.0),
        "journey_state": journey,
    }


@app.get("/api/session/{session_id}/audit")
async def session_audit(session_id: str):
    """Get audit entries for a session."""
    logger = AuditLogger()
    entries = logger.get_session_audit(session_id)
    summary = logger.get_session_summary(session_id)
    return {
        "audit_entries": entries,
        "session_metrics": summary,
    }


@app.post("/api/demo/scenario")
async def run_scenario(req: ScenarioRequest):
    """Run a demo scenario and return full transcript."""
    scenarios = {
        "cold_start": [
            "Hi, I came across Economic Times. I'm not sure where to start — I have some savings but they're just sitting in my bank account.",
            "I'm 28, work in IT, earn decent but never really looked into investing. My friends keep talking about SIPs but I don't really know what that means.",
            "I want to start with something simple. What should I do first?",
        ],
        "lapsed_subscriber": [
            "Hey, I used to have ET Prime but my subscription lapsed a while back. Just browsing again.",
            "Yeah I mainly used to follow the markets stuff, especially smallcap coverage.",
            "What's new? Is there a reason to come back?",
        ],
        "cross_sell_home_loan": [
            "What are the current home loan interest rates? Nifty is volatile today too.",
            "I've been comparing SBI and HDFC home loan rates. The ET article mentioned 8.5% but that seems high.",
            "Is this a good time to buy property given market conditions?",
        ],
    }

    if req.scenario not in scenarios:
        raise HTTPException(400, f"Unknown scenario. Choose from: {list(scenarios.keys())}")

    turns = scenarios[req.scenario]

    # Set up session depending on scenario
    from orchestrator import _init_components
    _init_components()
    store = get_memory_store()

    if req.scenario == "lapsed_subscriber":
        session = await start_session(returning_user_hint=True)
        store.update_profile(session["session_id"], "et_subscription_status", "lapsed", 0.9)
        store.update_profile(session["session_id"], "persona_tag", "lapsed_subscriber", 0.9)
        store.update_profile(session["session_id"], "preferred_content_types", ["markets", "smallcap"], 0.8)
        store.update_profile(session["session_id"], "days_since_last_visit", 90, 0.9)
    elif req.scenario == "cross_sell_home_loan":
        session = await start_session()
        store.update_profile(session["session_id"], "persona_tag", "seasoned_trader", 0.8)
        store.update_profile(session["session_id"], "investment_experience", "advanced", 0.8)
        store.update_profile(session["session_id"], "et_products_used", ["et_markets"], 0.8)
    else:
        session = await start_session()

    transcript = []
    transcript.append({"role": "assistant", "content": session["greeting"], "metadata": {"turn": 0, "type": "greeting"}})

    for i, user_msg in enumerate(turns, 1):
        result = await run_turn(user_msg, session["session_id"], session["user_id"])
        transcript.append({
            "role": "user",
            "content": user_msg,
            "metadata": {"turn": i},
        })
        transcript.append({
            "role": "assistant",
            "content": result.get("response", ""),
            "metadata": {
                "turn": i,
                "recommendations": result.get("recommendations", []),
                "intents": result.get("intents", []),
                "agents_called": result.get("agents_called", []),
                "profile_completeness": result.get("profile_completeness", 0.0),
                "evaluator_score": result.get("evaluator_score", 0.0),
                "cross_sell_signals": result.get("cross_sell_signals", []),
                "journey_phase": result.get("journey_phase", "discover"),
            },
        })

    # Get final profile
    final_session = store.get_session(session["session_id"])
    final_profile = final_session["user_profile"] if final_session else {}

    return {
        "scenario": req.scenario,
        "session_id": session["session_id"],
        "transcript": transcript,
        "final_profile": final_profile,
        "profile_completeness": final_profile.get("profile_completeness", 0.0),
    }


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket-based chat with streaming responses."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            user_id = msg.get("user_id", str(uuid.uuid4()))
            message = msg.get("message", "")

            if not message.strip():
                await websocket.send_json({"type": "error", "content": "Empty message"})
                continue

            # Send "typing" indicator
            await websocket.send_json({"type": "typing", "content": True})

            # Run the turn
            result = await run_turn(message.strip(), session_id, user_id)

            response_text = result.get("response", "")

            # Stream response word-by-word
            words = response_text.split()
            for i, word in enumerate(words):
                await websocket.send_json({
                    "type": "token",
                    "content": word + (" " if i < len(words) - 1 else ""),
                })
                await asyncio.sleep(0.03)  # Simulate streaming

            # Send complete message with metadata
            await websocket.send_json({
                "type": "complete",
                "content": response_text,
                "recommendations": result.get("recommendations", []),
                "journey_step": result.get("journey_step"),
                "journey_phase": result.get("journey_phase", "discover"),
                "profile_completeness": result.get("profile_completeness", 0.0),
                "evaluator_score": result.get("evaluator_score", 0.0),
                "intents": result.get("intents", []),
                "agents_called": result.get("agents_called", []),
                "cross_sell_signals": result.get("cross_sell_signals", []),
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Health & Metrics
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": CONFIG.get("app", {}).get("version", "1.0.0"),
        "model": CONFIG.get("llm", {}).get("primary", {}).get("model", "gemini-1.5-flash"),
    }


@app.get("/metrics")
async def metrics():
    try:
        logger = AuditLogger()
        return logger.get_metrics()
    except Exception:
        return {"sessions_today": 0, "avg_evaluator_score": 0.0, "top_recommended_products": []}


# ---------------------------------------------------------------------------
# Serve frontend files
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_chat_ui():
    if os.path.exists("frontend/chat_ui.html"):
        return FileResponse("frontend/chat_ui.html")
    return {"message": "ET AI Concierge API is running. Open /docs for API documentation."}


@app.get("/dashboard")
async def serve_dashboard():
    if os.path.exists("frontend/dashboard.html"):
        return FileResponse("frontend/dashboard.html")
    raise HTTPException(404, "Dashboard not found")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = CONFIG.get("app", {}).get("port", 8000)
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)
