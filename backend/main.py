from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal
import uvicorn

from model_loader import predict_full
from risk import assess_risk, Transaction, RiskResult
from responder import generate_response
from conversation import converse, LLM_ENABLED_ACTIONS

app = FastAPI(
    title="Smart Financial Assistant API",
    description="Spanish NLP: intent + sentiment + action + risk",
    version="2.0.0"
)

# Allow iOS app and local dev tools to call this API freely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    message: str
    # Optional transaction context sent by the client
    transaction_amount: Optional[float] = 0.0
    transaction_category: Optional[str] = "general"
    failed_attempts: Optional[int] = 0
    is_anomaly: Optional[bool] = False
    transaction_country: Optional[str] = None

class ChatResponse(BaseModel):
    user_id: str
    message: str
    intent: str
    intent_confidence: float
    sentiment: str
    sentiment_confidence: float
    action: str
    escalate: bool
    risk_level: str
    risk_score: int
    risk_flags: list[str]
    response: str
    response_hint: str


# ── Conversation (multi-turn LLM) models ─────────────────────────────────────

class ConverseTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ConverseRequest(BaseModel):
    user_id: str
    action: str
    message: str
    history: list[ConverseTurn] = []

class ConverseResponse(BaseModel):
    user_id: str
    action: str
    message: str
    resolved: bool
    escalate: bool


# ── Mock user profile store (replace with real DB in production) ──────────────

MOCK_USERS: dict[str, dict] = {
    "user_001": {
        "name": "Ana García",
        "avg_transaction_amount": 120.0,
        "common_categories": ["groceries", "transport", "utilities"],
        "country": "MX",
    },
    "user_002": {
        "name": "Carlos López",
        "avg_transaction_amount": 350.0,
        "common_categories": ["electronics", "dining", "travel"],
        "country": "MX",
    },
    "user_003": {
        "name": "María Hernández",
        "avg_transaction_amount": 200.0,
        "common_categories": ["clothing", "dining", "entertainment"],
        "country": "MX",
    },
}

def get_user_profile(user_id: str) -> dict:
    return MOCK_USERS.get(user_id, {
        "avg_transaction_amount": 150.0,
        "common_categories": [],
        "country": "MX",
    })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Smart Financial Assistant API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/intents")
def list_intents():
    """List all supported intents."""
    return {
        "intents": [
            "consulta_saldo",
            "transferencia",
            "cancelar_cuenta",
            "cancelar_cargo",
            "bloqueo_tarjeta",
            "fraude_cargo_no_reconocido",
            "solicitar_producto",
            "problema_tecnico",
            "queja_servicio",
            "hablar_asesor",
            "cambio_datos_perfil",
            "informacion_general",
        ],
        "sentiments": ["positivo", "neutral", "negativo_queja", "negativo_urgente"],
        "actions": [
            "retener_cliente_urgente",
            "retener_cliente",
            "escalar_agente",
            "resolver_problema",
            "procesar_solicitud",
            "informar",
            "oferta_producto",
        ],
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main endpoint: takes a Spanish customer message and returns
    intent + sentiment + action + risk + response.

    Example body:
    {
        "user_id": "user_001",
        "message": "Quiero cancelar mi cuenta, estoy muy decepcionado del servicio"
    }
    """
    # 1. Classify intent + sentiment → action
    prediction = predict_full(req.message)
    intent    = prediction["intent"]
    sentiment = prediction["sentiment"]
    action    = prediction["action"]

    print(
        f"[{req.user_id}] '{req.message[:60]}' → "
        f"intent={intent}({prediction['intent_conf']:.2f}) "
        f"sentiment={sentiment}({prediction['sentiment_conf']:.2f}) "
        f"action={action}"
    )

    # 2. Load user profile
    user_profile = get_user_profile(req.user_id)

    # 3. Assess risk from transaction context
    tx = Transaction(
        amount=req.transaction_amount or 0.0,
        category=req.transaction_category or "general",
        failed_attempts=req.failed_attempts or 0,
        is_anomaly=req.is_anomaly or False,
        country=req.transaction_country,
        user_avg_amount=user_profile.get("avg_transaction_amount", 100.0),
    )
    risk: RiskResult = assess_risk(tx, user_profile)

    print(f"  risk={risk.level} (score={risk.score}) flags={risk.flags}")

    # 4. Generate Spanish response based on action
    resp = generate_response(action, intent, sentiment, risk.flags)

    return ChatResponse(
        user_id=req.user_id,
        message=req.message,
        intent=intent,
        intent_confidence=prediction["intent_conf"],
        sentiment=sentiment,
        sentiment_confidence=prediction["sentiment_conf"],
        action=action,
        escalate=resp["escalate"],
        risk_level=risk.level,
        risk_score=risk.score,
        risk_flags=risk.flags,
        response=resp["message"],
        response_hint=resp["hint"],
    )


@app.post("/converse", response_model=ConverseResponse)
def converse_endpoint(req: ConverseRequest):
    """
    Multi-turn LLM conversation endpoint.

    The iOS app drops into this mode after the on-device classifier picks an
    action that benefits from a real conversation (retention, escalation,
    troubleshooting, etc.). The client sends:

        - action: the action context selected by the on-device classifier
        - message: the latest user turn (Spanish)
        - history: previous {role, content} pairs in this conversation

    The server replies with the assistant message plus two control flags:

        - resolved: true when the conversation has reached a natural close.
                    The client should then leave conversation mode and return
                    to neutral classification.
        - escalate: true when a human agent should take over.
    """
    if req.action not in LLM_ENABLED_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Action '{req.action}' is not enabled for LLM conversation. "
                f"Supported: {sorted(LLM_ENABLED_ACTIONS)}"
            ),
        )

    history = [{"role": t.role, "content": t.content} for t in req.history]

    try:
        result = converse(
            action=req.action,
            history=history,
            user_message=req.message,
        )
    except RuntimeError as e:
        # Most likely missing ANTHROPIC_API_KEY
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    print(
        f"[{req.user_id}] /converse action={req.action} "
        f"resolved={result['resolved']} escalate={result['escalate']}"
    )

    return ConverseResponse(
        user_id=req.user_id,
        action=req.action,
        message=result["message"],
        resolved=result["resolved"],
        escalate=result["escalate"],
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
