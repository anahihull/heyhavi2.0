from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from model_loader import predict_intent, predict_intent_proba
from risk import assess_risk, Transaction, RiskResult
from responder import generate_response

app = FastAPI(
    title="Smart Financial Assistant API",
    description="NLP intent classification + rule-based risk detection",
    version="1.0.0"
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
    risk_level: str
    risk_score: int
    risk_flags: list[str]
    response: str


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
            "check_balance",
            "transfer_money",
            "fraud_alert",
            "product_inquiry",
            "customer_support",
            "transaction_history",
        ]
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main endpoint: takes a user message and returns intent + risk + response.

    Example body:
    {
        "user_id": "user_001",
        "message": "I see a charge I don't recognize",
        "transaction_amount": 8500,
        "failed_attempts": 3,
        "is_anomaly": true,
        "transaction_country": "US"
    }
    """
    # 1. Classify intent using the ML model
    intent = predict_intent(req.message)
    proba  = predict_intent_proba(req.message)
    confidence = proba.get(intent, 0.0)

    print(f"[{req.user_id}] '{req.message}' → intent={intent} ({confidence:.2f})")

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

    # 4. Generate human-like response
    response_text = generate_response(intent, risk.level, risk.flags)

    return ChatResponse(
        user_id=req.user_id,
        message=req.message,
        intent=intent,
        intent_confidence=round(confidence, 3),
        risk_level=risk.level,
        risk_score=risk.score,
        risk_flags=risk.flags,
        response=response_text,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
