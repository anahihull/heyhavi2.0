"""
train.py
Trains two TF-IDF + Logistic Regression classifiers from the labeled Spanish
customer-service dataset produced by build_dataset.py:

  1. intent_model.pkl    – what the customer wants
  2. sentiment_model.pkl – how the customer feels

A rule-based action_map.json then maps (intent, sentiment) → action so the
assistant knows exactly what to do (retain, escalate, inform, etc.).

Run:
    python build_dataset.py   # first time only – labels the 50k CSV
    python train.py
"""

import json
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─────────────────────── load labeled data ──────────────────────────────────

data_path = os.path.join(os.path.dirname(__file__), "labeled_data.json")
if not os.path.exists(data_path):
    raise FileNotFoundError(
        "labeled_data.json not found. Run `python build_dataset.py` first."
    )

with open(data_path, encoding="utf-8") as f:
    data = json.load(f)

texts      = [d["text"]      for d in data]
intents    = [d["intent"]    for d in data]
sentiments = [d["sentiment"] for d in data]

print(f"Loaded {len(texts):,} labeled examples")

# ─────────────────────── shared TF-IDF config ────────────────────────────────

TFIDF_KWARGS = dict(
    ngram_range=(1, 3),
    max_features=15_000,
    sublinear_tf=True,
    strip_accents="unicode",   # handles Spanish accents robustly
    analyzer="word",
)

# ─────────────────────── intent classifier ──────────────────────────────────

print("\n── Training intent classifier ──")
X_tr, X_te, y_tr, y_te = train_test_split(
    texts, intents, test_size=0.15, random_state=42, stratify=intents
)

intent_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
    ("clf",   LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced")),
])
intent_pipeline.fit(X_tr, y_tr)

y_pred_intent = intent_pipeline.predict(X_te)
print(classification_report(y_te, y_pred_intent, zero_division=0))

# ─────────────────────── sentiment classifier ───────────────────────────────

print("── Training sentiment classifier ──")
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    texts, sentiments, test_size=0.15, random_state=42, stratify=sentiments
)

sentiment_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
    ("clf",   LogisticRegression(max_iter=1000, C=1.5, class_weight="balanced")),
])
sentiment_pipeline.fit(X_tr2, y_tr2)

y_pred_sent = sentiment_pipeline.predict(X_te2)
print(classification_report(y_te2, y_pred_sent, zero_division=0))

# ─────────────────────── save models ────────────────────────────────────────

joblib.dump(intent_pipeline,    "intent_model.pkl")
joblib.dump(sentiment_pipeline, "sentiment_model.pkl")
print("\n✅ intent_model.pkl   saved")
print("✅ sentiment_model.pkl saved")

# Save the action map so other services can import it without re-importing
# this training script.
from build_dataset import ACTION_MAP  # noqa: E402  (import after heavy work)
action_map_serialisable = {
    f"{intent}|{sentiment}": action
    for (intent, sentiment), action in ACTION_MAP.items()
}
with open("action_map.json", "w", encoding="utf-8") as f:
    json.dump(action_map_serialisable, f, ensure_ascii=False, indent=2)
print("✅ action_map.json     saved")

# ─────────────────────── inference helper ───────────────────────────────────

def predict(text: str) -> dict:
    """Return intent, sentiment, action and their confidence scores."""
    intent_proba    = intent_pipeline.predict_proba([text])[0]
    sentiment_proba = sentiment_pipeline.predict_proba([text])[0]

    intent    = intent_pipeline.classes_[intent_proba.argmax()]
    sentiment = sentiment_pipeline.classes_[sentiment_proba.argmax()]
    action    = ACTION_MAP.get((intent, sentiment), "informar")

    return {
        "intent":           intent,
        "intent_conf":      round(float(intent_proba.max()), 3),
        "sentiment":        sentiment,
        "sentiment_conf":   round(float(sentiment_proba.max()), 3),
        "action":           action,
    }

# ─────────────────────── sanity check ───────────────────────────────────────

print("\n── Sanity check (Spanish) ──")
test_cases = [
    "Quiero cancelar mi cuenta, estoy muy decepcionado del servicio",
    "¿Cuánto tengo en mi saldo?",
    "Me hicieron un cargo que yo no reconozco, es urgente",
    "Quiero solicitar una tarjeta de crédito",
    "No me funciona la app, ya no sé qué hacer",
    "Quiero transferir 500 pesos a mi hermana",
    "Hola, buenos días",
    "Quiero hablar con un asesor por favor",
    "Me robaron la tarjeta, necesito bloquearla URGENTE",
    "El servicio es PÉSIMO, exijo que me devuelvan mi dinero ahora mismo!!!",
]
for s in test_cases:
    r = predict(s)
    print(f"  '{s[:70]}'")
    print(f"    intent:    {r['intent']} ({r['intent_conf']:.2f})")
    print(f"    sentiment: {r['sentiment']} ({r['sentiment_conf']:.2f})")
    print(f"    → ACTION:  {r['action']}\n")
