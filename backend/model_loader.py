import joblib
import json
import os

_intent_model    = None
_sentiment_model = None
_action_map: dict[str, str] = {}

ML_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ml"))


def _load_models():
    global _intent_model, _sentiment_model, _action_map
    if _intent_model is not None:
        return

    intent_path    = os.path.join(ML_DIR, "intent_model.pkl")
    sentiment_path = os.path.join(ML_DIR, "sentiment_model.pkl")
    action_path    = os.path.join(ML_DIR, "action_map.json")

    for path in (intent_path, sentiment_path, action_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run 'python ml/build_dataset.py && python ml/train.py' first."
            )

    print("Loading models ...")
    _intent_model    = joblib.load(intent_path)
    _sentiment_model = joblib.load(sentiment_path)
    with open(action_path, encoding="utf-8") as f:
        _action_map = json.load(f)
    print("✅ Models loaded.")


def predict_intent(text: str) -> str:
    _load_models()
    return _intent_model.predict([text])[0]


def predict_intent_proba(text: str) -> dict:
    _load_models()
    proba = _intent_model.predict_proba([text])[0]
    return {cls: float(p) for cls, p in zip(_intent_model.classes_, proba)}


def predict_full(text: str) -> dict:
    """Return intent, sentiment, action and confidence scores."""
    _load_models()

    intent_proba    = _intent_model.predict_proba([text])[0]
    sentiment_proba = _sentiment_model.predict_proba([text])[0]

    intent    = _intent_model.classes_[intent_proba.argmax()]
    sentiment = _sentiment_model.classes_[sentiment_proba.argmax()]
    action    = _action_map.get(f"{intent}|{sentiment}", "informar")

    return {
        "intent":         intent,
        "intent_conf":    round(float(intent_proba.max()), 3),
        "sentiment":      sentiment,
        "sentiment_conf": round(float(sentiment_proba.max()), 3),
        "action":         action,
    }

