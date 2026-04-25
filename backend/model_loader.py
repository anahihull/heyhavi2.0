import joblib
import os

_model = None

def get_model():
    """Load the sklearn model once and cache it in memory."""
    global _model
    if _model is None:
        # Look for model relative to this file's location
        model_path = os.path.join(os.path.dirname(__file__), "../ml/intent_model.pkl")
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run 'python ml/train.py' first."
            )
        print(f"Loading model from {model_path} ...")
        _model = joblib.load(model_path)
        print("✅ Model loaded successfully.")
    return _model


def predict_intent(text: str) -> str:
    """Predict the single best intent for a given message."""
    model = get_model()
    return model.predict([text])[0]


def predict_intent_proba(text: str) -> dict:
    """Return a dict of intent → probability for all classes."""
    model = get_model()
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    return {cls: float(p) for cls, p in zip(classes, proba)}
