import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load training data
with open("sample_data.json") as f:
    data = json.load(f)

texts = [d["text"] for d in data]
labels = [d["intent"] for d in data]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000, C=1.0))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Save
joblib.dump(pipeline, "intent_model.pkl")
print("✅ Model saved to intent_model.pkl")

# Quick sanity check
print("\n=== Sanity Check ===")
test_sentences = [
    "What's my balance?",
    "I see a weird transaction",
    "Transfer 100 dollars to Sara",
    "Do you have savings accounts?",
    "I need help logging in",
    "Show me my purchases this month"
]
for s in test_sentences:
    intent = pipeline.predict([s])[0]
    proba = pipeline.predict_proba([s])[0]
    confidence = max(proba)
    print(f"  '{s}'\n    → {intent} (confidence: {confidence:.2f})\n")
