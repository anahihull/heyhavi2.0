"""
Export the trained sklearn intent classifier to Core ML format (.mlmodel).
Run this on Mac only — coremltools is not supported on Windows.

Usage:
    python export_coreml.py
"""
import joblib

try:
    import coremltools as ct
except ImportError:
    print("❌ coremltools not installed. Run: pip install coremltools")
    print("   (Mac only — skip this step on Windows)")
    exit(1)

# Load the trained sklearn pipeline
print("Loading model from intent_model.pkl ...")
pipeline = joblib.load("intent_model.pkl")

# Get class labels for metadata
intents = pipeline.classes_.tolist()
print(f"Intents: {intents}")

# Convert the full sklearn pipeline to Core ML
print("Converting to Core ML format ...")
model = ct.converters.sklearn.convert(
    pipeline,
    input_features="message",
    output_feature_names="intent"
)

# Add metadata
model.short_description = "Financial chatbot intent classifier"
model.input_description["message"] = "User chat message text"
model.output_description["intent"] = "Predicted intent label"
model.author = "Smart Financial Assistant Team"
model.version = "1.0"

# Save
model.save("intent_model.mlmodel")
print("✅ Core ML model saved to intent_model.mlmodel")
print(f"   Supported intents: {intents}")
print("\nNext step: drag intent_model.mlmodel into your Xcode project.")
