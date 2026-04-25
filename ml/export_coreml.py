"""
export_coreml.py
Exports both trained sklearn classifiers to Core ML (.mlmodel) using
manual protobuf GLMClassifier spec construction.

This bypasses coremltools' broken sklearn converter (which only supports
sklearn ≤ 1.5.1) and builds the spec directly from numpy weights.

Models produced:
  IntentClassifier.mlmodel    – input: float32 tfidf_input (15 000 dims)
  SentimentClassifier.mlmodel – input: float32 tfidf_input (15 000 dims)

The vocabulary and IDF weights are embedded in each model's
user_defined_metadata so TFIDFPreprocessor.swift can compute the exact
same features on-device without a server round-trip.

Usage (Mac only):
    python export_coreml.py
"""

import json
import os
import numpy as np
import joblib

try:
    import coremltools as ct
    from coremltools.proto import Model_pb2
except ImportError:
    print("❌ coremltools not installed. Run: pip install coremltools")
    exit(1)


def _build_glm_spec(
    coef: np.ndarray,   # (n_classes, n_features) float32
    bias: np.ndarray,   # (n_classes,)             float32
    labels: list,
    input_name: str,
    output_name: str,
) -> "Model_pb2.Model":
    """Build a CoreML GLMClassifier protobuf spec from raw numpy weights."""
    n_features = coef.shape[1]

    spec = Model_pb2.Model()
    spec.specificationVersion = 4   # CoreML 3 → iOS 13+

    glm = spec.glmClassifier
    for i, row_weights in enumerate(coef):
        row = glm.weights.add()
        for w in row_weights.tolist():
            row.value.append(float(w))
        glm.offset.append(float(bias[i]))

    for lbl in labels:
        glm.stringClassLabels.vector.append(lbl)

    glm.classEncoding = Model_pb2.GLMClassifier.OneVsRest

    # Input: float32 array of shape (n_features,)
    inp = spec.description.input.add()
    inp.name = input_name
    inp.type.multiArrayType.shape.append(n_features)
    inp.type.multiArrayType.dataType = Model_pb2.ArrayFeatureType.FLOAT32

    # Output 1: predicted label string
    out_lbl = spec.description.output.add()
    out_lbl.name = output_name
    out_lbl.type.stringType.CopyFrom(Model_pb2.StringFeatureType())

    # Output 2: probability dictionary {label: probability}
    out_proba = spec.description.output.add()
    out_proba.name = f"{output_name}Probability"
    out_proba.type.dictionaryType.stringKeyType.CopyFrom(
        Model_pb2.StringFeatureType()
    )

    spec.description.predictedFeatureName       = output_name
    spec.description.predictedProbabilitiesName = f"{output_name}Probability"
    return spec


def export_pipeline(
    pkl_name: str,
    output_mlmodel: str,
    prediction_feature: str,
) -> bool:
    if not os.path.exists(pkl_name):
        print(f"❌  {pkl_name} not found — run train.py first.")
        return False

    print(f"\nLoading {pkl_name} ...")
    pipe  = joblib.load(pkl_name)
    tfidf = pipe.named_steps["tfidf"]
    clf   = pipe.named_steps["clf"]

    vocab  = tfidf.vocabulary_                   # {token: feature_index}
    idf    = tfidf.idf_.astype("float32")        # (n_features,)
    labels = clf.classes_.tolist()
    coef   = clf.coef_.astype("float32")         # (n_classes, n_features)
    bias   = clf.intercept_.astype("float32")    # (n_classes,)

    print(f"  {len(labels)} classes | {coef.shape[1]:,} features")
    print("  Building GLMClassifier spec ...")

    spec    = _build_glm_spec(coef, bias, labels, "tfidf_input", prediction_feature)
    mlmodel = ct.models.MLModel(spec)

    # Embed preprocessing metadata — Swift reads these to replicate TF-IDF
    # vocab values are numpy int64 → cast to plain int for JSON serialization
    mlmodel.user_defined_metadata["vocabulary_json"]  = json.dumps({k: int(v) for k, v in vocab.items()})
    mlmodel.user_defined_metadata["idf_weights_json"] = json.dumps(idf.tolist())
    mlmodel.user_defined_metadata["labels_json"]      = json.dumps(labels)
    mlmodel.user_defined_metadata["n_features"]       = str(coef.shape[1])
    mlmodel.user_defined_metadata["sublinear_tf"]     = "true"
    mlmodel.user_defined_metadata["ngram_range_min"]  = "1"
    mlmodel.user_defined_metadata["ngram_range_max"]  = "3"
    mlmodel.user_defined_metadata["strip_accents"]    = "unicode"

    mlmodel.short_description = (
        f"Hey Banco – Spanish customer {prediction_feature} classifier. "
        "Input: TF-IDF vector computed by TFIDFPreprocessor.swift."
    )
    mlmodel.author  = "Smart Financial Assistant – Hey Banco"
    mlmodel.version = "2.0"
    mlmodel.input_description["tfidf_input"] = (
        f"TF-IDF feature vector ({coef.shape[1]:,} dims, float32). "
        "Compute with TFIDFPreprocessor.vectorize(text:modelURL:)."
    )
    mlmodel.output_description[prediction_feature] = (
        f"Predicted {prediction_feature} label"
    )

    mlmodel.save(output_mlmodel)
    print(f"  ✅ Saved → {output_mlmodel}")
    return True


# ── Export both classifiers ───────────────────────────────────────────────────

export_pipeline("intent_model.pkl",    "IntentClassifier.mlmodel",    "intent")
export_pipeline("sentiment_model.pkl", "SentimentClassifier.mlmodel", "sentiment")

# Verify action_map.json is present for bundling
if os.path.exists("action_map.json"):
    with open("action_map.json") as f:
        rules = json.load(f)
    print(f"\n✅ action_map.json ready  ({len(rules)} rules)")
else:
    print("\n⚠️  action_map.json not found — run train.py first.")

print("""
────────────────────────────────────────────────────────────────
Next steps in Xcode:
  1. Drag into SmartFinApp/ target:
       • IntentClassifier.mlmodel
       • SentimentClassifier.mlmodel
       • action_map.json   (Build Phases → Copy Bundle Resources)
       • TFIDFPreprocessor.swift

  2. In Swift:
       let vector    = try TFIDFPreprocessor.vectorize(text: userMessage,
                           modelURL: IntentClassifier.urlOfModelInThisBundle)
       let intent    = try IntentClassifier().prediction(tfidf_input: vector)
       let sentiment = try SentimentClassifier().prediction(tfidf_input: vector)
       let action    = ActionRouter.action(intent: intent.intent,
                                           sentiment: sentiment.sentiment)
────────────────────────────────────────────────────────────────
""")

