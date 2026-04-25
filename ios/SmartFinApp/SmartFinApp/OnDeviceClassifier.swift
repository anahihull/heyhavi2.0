// OnDeviceClassifier.swift
// Wraps the on-device pipeline:
//
//     text  →  TFIDFPreprocessor  →  IntentClassifier (Core ML) ─┐
//                                                                ├→ ActionRouter → action
//                                  →  SentimentClassifier (Core ML) ┘
//
// `IntentClassifier` and `SentimentClassifier` are *auto-generated* Swift classes
// produced by Xcode from IntentClassifier.mlmodel and SentimentClassifier.mlmodel.
// We can't define types with those names ourselves — that's why this wrapper is
// named `OnDeviceClassifier`.
//
// Usage:
//     let result = OnDeviceClassifier.shared.classify(message: "Quiero cancelar mi cuenta")
//     print(result.intent, result.sentiment, result.action)

import Foundation
import CoreML

struct OnDevicePrediction {
    let intent: String
    let intentConfidence: Double
    let sentiment: String
    let sentimentConfidence: Double
    let action: String
}

final class OnDeviceClassifier {

    static let shared = OnDeviceClassifier()

    private let intentModel: IntentClassifier?
    private let sentimentModel: SentimentClassifier?

    private init() {
        let config = MLModelConfiguration()
        self.intentModel    = try? IntentClassifier(configuration: config)
        self.sentimentModel = try? SentimentClassifier(configuration: config)

        if intentModel == nil || sentimentModel == nil {
            print("⚠️ OnDeviceClassifier: one or more Core ML models failed to load. Falling back to keywords.")
        } else {
            print("✅ OnDeviceClassifier: Core ML models loaded.")
        }
    }

    // MARK: - Public API

    /// Classify a message fully on-device. If the Core ML models or the
    /// TF-IDF metadata aren't available, returns a keyword-based best guess.
    func classify(message: String) -> OnDevicePrediction {
        if let coreML = try? coreMLClassify(message: message) {
            return coreML
        }
        return keywordClassify(message: message)
    }

    /// Convenience for callers that only need the intent label.
    func intent(for message: String) -> String {
        classify(message: message).intent
    }

    // MARK: - Core ML path

    private func coreMLClassify(message: String) throws -> OnDevicePrediction {
        guard
            let intentModel    = intentModel,
            let sentimentModel = sentimentModel,
            let intentURL      = Bundle.main.url(forResource: "IntentClassifier",
                                                 withExtension: "mlmodelc")
                                ?? Bundle.main.url(forResource: "IntentClassifier",
                                                   withExtension: "mlmodel"),
            let sentimentURL   = Bundle.main.url(forResource: "SentimentClassifier",
                                                 withExtension: "mlmodelc")
                                ?? Bundle.main.url(forResource: "SentimentClassifier",
                                                   withExtension: "mlmodel")
        else {
            throw NSError(domain: "OnDeviceClassifier", code: 1)
        }

        // Vectorize once per model (each model has its own vocab/IDF in metadata)
        let intentVec    = try TFIDFPreprocessor.vectorize(text: message, modelURL: intentURL)
        let sentimentVec = try TFIDFPreprocessor.vectorize(text: message, modelURL: sentimentURL)

        // Use the generic MLModel API so we don't depend on the exact auto-generated
        // input/output property names (they vary depending on how the .mlmodel was
        // exported). We read the model's input/output descriptions instead.
        let intentResult    = try predict(model: intentModel.model, vector: intentVec)
        let sentimentResult = try predict(model: sentimentModel.model, vector: sentimentVec)

        let action = ActionRouter.shared.action(intent: intentResult.label,
                                                sentiment: sentimentResult.label)

        return OnDevicePrediction(
            intent: intentResult.label,
            intentConfidence: intentResult.confidence,
            sentiment: sentimentResult.label,
            sentimentConfidence: sentimentResult.confidence,
            action: action
        )
    }

    /// Run a single prediction against an arbitrary classifier MLModel.
    /// Picks the first input feature (assumed to be the TF-IDF vector) and
    /// reads the model's classLabel + class probability dictionary out of the
    /// output features.
    private func predict(model: MLModel, vector: MLMultiArray) throws
        -> (label: String, confidence: Double)
    {
        let desc = model.modelDescription
        guard let inputName = desc.inputDescriptionsByName.keys.first else {
            throw NSError(domain: "OnDeviceClassifier", code: 2)
        }

        let provider = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(multiArray: vector)]
        )
        let prediction = try model.prediction(from: provider)

        // Find the predicted class label (Core ML classifiers typically expose
        // it under a feature whose name matches the model's predictedFeatureName).
        let labelName = desc.predictedFeatureName
            ?? prediction.featureNames.first(where: { prediction.featureValue(for: $0)?.type == .string })
            ?? prediction.featureNames.first!

        let label = prediction.featureValue(for: labelName)?.stringValue ?? "unknown"

        // Find the per-class probabilities dictionary (predictedProbabilitiesName).
        var confidence: Double = 0.0
        if let probName = desc.predictedProbabilitiesName,
           let probValue = prediction.featureValue(for: probName)?.dictionaryValue as? [String: Double] {
            confidence = probValue[label] ?? 0.0
        } else if let probValue = prediction.featureValue(for: "classProbability")?.dictionaryValue as? [String: Double] {
            confidence = probValue[label] ?? 0.0
        }

        return (label, confidence)
    }

    // MARK: - Keyword fallback (works if Core ML fails to load)

    private func keywordClassify(message: String) -> OnDevicePrediction {
        let intent = keywordIntent(for: message)
        let sentiment = keywordSentiment(for: message)
        let action = ActionRouter.shared.action(intent: intent, sentiment: sentiment)
        return OnDevicePrediction(
            intent: intent, intentConfidence: 0.0,
            sentiment: sentiment, sentimentConfidence: 0.0,
            action: action
        )
    }

    private func keywordIntent(for message: String) -> String {
        let text = message.lowercased()
        let rules: [(keywords: [String], intent: String)] = [
            (["saldo", "balance", "estado de cuenta", "movimientos"],            "consulta_saldo"),
            (["transferencia", "transferir", "spei", "enviar dinero", "depositar"], "transferencia"),
            (["fraude", "no reconozco", "no autoricé", "cargo extraño"],          "fraude_cargo_no_reconocido"),
            (["bloquear", "robaron", "extravié", "perdí mi tarjeta"],             "bloqueo_tarjeta"),
            (["cancelar mi cuenta", "darme de baja", "cerrar mi cuenta"],         "cancelar_cuenta"),
            (["cancelar cargo", "cancelar suscripción", "cobro recurrente"],      "cancelar_cargo"),
            (["solicitar", "tarjeta de crédito", "préstamo", "abrir cuenta"],     "solicitar_producto"),
            (["no funciona", "error", "no carga", "no me deja"],                  "problema_tecnico"),
            (["servicio terrible", "pésimo", "molesta", "queja"],                 "queja_servicio"),
            (["asesor", "agente humano", "comunicar"],                            "hablar_asesor"),
            (["cambiar nip", "actualizar teléfono", "cambiar correo"],            "cambio_datos_perfil"),
        ]
        for r in rules where r.keywords.contains(where: { text.contains($0) }) {
            return r.intent
        }
        return "informacion_general"
    }

    private func keywordSentiment(for message: String) -> String {
        let text = message.lowercased()
        if ["urgente", "ahora mismo", "inmediato", "fraude", "robaron", "ya"]
            .contains(where: { text.contains($0) }) {
            return "negativo_urgente"
        }
        if ["pésimo", "terrible", "molesto", "molesta", "decepcionado", "queja", "no me ayudan"]
            .contains(where: { text.contains($0) }) {
            return "negativo_queja"
        }
        if ["gracias", "excelente", "me interesa", "buenos días", "hola"]
            .contains(where: { text.contains($0) }) {
            return "positivo"
        }
        return "neutral"
    }
}
