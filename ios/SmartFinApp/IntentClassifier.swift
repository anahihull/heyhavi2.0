import Foundation

/// On-device intent classifier.
///
/// RIGHT NOW: uses a fast keyword matcher so the app builds without the .mlmodel file.
/// LATER: once Person 1 runs export_coreml.py and you drag intent_model.mlmodel
///        into Xcode, swap in the Core ML section below (just uncomment it).

class IntentClassifier {

    // MARK: - Public API

    func classify(message: String) -> String {
        return keywordClassify(message: message)

        // ── Core ML version (uncomment after adding intent_model.mlmodel to Xcode) ──
        // return coreMLClassify(message: message) ?? keywordClassify(message: message)
    }

    // MARK: - Keyword-based fallback (works with no model file)

    private func keywordClassify(message: String) -> String {
        let text = message.lowercased()

        let rules: [(keywords: [String], intent: String)] = [
            (["balance", "how much", "how much money", "funds available"],          "check_balance"),
            (["transfer", "send", "wire", "pay ", "payment to"],                    "transfer_money"),
            (["fraud", "suspicious", "didn't make", "don't recognize",
              "unauthorized", "stolen", "weird charge", "strange charge"],          "fraud_alert"),
            (["product", "offer", "savings account", "credit card",
              "investment", "loan", "interest rate"],                               "product_inquiry"),
            (["help", "support", "agent", "can't log", "reset", "password",
              "pin", "assistance", "problem"],                                      "customer_support"),
            (["history", "recent", "last transactions", "purchases",
              "spending", "statement", "transactions"],                             "transaction_history"),
        ]

        for rule in rules {
            if rule.keywords.contains(where: { text.contains($0) }) {
                return rule.intent
            }
        }

        return "customer_support"   // safe default
    }

    // MARK: - Core ML version (needs intent_model.mlmodel in the Xcode project)
    //
    // Steps to enable:
    //   1. Person 1 runs: cd ml && python export_coreml.py
    //   2. Drag intent_model.mlmodel into Xcode (Copy items if needed)
    //   3. Build once (Cmd+B) so Xcode generates the Swift class
    //   4. Uncomment the block below and the call in classify() above
    //
    // private var mlModel: intent_model?
    //
    // private func coreMLClassify(message: String) -> String? {
    //     if mlModel == nil {
    //         mlModel = try? intent_model(configuration: MLModelConfiguration())
    //     }
    //     guard let model = mlModel,
    //           let input = try? intent_modelInput(message: message),
    //           let output = try? model.prediction(input: input) else {
    //         return nil
    //     }
    //     return output.intent
    // }
}
