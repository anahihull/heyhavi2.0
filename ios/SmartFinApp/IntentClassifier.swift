import CoreML
import Foundation

/// On-device intent classifier using the exported Core ML model.
/// Falls back gracefully if the model isn't available.
class IntentClassifier {

    private var model: intent_model?

    init() {
        do {
            let config = MLModelConfiguration()
            model = try intent_model(configuration: config)
            print("✅ Core ML intent model loaded successfully.")
        } catch {
            print("⚠️ Could not load Core ML model: \(error.localizedDescription)")
            print("   Make sure intent_model.mlmodel is added to the Xcode project.")
        }
    }

    /// Classify a message and return the predicted intent label.
    func classify(message: String) -> String {
        guard let model = model else {
            return "unknown"
        }
        do {
            let input = intent_modelInput(message: message)
            let output = try model.prediction(input: input)
            return output.intent
        } catch {
            print("Prediction error: \(error.localizedDescription)")
            return "unknown"
        }
    }
}
