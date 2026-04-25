import Foundation

struct Message: Identifiable {
    let id = UUID()
    let text: String
    let isUser: Bool
    var riskLevel: String = "low"
    var intent: String = ""
    var confidence: Double = 0.0
}
