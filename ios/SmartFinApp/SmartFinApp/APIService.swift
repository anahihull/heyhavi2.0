import Foundation

// MARK: - Request / Response Models

struct ChatRequest: Codable {
    let user_id: String
    let message: String
    let transaction_amount: Double?
    let transaction_category: String?
    let failed_attempts: Int?
    let is_anomaly: Bool?
    let transaction_country: String?
}

struct ChatResponse: Codable {
    let user_id: String
    let message: String
    let intent: String
    let intent_confidence: Double
    let sentiment: String
    let sentiment_confidence: Double
    let action: String
    let escalate: Bool
    let risk_level: String
    let risk_score: Int
    let risk_flags: [String]
    let response: String
    let response_hint: String
}

// MARK: - Converse (multi-turn LLM)

struct ConverseTurn: Codable {
    let role: String   // "user" or "assistant"
    let content: String
}

struct ConverseRequest: Codable {
    let user_id: String
    let action: String
    let message: String
    let history: [ConverseTurn]
}

struct ConverseResponse: Codable {
    let user_id: String
    let action: String
    let message: String
    let resolved: Bool
    let escalate: Bool
}

// MARK: - API Service

class APIService {

    // ✅ Simulator: localhost works fine.
    // 📱 Real iPhone: change to your Mac's local IP, e.g. "http://192.168.1.42:8000"
    //    Find it with: System Settings → Wi-Fi → Details → IP Address
    static let baseURL = "http://localhost:8000"

    static func sendMessage(
        userId: String,
        message: String,
        transactionAmount: Double? = nil,
        transactionCategory: String? = nil,
        failedAttempts: Int? = nil,
        isAnomaly: Bool? = nil,
        transactionCountry: String? = nil,
        completion: @escaping (Result<ChatResponse, Error>) -> Void
    ) {
        guard let url = URL(string: "\(baseURL)/chat") else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 10.0

        let body = ChatRequest(
            user_id: userId,
            message: message,
            transaction_amount: transactionAmount,
            transaction_category: transactionCategory,
            failed_attempts: failedAttempts,
            is_anomaly: isAnomaly,
            transaction_country: transactionCountry
        )

        guard let encoded = try? JSONEncoder().encode(body) else { return }
        request.httpBody = encoded

        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    completion(.failure(error))
                    return
                }
                guard let data = data else { return }
                do {
                    let decoded = try JSONDecoder().decode(ChatResponse.self, from: data)
                    completion(.success(decoded))
                } catch {
                    print("❌ Decode error: \(error)")
                    completion(.failure(error))
                }
            }
        }.resume()
    }

    /// Multi-turn LLM conversation. Call this on every user turn while the
    /// app is in conversation mode (i.e. after the on-device classifier
    /// picked an LLM-enabled action). The server returns the assistant
    /// reply plus `resolved` / `escalate` control flags.
    static func converse(
        userId: String,
        action: String,
        message: String,
        history: [ConverseTurn],
        completion: @escaping (Result<ConverseResponse, Error>) -> Void
    ) {
        guard let url = URL(string: "\(baseURL)/converse") else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30.0  // LLM calls can be slower than /chat

        let body = ConverseRequest(
            user_id: userId,
            action: action,
            message: message,
            history: history
        )

        guard let encoded = try? JSONEncoder().encode(body) else { return }
        request.httpBody = encoded

        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    completion(.failure(error))
                    return
                }
                guard let data = data else { return }
                do {
                    let decoded = try JSONDecoder().decode(ConverseResponse.self, from: data)
                    completion(.success(decoded))
                } catch {
                    print("❌ Converse decode error: \(error)")
                    completion(.failure(error))
                }
            }
        }.resume()
    }
}
