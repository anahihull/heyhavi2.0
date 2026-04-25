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
    let risk_level: String
    let risk_score: Int
    let risk_flags: [String]
    let response: String
}

// MARK: - API Service

class APIService {

    // ⚠️ If testing on a real iPhone (not Simulator), replace with your Mac's local IP.
    // Example: "http://192.168.1.100:8000"
    // Find your IP with: ifconfig | grep "inet " (Mac terminal)
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
        guard let url = URL(string: "\(baseURL)/chat") else {
            print("❌ Invalid URL")
            return
        }

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

        do {
            request.httpBody = try JSONEncoder().encode(body)
        } catch {
            completion(.failure(error))
            return
        }

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            guard let data = data else {
                completion(.failure(NSError(
                    domain: "APIService",
                    code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "No data received"]
                )))
                return
            }
            do {
                let decoded = try JSONDecoder().decode(ChatResponse.self, from: data)
                completion(.success(decoded))
            } catch {
                print("Decode error: \(error)")
                completion(.failure(error))
            }
        }.resume()
    }
}
