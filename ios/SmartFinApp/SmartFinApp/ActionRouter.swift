// ActionRouter.swift
// Maps (intent, sentiment) → action using the action_map.json bundled with the app.
// Mirrors the Python ACTION_MAP in build_dataset.py so the iOS app can route
// customer interactions fully on-device.
//
// Usage:
//   let action = ActionRouter.shared.action(intent: "cancelar_cuenta",
//                                            sentiment: "negativo_queja")
//   // → "retener_cliente"

import Foundation

final class ActionRouter {

    static let shared = ActionRouter()

    private var rules: [String: String] = [:]

    private init() {
        guard
            let url  = Bundle.main.url(forResource: "action_map", withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let map  = try? JSONDecoder().decode([String: String].self, from: data)
        else {
            print("⚠️ ActionRouter: action_map.json not found in bundle.")
            return
        }
        rules = map
    }

    /// Returns the recommended action for a given (intent, sentiment) pair.
    /// Falls back to `"informar"` when no rule matches.
    func action(intent: String, sentiment: String) -> String {
        let key = "\(intent)|\(sentiment)"
        return rules[key] ?? "informar"
    }

    // MARK: - Action labels for UI
    static func displayName(for action: String) -> String {
        switch action {
        case "retener_cliente_urgente": return "🚨 Retención urgente"
        case "retener_cliente":         return "💛 Retener cliente"
        case "escalar_agente":          return "👤 Escalar a agente"
        case "resolver_problema":       return "🔧 Resolver problema"
        case "procesar_solicitud":      return "✅ Procesar solicitud"
        case "oferta_producto":         return "🎁 Ofrecer producto"
        case "informar":                return "ℹ️ Informar"
        default:                        return action
        }
    }
}
