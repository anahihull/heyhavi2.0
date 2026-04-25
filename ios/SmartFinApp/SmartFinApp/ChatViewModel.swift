import Foundation
import Combine

/// Owns all chat state and business logic.
/// ContentView observes this object and re-renders automatically.
///
/// Architecture:
///   1. NEUTRAL MODE — every send runs the on-device Core ML pipeline
///      (intent + sentiment + action). The Responder turns its output into
///      a Spanish message that is shown immediately so the app is fully
///      usable offline. In parallel, the backend's /chat endpoint is
///      called to enrich the reply with server-side risk analysis.
///
///   2. CONVERSATION MODE — once /chat returns an action that is
///      LLM-enabled (retener_cliente, escalar_agente, etc.), the view
///      model latches that action and starts routing every subsequent
///      user turn through /converse with growing history. The LLM owns
///      the conversation until it returns `resolved=true`, at which
///      point the view model drops back to neutral mode.
@MainActor
class ChatViewModel: ObservableObject {

    // Published state — any change triggers a SwiftUI re-render
    @Published var messages: [Message] = [
        Message(
            text: "👋 ¡Hola! Soy tu Asistente Financiero de Hey Banco. ¿En qué puedo ayudarte hoy?",
            isUser: false
        )
    ]
    @Published var isLoading = false
    @Published var errorMessage: String? = nil

    // Dependencies
    private let classifier = OnDeviceClassifier.shared
    let userId = "user_001"   // Replace with real auth user ID

    // MARK: - Conversation mode state

    /// The LLM-enabled action currently driving the conversation.
    /// `nil` means we are in neutral classification mode.
    private var conversationAction: String? = nil

    /// Turns accumulated in the current LLM conversation. The latest user
    /// turn is *not* included here — it's sent as `message` on /converse.
    private var conversationHistory: [ConverseTurn] = []

    /// Backend actions for which we run a multi-turn LLM dialog. Must be a
    /// subset of `LLM_ENABLED_ACTIONS` on the server (backend/conversation.py).
    private static let llmEnabledActions: Set<String> = [
        "retener_cliente_urgente",
        "retener_cliente",
        "escalar_agente",
        "resolver_problema",
        "procesar_solicitud",
        "oferta_producto",
        "informar",
    ]

    // MARK: - Send a message

    func send(text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }

        // 1. Append the user's message immediately
        messages.append(Message(text: trimmed, isUser: true))
        isLoading = true
        errorMessage = nil

        // 2. If we are mid-LLM conversation, route this turn straight to
        //    /converse and skip on-device classification — Claude is
        //    already steering the conversation.
        if let action = conversationAction {
            sendToConverse(action: action, userMessage: trimmed)
            return
        }

        // 3. NEUTRAL MODE — on-device pipeline (instant, offline, Spanish)
        let local = classifier.classify(message: trimmed)
        let localReply = Responder.generate(
            intent:    local.intent,
            sentiment: local.sentiment,
            action:    local.action
        )
        print("📱 On-device → intent=\(local.intent) sentiment=\(local.sentiment) action=\(local.action)")

        // Show the on-device reply right away — the app works fully offline.
        let localMessage = Message(
            text: localReply.message,
            isUser: false,
            riskLevel: "low",
            intent: local.intent,
            confidence: local.intentConfidence
        )
        messages.append(localMessage)
        let localMessageID = localMessage.id

        // 4. BACKEND enrichment via /chat. On success, upgrade the reply.
        //    If the resulting action is LLM-enabled, latch into conversation
        //    mode so that the next user turn goes through /converse.
        APIService.sendMessage(userId: userId, message: trimmed) { [weak self] result in
            guard let self = self else { return }
            self.isLoading = false

            switch result {
            case .success(let api):
                let riskEmoji: String
                switch api.risk_level {
                case "high":   riskEmoji = "🚨 "
                case "medium": riskEmoji = "⚠️ "
                default:       riskEmoji = ""
                }
                let enrichedText = riskEmoji + api.response
                if let idx = self.messages.firstIndex(where: { $0.id == localMessageID }) {
                    self.messages[idx] = Message(
                        text: enrichedText,
                        isUser: false,
                        riskLevel: api.risk_level,
                        intent: api.intent,
                        confidence: api.intent_confidence
                    )
                }

                // Latch into conversation mode if applicable.
                if Self.llmEnabledActions.contains(api.action) {
                    self.conversationAction = api.action
                    // Seed history with this opening exchange so Claude has
                    // context for the next turn.
                    self.conversationHistory = [
                        ConverseTurn(role: "user", content: trimmed),
                        ConverseTurn(role: "assistant", content: enrichedText),
                    ]
                    print("🤖 Entering conversation mode → action=\(api.action)")
                }

            case .failure(let error):
                // Backend unreachable — keep the on-device reply, log only.
                print("ℹ️ Backend offline (\(error.localizedDescription)). Using on-device reply.")
            }
        }
    }

    // MARK: - Conversation-mode helpers

    private func sendToConverse(action: String, userMessage: String) {
        let history = conversationHistory

        APIService.converse(
            userId: userId,
            action: action,
            message: userMessage,
            history: history
        ) { [weak self] result in
            guard let self = self else { return }
            self.isLoading = false

            switch result {
            case .success(let api):
                let prefix = api.escalate ? "🤝 " : ""
                let assistantText = prefix + api.message

                self.messages.append(Message(
                    text: assistantText,
                    isUser: false,
                    riskLevel: api.escalate ? "medium" : "low",
                    intent: action,
                    confidence: 1.0
                ))

                // Grow history with this turn for the next round.
                self.conversationHistory.append(
                    ConverseTurn(role: "user", content: userMessage)
                )
                self.conversationHistory.append(
                    ConverseTurn(role: "assistant", content: api.message)
                )

                // Exit conversation mode when Claude says we're done.
                if api.resolved {
                    print("✅ Conversation resolved (action=\(action), escalate=\(api.escalate)) — back to neutral mode")
                    self.conversationAction = nil
                    self.conversationHistory.removeAll()
                }

            case .failure(let error):
                // /converse failed — fall back to on-device responder so the
                // user still gets a reply, and stay in conversation mode so
                // they can retry. Don't pollute history with a failed turn.
                print("⚠️ /converse failed (\(error.localizedDescription)). Falling back to on-device reply.")

                let local = self.classifier.classify(message: userMessage)
                let localReply = Responder.generate(
                    intent:    local.intent,
                    sentiment: local.sentiment,
                    action:    local.action
                )
                self.messages.append(Message(
                    text: localReply.message,
                    isUser: false,
                    riskLevel: "low",
                    intent: local.intent,
                    confidence: local.intentConfidence
                ))
            }
        }
    }
}
