import SwiftUI

// MARK: - Message Model

struct Message: Identifiable {
    let id = UUID()
    let text: String
    let isUser: Bool
    var riskLevel: String = "low"
    var intent: String = ""
}

// MARK: - Main Chat View

struct ContentView: View {
    @State private var messages: [Message] = [
        Message(
            text: "👋 Hello! I'm your Smart Financial Assistant. How can I help you today?",
            isUser: false
        )
    ]
    @State private var inputText = ""
    @State private var isLoading = false

    let classifier = IntentClassifier()
    let userId = "user_001"   // Change to logged-in user ID in production

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Chat message list
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 4) {
                            ForEach(messages) { msg in
                                MessageBubble(message: msg)
                                    .id(msg.id)
                            }
                            if isLoading {
                                TypingIndicator()
                            }
                        }
                        .padding(.vertical, 8)
                    }
                    .onChange(of: messages.count) { _ in
                        withAnimation {
                            proxy.scrollTo(messages.last?.id, anchor: .bottom)
                        }
                    }
                    .onChange(of: isLoading) { _ in
                        withAnimation {
                            proxy.scrollTo(messages.last?.id, anchor: .bottom)
                        }
                    }
                }

                Divider()

                // Input bar
                HStack(spacing: 12) {
                    TextField("Type a message...", text: $inputText)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .onSubmit { sendMessage() }
                        .disabled(isLoading)

                    Button(action: sendMessage) {
                        Image(systemName: isLoading ? "ellipsis.circle.fill" : "paperplane.fill")
                            .font(.system(size: 22))
                            .foregroundColor(inputText.isEmpty || isLoading ? .gray : .blue)
                    }
                    .disabled(inputText.isEmpty || isLoading)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(Color(.systemBackground))
            }
            .navigationTitle("💳 FinAssist")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    // MARK: - Send Message

    func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return }

        // Append user message
        messages.append(Message(text: text, isUser: true))
        inputText = ""
        isLoading = true

        // On-device intent preview (fast, works offline)
        let localIntent = classifier.classify(message: text)
        print("📱 Local intent: \(localIntent)")

        // Call backend for full risk + response
        APIService.sendMessage(userId: userId, message: text) { result in
            DispatchQueue.main.async {
                isLoading = false
                switch result {
                case .success(let apiResponse):
                    let riskEmoji: String
                    switch apiResponse.risk_level {
                    case "high":   riskEmoji = "🚨 "
                    case "medium": riskEmoji = "⚠️ "
                    default:       riskEmoji = ""
                    }
                    messages.append(Message(
                        text: riskEmoji + apiResponse.response,
                        isUser: false,
                        riskLevel: apiResponse.risk_level,
                        intent: apiResponse.intent
                    ))
                case .failure(let error):
                    messages.append(Message(
                        text: "⚠️ Could not connect to server. Make sure the backend is running.\n(\(error.localizedDescription))",
                        isUser: false,
                        riskLevel: "low"
                    ))
                }
            }
        }
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: Message

    var bubbleColor: Color {
        if message.isUser { return .blue }
        switch message.riskLevel {
        case "high":   return Color.red.opacity(0.12)
        case "medium": return Color.orange.opacity(0.12)
        default:       return Color(.systemGray6)
        }
    }

    var textColor: Color {
        message.isUser ? .white : .primary
    }

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if message.isUser { Spacer(minLength: 60) }

            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 4) {
                Text(message.text)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(bubbleColor)
                    .foregroundColor(textColor)
                    .cornerRadius(18)
                    .frame(maxWidth: 300, alignment: message.isUser ? .trailing : .leading)

                if !message.intent.isEmpty && !message.isUser {
                    Text("Intent: \(message.intent.replacingOccurrences(of: "_", with: " "))")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 4)
                }
            }

            if !message.isUser { Spacer(minLength: 60) }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 2)
    }
}

// MARK: - Typing Indicator

struct TypingIndicator: View {
    @State private var animate = false

    var body: some View {
        HStack(alignment: .bottom) {
            HStack(spacing: 4) {
                ForEach(0..<3) { i in
                    Circle()
                        .fill(Color.secondary)
                        .frame(width: 8, height: 8)
                        .offset(y: animate ? -4 : 0)
                        .animation(
                            .easeInOut(duration: 0.4)
                                .repeatForever()
                                .delay(Double(i) * 0.15),
                            value: animate
                        )
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color(.systemGray6))
            .cornerRadius(18)
            Spacer()
        }
        .padding(.horizontal, 12)
        .onAppear { animate = true }
    }
}

// MARK: - Preview

#Preview {
    ContentView()
}
