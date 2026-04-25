import SwiftUI

// MARK: - Cross-platform helpers (so the file builds for both iOS and macOS)

extension Color {
    /// Equivalent of UIColor.systemGray6 on iOS; a sensible fallback on macOS.
    static var appSecondaryBackground: Color {
        #if os(iOS)
        return Color(.systemGray6)
        #else
        return Color.gray.opacity(0.15)
        #endif
    }

    /// Equivalent of UIColor.systemBackground on iOS; a sensible fallback on macOS.
    static var appBackground: Color {
        #if os(iOS)
        return Color(.systemBackground)
        #else
        return Color(NSColor.windowBackgroundColor)
        #endif
    }
}

extension View {
    /// Apply `.navigationBarTitleDisplayMode(.inline)` only on iOS.
    @ViewBuilder
    func inlineNavTitleIfAvailable() -> some View {
        #if os(iOS)
        self.navigationBarTitleDisplayMode(.inline)
        #else
        self
        #endif
    }
}

struct ContentView: View {
    @StateObject private var viewModel = ChatViewModel()
    @State private var inputText = ""
    @FocusState private var inputFocused: Bool

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {

                // ── Message list ──────────────────────────────────────────
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 6) {
                            ForEach(viewModel.messages) { msg in
                                MessageBubble(message: msg)
                                    .id(msg.id)
                            }
                            if viewModel.isLoading {
                                TypingIndicator()
                                    .id("typing")
                            }
                        }
                        .padding(.vertical, 10)
                    }
                    .onChange(of: viewModel.messages.count) { _ in
                        scrollToBottom(proxy: proxy)
                    }
                    .onChange(of: viewModel.isLoading) { _ in
                        scrollToBottom(proxy: proxy)
                    }
                }

                Divider()

                // ── Input bar ─────────────────────────────────────────────
                HStack(spacing: 10) {
                    TextField("Escribe tu mensaje...", text: $inputText)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(Color.appSecondaryBackground)
                        .cornerRadius(22)
                        .focused($inputFocused)
                        .onSubmit { send() }

                    Button(action: send) {
                        ZStack {
                            Circle()
                                .fill(canSend ? Color.blue : Color.gray.opacity(0.3))
                                .frame(width: 44, height: 44)
                            Image(systemName: "paperplane.fill")
                                .font(.system(size: 16, weight: .semibold))
                                .foregroundColor(.white)
                        }
                    }
                    .disabled(!canSend)
                    .animation(.easeInOut(duration: 0.15), value: canSend)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(Color.appBackground)
            }
            .navigationTitle("💳 FinAssist")
            .inlineNavTitleIfAvailable()
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    RiskLegendButton()
                }
            }
        }
    }

    // MARK: - Helpers

    private var canSend: Bool {
        !inputText.trimmingCharacters(in: .whitespaces).isEmpty && !viewModel.isLoading
    }

    private func send() {
        let text = inputText
        inputText = ""
        viewModel.send(text: text)
    }

    private func scrollToBottom(proxy: ScrollViewProxy) {
        withAnimation(.easeOut(duration: 0.25)) {
            if viewModel.isLoading {
                proxy.scrollTo("typing", anchor: .bottom)
            } else if let last = viewModel.messages.last {
                proxy.scrollTo(last.id, anchor: .bottom)
            }
        }
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: Message

    private var bubbleColor: Color {
        if message.isUser { return .blue }
        switch message.riskLevel {
        case "high":   return Color.red.opacity(0.1)
        case "medium": return Color.orange.opacity(0.1)
        default:       return Color.appSecondaryBackground
        }
    }

    private var borderColor: Color {
        switch message.riskLevel {
        case "high":   return .red.opacity(0.4)
        case "medium": return .orange.opacity(0.4)
        default:       return .clear
        }
    }

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if message.isUser { Spacer(minLength: 60) }

            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 4) {
                Text(message.text)
                    .font(.body)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(bubbleColor)
                    .foregroundColor(message.isUser ? .white : .primary)
                    .cornerRadius(18)
                    .overlay(
                        RoundedRectangle(cornerRadius: 18)
                            .stroke(borderColor, lineWidth: 1)
                    )
                    .frame(maxWidth: 300, alignment: message.isUser ? .trailing : .leading)

                // Intent + confidence tag (assistant messages only)
                if !message.intent.isEmpty && !message.isUser {
                    HStack(spacing: 4) {
                        Image(systemName: intentIcon(message.intent))
                            .font(.caption2)
                        Text("\(message.intent.replacingOccurrences(of: "_", with: " ")) · \(Int(message.confidence * 100))%")
                            .font(.caption2)
                    }
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 6)
                }
            }

            if !message.isUser { Spacer(minLength: 60) }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 2)
    }

    private func intentIcon(_ intent: String) -> String {
        switch intent {
        case "consulta_saldo":              return "banknote"
        case "transferencia":               return "arrow.left.arrow.right"
        case "fraude_cargo_no_reconocido":  return "exclamationmark.shield"
        case "bloqueo_tarjeta":             return "lock.shield"
        case "cancelar_cuenta":             return "xmark.circle"
        case "cancelar_cargo":              return "minus.circle"
        case "solicitar_producto":          return "tag"
        case "problema_tecnico":            return "wrench.and.screwdriver"
        case "queja_servicio":              return "exclamationmark.bubble"
        case "hablar_asesor":               return "person.fill.questionmark"
        case "cambio_datos_perfil":         return "person.crop.circle.badge.checkmark"
        case "informacion_general":         return "info.circle"
        default:                            return "bubble.left"
        }
    }
}

// MARK: - Typing Indicator

struct TypingIndicator: View {
    @State private var animate = false

    var body: some View {
        HStack(alignment: .bottom) {
            HStack(spacing: 5) {
                ForEach(0..<3, id: \.self) { i in
                    Circle()
                        .fill(Color.secondary.opacity(0.6))
                        .frame(width: 8, height: 8)
                        .scaleEffect(animate ? 1.2 : 0.8)
                        .animation(
                            .easeInOut(duration: 0.45)
                                .repeatForever(autoreverses: true)
                                .delay(Double(i) * 0.15),
                            value: animate
                        )
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(Color.appSecondaryBackground)
            .cornerRadius(18)

            Spacer()
        }
        .padding(.horizontal, 12)
        .onAppear { animate = true }
    }
}

// MARK: - Risk Legend

struct RiskLegendButton: View {
    @State private var showSheet = false

    var body: some View {
        Button(action: { showSheet = true }) {
            Image(systemName: "info.circle")
        }
        .sheet(isPresented: $showSheet) {
            RiskLegendView()
        }
    }
}

struct RiskLegendView: View {
    var body: some View {
        NavigationView {
            List {
                Section("Niveles de Riesgo") {
                    Label("Bajo — actividad normal", systemImage: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Label("Medio — revisión recomendada", systemImage: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    Label("Alto — acción inmediata necesaria", systemImage: "xmark.octagon.fill")
                        .foregroundColor(.red)
                }
                Section("Categorías de Intención") {
                    Label("Consulta de saldo", systemImage: "banknote")
                    Label("Transferencia", systemImage: "arrow.left.arrow.right")
                    Label("Alerta de fraude", systemImage: "exclamationmark.shield")
                    Label("Solicitar producto", systemImage: "tag")
                    Label("Hablar con asesor", systemImage: "person.fill.questionmark")
                    Label("Bloqueo de tarjeta", systemImage: "lock.shield")
                }
            }
            .navigationTitle("Cómo funciona")
            .inlineNavTitleIfAvailable()
        }
    }
}

// MARK: - Preview

#Preview {
    ContentView()
}
