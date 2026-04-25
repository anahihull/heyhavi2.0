// Responder.swift
// On-device Spanish response generator. Mirrors backend/responder.py so the app
// can answer fully offline using only the Core ML predictions + ActionRouter.
//
// Pipeline:
//     intent + sentiment  ──►  ActionRouter  ──►  action
//                                                   │
//                                                   ▼
//                            ACTION_RESPONSES[action]  ──►  base message + escalate
//                                                   │
//                                                   ▼
//                            +  sentiment opener   (SENTIMENT_OPENERS)
//                            +  intent hint        (INTENT_HINTS)

import Foundation

struct ResponderResult {
    let message: String
    let escalate: Bool
    let hint: String
}

enum Responder {

    // MARK: - Action templates (Spanish)

    private static let actionResponses: [String: (message: String, escalate: Bool)] = [
        "retener_cliente_urgente": (
            message: "Lamento mucho que te sientas así y entiendo perfectamente tu frustración. " +
                     "Es muy importante para nosotros que te quedes con nosotros. " +
                     "Antes de tomar cualquier decisión, me gustaría ofrecerte una solución personalizada " +
                     "y comunicarte de inmediato con un especialista que puede ayudarte. ¿Me permites un momento?",
            escalate: true
        ),
        "retener_cliente": (
            message: "Entiendo que estás considerando cerrar tu cuenta y queremos asegurarnos " +
                     "de que tu experiencia mejore. ¿Podrías contarme qué ha pasado? " +
                     "Estamos aquí para resolver cualquier inconveniente y encontrar la mejor solución para ti. " +
                     "Tenemos opciones que podrían ser de tu interés.",
            escalate: false
        ),
        "escalar_agente": (
            message: "Entiendo la importancia de tu situación. Voy a conectarte ahora mismo " +
                     "con un especialista del equipo de Hey Banco que podrá atenderte de forma personalizada. " +
                     "Por favor, mantente en la conversación.",
            escalate: true
        ),
        "resolver_problema": (
            message: "Lamento que estés teniendo este inconveniente. Voy a ayudarte a resolverlo ahora mismo. " +
                     "¿Puedes darme más detalles sobre lo que está ocurriendo para darte la solución más precisa?",
            escalate: false
        ),
        "procesar_solicitud": (
            message: "Con gusto proceso tu solicitud. Dame un momento para verificar tu información " +
                     "y llevarla a cabo de forma segura.",
            escalate: false
        ),
        "informar": (
            message: "Claro, con mucho gusto te ayudo. ¿En qué puedo orientarte hoy? " +
                     "Estoy aquí para responder todas tus preguntas sobre Hey Banco.",
            escalate: false
        ),
        "oferta_producto": (
            message: "¡Excelente! En Hey Banco tenemos productos diseñados para ti: " +
                     "tarjetas de crédito con cashback, préstamos personales con tasas competitivas, " +
                     "y cuentas con beneficios exclusivos. ¿Te gustaría conocer cuál se adapta mejor a tus necesidades?",
            escalate: false
        ),
    ]

    // MARK: - Intent hints (warmer sub-message)

    private static let intentHints: [String: String] = [
        "cancelar_cuenta":             "Si decides quedarte, tenemos beneficios exclusivos que pueden mejorar tu experiencia.",
        "fraude_cargo_no_reconocido":  "Vamos a revisar ese cargo de inmediato y proteger tu cuenta.",
        "bloqueo_tarjeta":             "Procederemos a bloquear tu tarjeta de forma inmediata para protegerte.",
        "transferencia":               "Tu transferencia se realizará de forma segura.",
        "solicitar_producto":          "Tenemos opciones ideales para tu perfil financiero.",
        "problema_tecnico":            "Revisaremos el problema técnico contigo paso a paso.",
        "queja_servicio":              "Tu retroalimentación es muy valiosa y nos ayuda a mejorar.",
        "hablar_asesor":               "Un asesor estará contigo en breve.",
        "consulta_saldo":              "Aquí tienes la información de tu cuenta.",
        "cambio_datos_perfil":         "Actualizaremos tus datos de forma segura.",
        "informacion_general":         "Con gusto te oriento.",
        "cancelar_cargo":              "Revisaremos ese cargo y tomaremos las medidas necesarias.",
    ]

    // MARK: - Sentiment opener

    private static let sentimentOpeners: [String: String] = [
        "negativo_urgente": "Entiendo que esto es urgente. ",
        "negativo_queja":   "Lamento mucho lo que estás viviendo. ",
        "neutral":          "",
        "positivo":         "",
    ]

    // MARK: - Public API

    static func generate(intent: String, sentiment: String, action: String) -> ResponderResult {
        let template = actionResponses[action] ?? actionResponses["informar"]!
        let opener   = sentimentOpeners[sentiment] ?? ""
        let hint     = intentHints[intent] ?? ""

        var message = (opener + template.message).trimmingCharacters(in: .whitespaces)
        if !hint.isEmpty {
            message += " " + hint
        }

        return ResponderResult(message: message, escalate: template.escalate, hint: hint)
    }
}
