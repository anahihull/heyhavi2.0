"""
Response generator for the Smart Financial Assistant.

The pipeline upstream (model_loader.predict_full) decides three things from
the user's message:
    intent      → e.g. "fraude_cargo_no_reconocido"
    sentiment   → e.g. "negativo_urgente"
    action      → mapped from (intent, sentiment) via ml/action_map.json,
                  e.g. "escalar_agente"

This module turns those three labels (plus any risk flags) into a single
Spanish reply ready for the iOS app:

    {
        "message":  "Lamento lo que está ocurriendo...",
        "escalate": True,
        "hint":     "Vamos a revisar ese cargo de inmediato y proteger tu cuenta.",
    }
"""

from typing import Optional


# ── Action templates ─────────────────────────────────────────────────────────
# These keys match the action set produced by ml/action_map.json.
# Each action has a base Spanish message and a default "escalate to human"
# preference.

ACTION_RESPONSES: dict[str, dict] = {
    "retener_cliente_urgente": {
        "message": (
            "Lamento mucho que te sientas así y entiendo perfectamente tu frustración. "
            "Es muy importante para nosotros que te quedes con nosotros. "
            "Antes de tomar cualquier decisión, me gustaría ofrecerte una solución personalizada "
            "y comunicarte de inmediato con un especialista que puede ayudarte. ¿Me permites un momento?"
        ),
        "escalate": True,
    },
    "retener_cliente": {
        "message": (
            "Entiendo que estás considerando cerrar tu cuenta y queremos asegurarnos "
            "de que tu experiencia mejore. ¿Podrías contarme qué ha pasado? "
            "Estamos aquí para resolver cualquier inconveniente y encontrar la mejor solución para ti. "
            "Tenemos opciones que podrían ser de tu interés."
        ),
        "escalate": False,
    },
    "escalar_agente": {
        "message": (
            "Entiendo la importancia de tu situación. Voy a conectarte ahora mismo "
            "con un especialista del equipo de Hey Banco que podrá atenderte de forma personalizada. "
            "Por favor, mantente en la conversación."
        ),
        "escalate": True,
    },
    "resolver_problema": {
        "message": (
            "Lamento que estés teniendo este inconveniente. Voy a ayudarte a resolverlo ahora mismo. "
            "¿Puedes darme más detalles sobre lo que está ocurriendo para darte la solución más precisa?"
        ),
        "escalate": False,
    },
    "procesar_solicitud": {
        "message": (
            "Con gusto proceso tu solicitud. Dame un momento para verificar tu información "
            "y llevarla a cabo de forma segura."
        ),
        "escalate": False,
    },
    "informar": {
        "message": (
            "Claro, con mucho gusto te ayudo. ¿En qué puedo orientarte hoy? "
            "Estoy aquí para responder todas tus preguntas sobre Hey Banco."
        ),
        "escalate": False,
    },
    "oferta_producto": {
        "message": (
            "¡Excelente! En Hey Banco tenemos productos diseñados para ti: "
            "tarjetas de crédito con cashback, préstamos personales con tasas competitivas, "
            "y cuentas con beneficios exclusivos. ¿Te gustaría conocer cuál se adapta mejor a tus necesidades?"
        ),
        "escalate": False,
    },
}


# ── Intent-specific warm hints (appended after the action message) ───────────

INTENT_HINTS: dict[str, str] = {
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
}


# ── Sentiment-driven empathy prefix ──────────────────────────────────────────
# A tiny opener that softens the message when the customer is upset.

SENTIMENT_OPENERS: dict[str, str] = {
    "negativo_urgente": "Entiendo que esto es urgente. ",
    "negativo_queja":   "Lamento mucho lo que estás viviendo. ",
    "neutral":          "",
    "positivo":         "",
}


# Risk-flag keywords that should force a human handoff regardless of action.
# (These match the kinds of strings risk.assess_risk emits.)
ESCALATE_FLAG_KEYWORDS = ("fraud", "anomal", "foreign", "5x above")


def generate_response(
    action: str,
    intent: str,
    sentiment: str,
    risk_flags: Optional[list[str]] = None,
) -> dict:
    """
    Build the assistant's reply.

    Parameters
    ----------
    action : str
        Action chosen by the action_map (e.g. "escalar_agente").
    intent : str
        Predicted intent label (e.g. "fraude_cargo_no_reconocido").
    sentiment : str
        Predicted sentiment label (e.g. "negativo_urgente").
    risk_flags : list[str] | None
        Human-readable risk reasons from risk.assess_risk.

    Returns
    -------
    dict with keys:
        message  : final Spanish text to show the user
        escalate : whether to route to a human agent
        hint     : intent-specific sub-message (also included in `message`)
    """
    risk_flags = risk_flags or []

    template = ACTION_RESPONSES.get(action, ACTION_RESPONSES["informar"])
    base_msg = template["message"]
    escalate = template["escalate"]

    # 1. Sentiment-aware empathy prefix
    opener = SENTIMENT_OPENERS.get(sentiment, "")

    # 2. Intent-specific warm hint
    hint = INTENT_HINTS.get(intent, "")

    # 3. Compose final message
    message = f"{opener}{base_msg}".strip()
    if hint:
        message = f"{message} {hint}"

    # 4. Force-escalate if a risk flag mentions fraud/anomaly/foreign/big-jump
    if any(any(k in flag.lower() for k in ESCALATE_FLAG_KEYWORDS) for flag in risk_flags):
        escalate = True

    return {
        "message":  message,
        "escalate": escalate,
        "hint":     hint,
    }


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Responder Test ===\n")
    cases = [
        ("informar",                 "consulta_saldo",             "neutral",          []),
        ("procesar_solicitud",       "transferencia",              "neutral",          []),
        ("retener_cliente_urgente",  "cancelar_cuenta",            "negativo_urgente", []),
        ("escalar_agente",           "fraude_cargo_no_reconocido", "negativo_urgente", ["fraud_alert"]),
        ("oferta_producto",          "solicitar_producto",         "positivo",         []),
        ("resolver_problema",        "problema_tecnico",           "negativo_queja",   []),
    ]
    for action, intent, sentiment, flags in cases:
        out = generate_response(action, intent, sentiment, flags)
        print(f"action={action} intent={intent} sentiment={sentiment} → escalate={out['escalate']}")
        print(f"  {out['message']}\n")
