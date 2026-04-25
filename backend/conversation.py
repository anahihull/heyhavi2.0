"""
LLM-powered conversational layer.

After the on-device classifier picks an action (e.g. "retener_cliente"),
the iOS app drops into a multi-turn conversation with Claude. This module
provides:

    - ACTION_PROMPTS:   action-specific Spanish system prompts
    - converse():       single function that takes (action, history, message)
                        and returns {message, resolved, escalate}

Claude is instructed to reply in strict JSON so the iOS app can:
    - render the assistant message
    - know when the conversation is *resolved* (action complete) and exit
      back to neutral classification mode
    - know when to *escalate* to a human agent
"""

from __future__ import annotations

import json
import os
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Lazily initialized so the backend boots even without an API key
# (the /converse endpoint will return a clear error in that case).
_client: Optional[Anthropic] = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to backend/.env "
                "(see backend/.env.example) and restart the server."
            )
        _client = Anthropic(api_key=api_key)
    return _client


# ── Brand voice & shared formatting rules ────────────────────────────────────

BRAND_VOICE = """\
Eres un asistente conversacional de Hey Banco. Hablas español mexicano, tono
cálido, profesional y empático. Tuteas al cliente. Eres directo pero humano:
nada de respuestas robóticas o demasiado formales. Nunca inventas datos
financieros que no te dieron (saldos, montos, nombres). Si necesitas
información que no tienes, pídela amablemente."""

OUTPUT_FORMAT = """\
RESPONDE SIEMPRE en JSON válido y nada más, con exactamente este formato:

{"message": "<tu respuesta al cliente en español>", "resolved": <true|false>, "escalate": <true|false>}

Reglas:
- "message": lo que ves al cliente. Una a tres oraciones, conversacional.
- "resolved": true SOLO cuando la conversación llegó a un cierre natural
  (cliente confirmó la acción, agradeció y se despide, decidió no continuar,
  o tema completamente resuelto). false si todavía hay algo que aclarar o
  el cliente pidió más detalles.
- "escalate": true si el cliente pide explícitamente hablar con humano,
  o si detectas riesgo alto (fraude confirmado, amenaza legal, urgencia
  crítica). false en casos normales.
- NO uses markdown, NO uses backticks, NO antepongas texto. SOLO el JSON."""


# ── Action-specific system prompts ───────────────────────────────────────────
# These guide the model's strategy depending on what the customer wants to do.

ACTION_PROMPTS: dict[str, str] = {
    "retener_cliente_urgente": (
        BRAND_VOICE + "\n\n"
        "Contexto: el cliente está MUY MOLESTO y quiere cancelar su cuenta. "
        "Tu objetivo principal es retenerlo, pero respetando su decisión final.\n"
        "Estrategia:\n"
        "1) Reconoce su frustración con empatía genuina antes de ofrecer nada.\n"
        "2) Pregunta qué fue lo que pasó para entender la causa raíz.\n"
        "3) Ofrece soluciones específicas: condonar comisiones, escalar con un "
        "   especialista de retención, ofrecer un plan personalizado.\n"
        "4) Si después de 2-3 intercambios el cliente sigue firme en cancelar, "
        "   acepta su decisión, agradécele su tiempo y marca resolved=true.\n"
        "5) Nunca seas insistente al punto de molestar. La retención debe sentirse "
        "   como ayuda, no como presión.\n\n"
        + OUTPUT_FORMAT
    ),

    "retener_cliente": (
        BRAND_VOICE + "\n\n"
        "Contexto: el cliente está pensando en cancelar su cuenta o cambiar de banco.\n"
        "Estrategia:\n"
        "1) Pregunta qué motiva su decisión sin juzgar.\n"
        "2) Resalta beneficios relevantes: cashback, comisiones bajas, atención 24/7, "
        "   productos exclusivos como Hey Pro.\n"
        "3) Si el cliente confirma que se queda, marca resolved=true.\n"
        "4) Si insiste en cancelar, agradece, ofrece el proceso de cancelación y "
        "   marca resolved=true.\n\n"
        + OUTPUT_FORMAT
    ),

    "escalar_agente": (
        BRAND_VOICE + "\n\n"
        "Contexto: la situación requiere un agente humano. Tu rol es preparar "
        "el caso para que el agente lo retome con todo el contexto.\n"
        "Estrategia:\n"
        "1) Confirma al cliente que lo vas a conectar con un especialista.\n"
        "2) Pregunta los datos clave que el agente necesitará (número de "
        "   tarjeta enmascarado, fecha del incidente, monto, descripción breve).\n"
        "3) Una vez que tengas la información esencial, marca escalate=true y "
        "   resolved=true, despídete confirmando que un agente lo contactará.\n\n"
        + OUTPUT_FORMAT
    ),

    "resolver_problema": (
        BRAND_VOICE + "\n\n"
        "Contexto: el cliente tiene un problema técnico o de servicio. Tu rol "
        "es diagnosticar y resolver.\n"
        "Estrategia:\n"
        "1) Pregunta detalles específicos: dispositivo, mensaje de error, qué "
        "   estaba intentando hacer.\n"
        "2) Ofrece pasos concretos en orden (cerrar sesión y volver a entrar, "
        "   actualizar app, revisar conexión). Uno a la vez.\n"
        "3) Si el problema se resuelve, confirma con el cliente y marca "
        "   resolved=true.\n"
        "4) Si después de 2-3 pasos no se resuelve, marca escalate=true y "
        "   resolved=true para enviarlo a soporte técnico humano.\n\n"
        + OUTPUT_FORMAT
    ),

    "procesar_solicitud": (
        BRAND_VOICE + "\n\n"
        "Contexto: el cliente quiere ejecutar una operación (transferencia, "
        "cambio de datos, cancelación de cargo, etc.).\n"
        "Estrategia:\n"
        "1) Confirma exactamente qué quiere hacer.\n"
        "2) Pide los datos necesarios uno a uno (cuenta destino, monto, "
        "   concepto, etc.) sin abrumar.\n"
        "3) Antes de 'ejecutar', resume la operación y pide confirmación "
        "   explícita ('¿confirmas que quieres transferir $500 a la cuenta "
        "   1234?').\n"
        "4) Una vez confirmado, simula la ejecución y marca resolved=true.\n\n"
        + OUTPUT_FORMAT
    ),

    "oferta_producto": (
        BRAND_VOICE + "\n\n"
        "Contexto: el cliente está interesado en un producto financiero.\n"
        "Estrategia:\n"
        "1) Pregunta para qué lo necesita y su perfil (ingresos aproximados, "
        "   uso esperado), sin pedir datos sensibles.\n"
        "2) Recomienda el producto más adecuado de Hey Banco (Hey Pro, tarjeta "
        "   de crédito, préstamo personal, cuenta de inversión) explicando "
        "   por qué encaja.\n"
        "3) Ofrece los siguientes pasos para contratarlo.\n"
        "4) Marca resolved=true cuando el cliente decide (positivo o negativo).\n\n"
        + OUTPUT_FORMAT
    ),

    "informar": (
        BRAND_VOICE + "\n\n"
        "Contexto: el cliente tiene una pregunta general sobre Hey Banco.\n"
        "Estrategia:\n"
        "1) Responde la pregunta de forma clara y breve.\n"
        "2) Si necesita más, pregunta qué le gustaría profundizar.\n"
        "3) Marca resolved=true cuando ya respondiste lo que preguntó y el "
        "   cliente no tiene más dudas.\n\n"
        + OUTPUT_FORMAT
    ),
}


# Actions where the LLM conversation makes sense. Other actions can be handled
# by the static responder.
LLM_ENABLED_ACTIONS = set(ACTION_PROMPTS.keys())


# ── Public API ───────────────────────────────────────────────────────────────

def converse(
    action: str,
    history: list[dict],
    user_message: str,
    model: str = "claude-sonnet-4-5",
) -> dict:
    """
    Run one turn of the LLM conversation.

    Parameters
    ----------
    action : str
        The current action context (e.g. "retener_cliente"). Determines the
        system prompt.
    history : list[dict]
        Previous messages in this conversation, each {"role": "user"|"assistant",
        "content": "..."}. The latest user_message is NOT in this list.
    user_message : str
        The new message from the customer.
    model : str
        Anthropic model id to use.

    Returns
    -------
    dict with keys:
        message  : assistant reply (Spanish)
        resolved : whether the conversation should end
        escalate : whether to route to a human agent
    """
    system_prompt = ACTION_PROMPTS.get(action, ACTION_PROMPTS["informar"])

    # Build the full message list for Claude
    messages = list(history) + [{"role": "user", "content": user_message}]

    client = _get_client()
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system=system_prompt,
        messages=messages,
    )

    # Claude returns content as a list of blocks; we only ask for text.
    raw_text = "".join(
        block.text for block in resp.content if getattr(block, "type", "") == "text"
    ).strip()

    # Try strict JSON parse first
    parsed = _safe_parse_json(raw_text)
    if parsed is None:
        # Model didn't follow the JSON instruction — fall back gracefully
        return {
            "message":  raw_text or "Disculpa, ¿podrías repetir eso?",
            "resolved": False,
            "escalate": False,
        }

    return {
        "message":  str(parsed.get("message", "")).strip()
                    or "Disculpa, ¿podrías repetir eso?",
        "resolved": bool(parsed.get("resolved", False)),
        "escalate": bool(parsed.get("escalate", False)),
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_parse_json(text: str) -> Optional[dict]:
    """Tolerant JSON parser. Strips code fences, finds the first {...} block."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # remove ```json ... ``` fences
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    # find first { and last } to be safe
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


# ── Standalone smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    out = converse(
        action="retener_cliente_urgente",
        history=[],
        user_message="Quiero cancelar mi cuenta ahora mismo, estoy harto del servicio",
    )
    print(out)
