"""
build_dataset.py
Reads the Hey Banco raw conversation CSV and auto-labels each customer message
with:  intent, sentiment, and derived action.

Run from the ml/ folder:
    python build_dataset.py
Outputs:
    labeled_data.json   – used by train.py
"""

import csv
import json
import re
import os

# ─────────────────────────── label rules ────────────────────────────────────

INTENT_RULES = [
    # (intent_label, [keyword/phrase list])   – order matters: first match wins
    ("cancelar_cuenta",           ["darme de baja", "cancelar mi cuenta", "cerrar cuenta",
                                   "cerrar mi cuenta", "quiero cancelar cuenta", "baja de cuenta",
                                   "quiero dar de baja"]),
    ("cancelar_cargo",            ["cancelar cargo", "cargo recurrente", "domiciliado",
                                   "cancelar cobro", "quitar cobro", "cargo no autorizado",
                                   "suspender cargo", "cargo domiciliado"]),
    ("bloqueo_tarjeta",           ["bloquear tarjeta", "robo de tarjeta", "robo tarjeta",
                                   "reportar robo", "extravío", "extraviada", "bloqueada",
                                   "tarjeta robada", "reporte de robo"]),
    ("fraude_cargo_no_reconocido",["no reconozco", "cargo extraño", "fraude", "cobro incorrecto",
                                   "cobro que no hice", "no lo hice", "no autoricé",
                                   "no realicé", "cargo que no es mío"]),
    ("transferencia",             ["transferencia", "enviar dinero", "transferir", "spei",
                                   "depositar", "mandar dinero", "pagar a", "envío de"]),
    ("consulta_saldo",            ["saldo", "balance", "cuánto tengo", "cuanto tengo",
                                   "estado de cuenta", "movimientos", "mis compras",
                                   "mis transacciones", "historial"]),
    ("solicitar_producto",        ["solicitar tarjeta", "solicitar crédito", "quiero una tarjeta",
                                   "abrir cuenta", "préstamo", "prestamo", "crédito personal",
                                   "credito personal", "solicitud de", "abrir una cuenta",
                                   "quiero solicitar"]),
    ("cambio_datos_perfil",       ["cambiar nip", "cambiar pin", "cambiar número", "cambiar correo",
                                   "actualizar datos", "cambiar teléfono", "cambiar contraseña",
                                   "actualizar perfil", "actualizar información"]),
    ("problema_tecnico",          ["no funciona", "error", "no me deja", "no puedo",
                                   "falla", "no carga", "no abre", "no jala",
                                   "no sirve", "pantalla", "app no", "aplicación no"]),
    ("queja_servicio",            ["mal servicio", "muy mal", "pésimo", "pesimo",
                                   "terrible", "inconforme", "queja", "decepcionado",
                                   "decepciona", "no me ayudan", "no me resuelven",
                                   "desastre", "inaceptable", "molesto", "molesta",
                                   "enojado", "enojada"]),
    ("hablar_asesor",             ["hablar con asesor", "hablar con agente", "hablar con ejecutivo",
                                   "quiero hablar", "necesito hablar", "número de atención",
                                   "llamar a", "centro de atención", "asesor humano",
                                   "hablar con alguien", "comunicarme con"]),
    ("informacion_general",       []),   # catch-all
]

SENTIMENT_RULES = [
    # (sentiment_label, [keyword list])
    ("negativo_urgente",  ["urgente", "urgentemente", "inmediatamente", "ahora mismo",
                           "ya no puedo más", "harto", "harta", "cansado", "no puedo más",
                           "exijo", "demando", "amenaza", "denunciar", "denuncia",
                           "proceder legalmente", "urgente por favor"]),
    ("negativo_queja",    ["mal servicio", "muy mal", "pésimo", "pesimo", "terrible",
                           "inconforme", "decepcionado", "decepcionada", "frustra",
                           "frustrado", "frustrante", "molesto", "molesta",
                           "enojado", "enojada", "no sirve", "no funciona bien",
                           "no me ayudan", "desastre", "inaceptable", "no es posible",
                           "que mal", "qué mal", "no entiendo por qué"]),
    ("positivo",          ["gracias", "excelente", "perfecto", "muy bien", "genial",
                           "encanta", "feliz", "contento", "contenta", "satisfecho",
                           "satisfecha", "buen servicio", "muy amable", "me funcionó",
                           "ya quedó", "resuelto", "solucionado"]),
    ("neutral",           []),   # catch-all
]

# Derived action: (intent, sentiment) → action
ACTION_MAP = {
    # Retention priority
    ("cancelar_cuenta",            "negativo_urgente"): "retener_cliente_urgente",
    ("cancelar_cuenta",            "negativo_queja"):   "retener_cliente",
    ("cancelar_cuenta",            "neutral"):          "retener_cliente",
    ("cancelar_cuenta",            "positivo"):         "retener_cliente",
    # Complaints → escalate if urgent, else resolve
    ("queja_servicio",             "negativo_urgente"): "escalar_agente",
    ("queja_servicio",             "negativo_queja"):   "escalar_agente",
    ("queja_servicio",             "neutral"):          "resolver_problema",
    # Fraud → always escalate
    ("fraude_cargo_no_reconocido", "negativo_urgente"): "escalar_agente",
    ("fraude_cargo_no_reconocido", "negativo_queja"):   "escalar_agente",
    ("fraude_cargo_no_reconocido", "neutral"):          "escalar_agente",
    ("fraude_cargo_no_reconocido", "positivo"):         "escalar_agente",
    # Card blocking → urgent resolution
    ("bloqueo_tarjeta",            "negativo_urgente"): "escalar_agente",
    ("bloqueo_tarjeta",            "negativo_queja"):   "resolver_problema",
    ("bloqueo_tarjeta",            "neutral"):          "resolver_problema",
    ("bloqueo_tarjeta",            "positivo"):         "resolver_problema",
    # Charge cancellation
    ("cancelar_cargo",             "negativo_urgente"): "escalar_agente",
    ("cancelar_cargo",             "negativo_queja"):   "resolver_problema",
    ("cancelar_cargo",             "neutral"):          "procesar_solicitud",
    ("cancelar_cargo",             "positivo"):         "procesar_solicitud",
    # Transfer
    ("transferencia",              "negativo_urgente"): "escalar_agente",
    ("transferencia",              "negativo_queja"):   "resolver_problema",
    ("transferencia",              "neutral"):          "procesar_solicitud",
    ("transferencia",              "positivo"):         "procesar_solicitud",
    # Balance query
    ("consulta_saldo",             "negativo_urgente"): "resolver_problema",
    ("consulta_saldo",             "negativo_queja"):   "resolver_problema",
    ("consulta_saldo",             "neutral"):          "informar",
    ("consulta_saldo",             "positivo"):         "informar",
    # Product requests
    ("solicitar_producto",         "negativo_urgente"): "informar",
    ("solicitar_producto",         "negativo_queja"):   "informar",
    ("solicitar_producto",         "neutral"):          "oferta_producto",
    ("solicitar_producto",         "positivo"):         "oferta_producto",
    # Profile/data changes
    ("cambio_datos_perfil",        "negativo_urgente"): "escalar_agente",
    ("cambio_datos_perfil",        "negativo_queja"):   "resolver_problema",
    ("cambio_datos_perfil",        "neutral"):          "procesar_solicitud",
    ("cambio_datos_perfil",        "positivo"):         "procesar_solicitud",
    # Technical issues
    ("problema_tecnico",           "negativo_urgente"): "escalar_agente",
    ("problema_tecnico",           "negativo_queja"):   "resolver_problema",
    ("problema_tecnico",           "neutral"):          "resolver_problema",
    ("problema_tecnico",           "positivo"):         "resolver_problema",
    # Speak with agent
    ("hablar_asesor",              "negativo_urgente"): "escalar_agente",
    ("hablar_asesor",              "negativo_queja"):   "escalar_agente",
    ("hablar_asesor",              "neutral"):          "escalar_agente",
    ("hablar_asesor",              "positivo"):         "escalar_agente",
    # General info
    ("informacion_general",        "negativo_urgente"): "escalar_agente",
    ("informacion_general",        "negativo_queja"):   "resolver_problema",
    ("informacion_general",        "neutral"):          "informar",
    ("informacion_general",        "positivo"):         "informar",
}


def classify_intent(text: str) -> str:
    t = text.lower()
    for intent, keywords in INTENT_RULES:
        if keywords and any(kw in t for kw in keywords):
            return intent
    return "informacion_general"


def classify_sentiment(text: str) -> str:
    t = text.lower()
    # Boost negative signal: excessive caps or repeated punctuation
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclamation_count = text.count("!") + text.count("?!")
    if caps_ratio > 0.4 or exclamation_count >= 3:
        return "negativo_urgente"
    for sentiment, keywords in SENTIMENT_RULES:
        if keywords and any(kw in t for kw in keywords):
            return sentiment
    return "neutral"


def derive_action(intent: str, sentiment: str) -> str:
    return ACTION_MAP.get((intent, sentiment), "informar")


# ─────────────────────────── main ───────────────────────────────────────────

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    csv_path = os.path.join(data_dir, "dataset_50k_anonymized.csv")

    labeled = []
    intent_counts: dict[str, int] = {}
    sentiment_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}

    print(f"Reading {csv_path} ...")
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["input"].strip()
            if not text or len(text) < 3:
                continue

            intent    = classify_intent(text)
            sentiment = classify_sentiment(text)
            action    = derive_action(intent, sentiment)

            labeled.append({
                "text":      text,
                "intent":    intent,
                "sentiment": sentiment,
                "action":    action,
            })
            intent_counts[intent]       = intent_counts.get(intent, 0) + 1
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            action_counts[action]       = action_counts.get(action, 0) + 1

    out_path = os.path.join(os.path.dirname(__file__), "labeled_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Wrote {len(labeled)} labeled examples → {out_path}")
    print("\n── Intent distribution ──")
    for k, v in sorted(intent_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<35} {v:>6}")
    print("\n── Sentiment distribution ──")
    for k, v in sorted(sentiment_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<35} {v:>6}")
    print("\n── Action distribution ──")
    for k, v in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<35} {v:>6}")


if __name__ == "__main__":
    main()
