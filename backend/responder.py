from typing import Optional

# Response matrix keyed on (intent, risk_level)
RESPONSES: dict[tuple[str, str], str] = {
    # ── check_balance ─────────────────────────────────────────────────────────
    ("check_balance", "low"):
        "Your current balance is $4,823.50. Everything looks normal — no unusual activity detected.",
    ("check_balance", "medium"):
        "Your balance is $4,823.50. We noticed some unusual activity on your account. "
        "Please review your recent transactions.",
    ("check_balance", "high"):
        "Your balance is $4,823.50. ⚠️ We've detected high-risk activity on your account. "
        "Please contact support immediately or visit your nearest branch.",

    # ── transfer_money ────────────────────────────────────────────────────────
    ("transfer_money", "low"):
        "Your transfer has been initiated successfully. It should arrive within 1–2 business days.",
    ("transfer_money", "medium"):
        "We're processing your transfer, but it has been flagged for a quick review. "
        "You'll receive a confirmation within 30 minutes.",
    ("transfer_money", "high"):
        "⚠️ Your transfer has been temporarily held due to unusual activity. "
        "Please verify your identity to proceed.",

    # ── fraud_alert ───────────────────────────────────────────────────────────
    ("fraud_alert", "low"):
        "We reviewed the transaction you mentioned and it appears normal. "
        "If you still have concerns, please don't hesitate to contact us.",
    ("fraud_alert", "medium"):
        "We're investigating the flagged transaction on your behalf. "
        "You'll receive an update within 24 hours. No action needed from you right now.",
    ("fraud_alert", "high"):
        "🚨 We've detected potentially fraudulent activity on your account. "
        "Your card has been temporarily blocked as a precaution. "
        "Please call 1-800-SECURE now to verify your identity.",

    # ── product_inquiry ───────────────────────────────────────────────────────
    ("product_inquiry", "low"):
        "We offer a range of products: savings accounts, credit cards with cashback, "
        "and investment portfolios. Would you like details on any specific product?",
    ("product_inquiry", "medium"):
        "We have excellent financial products available! Based on your profile, "
        "our Premium Savings Account may be a great fit. "
        "Note: there is a pending review on your account — we'll resolve it shortly.",
    ("product_inquiry", "high"):
        "We'd love to help you explore our products. However, please resolve the "
        "active security alert on your account first. "
        "Call 1-800-SECURE and we'll get you set up right after.",

    # ── customer_support ──────────────────────────────────────────────────────
    ("customer_support", "low"):
        "I'm here to help! What do you need assistance with today? "
        "You can also reach our support team 24/7 at 1-800-FINHELP.",
    ("customer_support", "medium"):
        "I'll connect you with a support agent right away. "
        "Please note: there's a pending review on your account that the agent will also address.",
    ("customer_support", "high"):
        "A security specialist will contact you within the next 10 minutes "
        "regarding the urgent alert on your account. Please stay available.",

    # ── transaction_history ───────────────────────────────────────────────────
    ("transaction_history", "low"):
        "Here are your last 5 transactions:\n"
        "  • Amazon       — $45.00\n"
        "  • Starbucks    — $6.50\n"
        "  • Uber         — $12.30\n"
        "  • Netflix      — $15.99\n"
        "  • Walmart      — $89.40",
    ("transaction_history", "medium"):
        "Here are your recent transactions. One item is under review:\n"
        "  • Amazon       — $45.00\n"
        "  • Starbucks    — $6.50\n"
        "  • [Under Review: Unknown Merchant — $1,200.00] ⚠️\n"
        "  • Netflix      — $15.99\n"
        "  • Walmart      — $89.40",
    ("transaction_history", "high"):
        "⚠️ Your transaction history shows suspicious activity. "
        "Full account access has been temporarily limited. "
        "Please call 1-800-SECURE to verify your identity and restore access.",
}

DEFAULT_RESPONSE = (
    "I'm sorry, I didn't fully understand that. "
    "Could you rephrase your question? "
    "You can also call us at 1-800-FINHELP for immediate assistance."
)


def generate_response(
    intent: str,
    risk_level: str,
    flags: Optional[list] = None
) -> str:
    """
    Map (intent, risk_level) to a human-like response.
    Appends the top risk flag as a reason for medium/high alerts.
    """
    response = RESPONSES.get((intent, risk_level), DEFAULT_RESPONSE)

    # Append the top risk flag for context on medium/high alerts
    if flags and risk_level in ("medium", "high"):
        top_flag = flags[0]
        response += f"\n\n📋 Reason flagged: {top_flag}"

    return response
