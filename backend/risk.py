from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Transaction:
    amount: float
    category: str
    failed_attempts: int
    is_anomaly: bool
    country: Optional[str] = None
    user_avg_amount: float = 0.0


@dataclass
class RiskResult:
    level: str          # "low", "medium", "high"
    score: int          # 0–100
    flags: list = field(default_factory=list)   # Human-readable reasons


def assess_risk(tx: Transaction, user_profile: dict) -> RiskResult:
    """
    Rule-based risk scoring.
    Score thresholds: 0–30 = low, 31–60 = medium, 61+ = high
    """
    score = 0
    flags = []

    # Rule 1: Large transaction relative to user average
    avg = user_profile.get("avg_transaction_amount", 100.0)
    if avg > 0:
        if tx.amount > avg * 5:
            score += 35
            flags.append(
                f"Transaction amount (${tx.amount:.2f}) is 5x above your average (${avg:.2f})"
            )
        elif tx.amount > avg * 2:
            score += 15
            flags.append(
                f"Transaction amount (${tx.amount:.2f}) is above your typical spending"
            )

    # Rule 2: Absolute high-value threshold
    if tx.amount > 5000:
        score += 20
        flags.append(f"High-value transaction: ${tx.amount:.2f}")

    # Rule 3: Failed attempts before this transaction
    if tx.failed_attempts >= 3:
        score += 30
        flags.append(f"Multiple failed attempts detected ({tx.failed_attempts})")
    elif tx.failed_attempts >= 1:
        score += 10
        flags.append(f"Failed attempt recorded before this transaction")

    # Rule 4: System anomaly flag
    if tx.is_anomaly:
        score += 25
        flags.append("Transaction flagged as anomalous by system")

    # Rule 5: Unusual merchant category for this user
    user_categories = user_profile.get("common_categories", [])
    if user_categories and tx.category not in user_categories:
        score += 10
        flags.append(f"Unusual spending category for your profile: {tx.category}")

    # Rule 6: Foreign transaction for a domestic-only user
    user_country = user_profile.get("country", "MX")
    if tx.country and tx.country != user_country:
        score += 15
        flags.append(f"Transaction from foreign country: {tx.country}")

    # Cap score at 100
    score = min(score, 100)

    # Determine risk level
    if score >= 61:
        level = "high"
    elif score >= 31:
        level = "medium"
    else:
        level = "low"

    return RiskResult(level=level, score=score, flags=flags)


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Risk Engine Test ===\n")

    scenarios = [
        {
            "label": "Normal everyday transaction",
            "tx": Transaction(amount=45.0, category="groceries",
                              failed_attempts=0, is_anomaly=False),
            "profile": {"avg_transaction_amount": 80.0,
                        "common_categories": ["groceries", "transport"],
                        "country": "MX"}
        },
        {
            "label": "Slightly above average, one failed attempt",
            "tx": Transaction(amount=350.0, category="dining",
                              failed_attempts=1, is_anomaly=False),
            "profile": {"avg_transaction_amount": 120.0,
                        "common_categories": ["groceries", "utilities"],
                        "country": "MX"}
        },
        {
            "label": "High-value foreign fraud scenario",
            "tx": Transaction(amount=8500.0, category="electronics",
                              failed_attempts=3, is_anomaly=True, country="US"),
            "profile": {"avg_transaction_amount": 120.0,
                        "common_categories": ["groceries", "transport", "utilities"],
                        "country": "MX"}
        },
    ]

    for s in scenarios:
        result = assess_risk(s["tx"], s["profile"])
        print(f"Scenario: {s['label']}")
        print(f"  Risk Level : {result.level.upper()}")
        print(f"  Risk Score : {result.score}/100")
        for f in result.flags:
            print(f"  ⚠️  {f}")
        print()
