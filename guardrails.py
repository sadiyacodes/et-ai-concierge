"""
ET AI Concierge — Guardrails Engine
Compliance, safety, brand rules — content filtering and disclaimer injection.
"""

import re
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

CONFIG = _load_config()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMPETITOR_BLACKLIST = [c.lower() for c in CONFIG.get("guardrails", {}).get("competitor_blacklist", [
    "Moneycontrol", "Mint", "Livemint", "Zerodha", "Groww", "Kuvera",
    "Smallcase", "Bloomberg Quint", "NDTV Profit", "Business Standard",
])]

# Map competitors to ET equivalents for redirection
COMPETITOR_REDIRECTS = {
    "moneycontrol": ("ET Markets", "https://economictimes.indiatimes.com/markets"),
    "mint": ("ET Prime", "https://economictimes.indiatimes.com/prime"),
    "livemint": ("ET Prime", "https://economictimes.indiatimes.com/prime"),
    "zerodha": ("ET Money", "https://www.etmoney.com"),
    "groww": ("ET Money", "https://www.etmoney.com"),
    "kuvera": ("ET Money", "https://www.etmoney.com"),
    "smallcase": ("ET Markets", "https://economictimes.indiatimes.com/markets"),
    "bloomberg quint": ("ET Prime", "https://economictimes.indiatimes.com/prime"),
    "ndtv profit": ("ET Markets", "https://economictimes.indiatimes.com/markets"),
    "business standard": ("ET Prime", "https://economictimes.indiatimes.com/prime"),
}

FORBIDDEN_PATTERNS = CONFIG.get("guardrails", {}).get("forbidden_patterns", [
    "guaranteed returns", "risk-free investment", "definitely will grow", "100% safe", "no risk",
])

REQUIRED_DISCLAIMERS = CONFIG.get("guardrails", {}).get("required_disclaimers", {
    "mutual_funds": "Mutual fund investments are subject to market risks. Please read all scheme-related documents carefully.",
    "stocks": "This is not investment advice. Please consult a SEBI-registered advisor.",
})

# Keywords that trigger disclaimers
MF_KEYWORDS = ["mutual fund", "sip", "elss", "nfo", "fund", "amc", "nav", "aum", "et money"]
STOCK_KEYWORDS = ["stock", "share", "nifty", "sensex", "buy", "sell", "portfolio", "smallcap", "midcap", "largecap"]

# Sensitive data patterns
SENSITIVE_PATTERNS = [
    (r"\b[A-Z]{5}\d{4}[A-Z]\b", "PAN number"),
    (r"\b\d{4}\s?\d{4}\s?\d{4}\b", "Aadhaar number"),
    (r"\b\d{9,18}\b", "bank account number"),
    (r"\b\d{6}\b", "OTP"),
]


class GuardrailsEngine:
    """Content safety, compliance, and brand rule enforcement."""

    def __init__(self):
        self._disclaimer_added_this_session = {"mutual_funds": False, "stocks": False}

    def filter_response(self, response: str) -> tuple[str, list[str]]:
        """
        Apply all guardrail checks and return cleaned response + list of violations.
        """
        violations = []
        cleaned = response

        # 1. Check competitor mentions in response
        competitor_violations = self.check_competitor_mention(cleaned)
        if competitor_violations:
            violations.extend([f"Competitor mentioned: {c}" for c in competitor_violations])
            # Remove competitor names from response
            for comp in competitor_violations:
                pattern = re.compile(re.escape(comp), re.IGNORECASE)
                cleaned = pattern.sub("[a competing platform]", cleaned)

        # 2. Check forbidden patterns
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.lower() in cleaned.lower():
                violations.append(f"Forbidden pattern: '{pattern}'")
                # Soften the language
                cleaned = re.sub(
                    re.escape(pattern),
                    "potential for growth (subject to market conditions)",
                    cleaned,
                    flags=re.IGNORECASE,
                )

        # 3. Check for specific stock buy/sell advice
        stock_advice = self._check_stock_advice(cleaned)
        if stock_advice:
            violations.extend(stock_advice)

        # 4. Validate financial claims
        claim_violations = self.validate_financial_claims(cleaned)
        violations.extend(claim_violations)

        # 5. Check tone guardrails
        tone_issues = self._check_tone(cleaned)
        violations.extend(tone_issues)

        return cleaned, violations

    def check_competitor_mention(self, text: str) -> list[str]:
        """Return list of mentioned competitors."""
        mentioned = []
        text_lower = text.lower()
        for comp in COMPETITOR_BLACKLIST:
            if comp in text_lower:
                mentioned.append(comp)
        return mentioned

    def get_competitor_redirect(self, competitor: str) -> tuple[str, str] | None:
        """Get the ET equivalent for a competitor."""
        return COMPETITOR_REDIRECTS.get(competitor.lower())

    def add_required_disclaimers(self, response: str, context: dict) -> str:
        """Add disclaimers contextually — only once per session, non-intrusively."""
        response_lower = response.lower()
        disclaimers_to_add = []

        # Check if mutual fund disclaimer needed
        if any(kw in response_lower for kw in MF_KEYWORDS):
            if not self._disclaimer_added_this_session.get("mutual_funds"):
                disclaimers_to_add.append(REQUIRED_DISCLAIMERS.get("mutual_funds", ""))
                self._disclaimer_added_this_session["mutual_funds"] = True

        # Check if stock disclaimer needed
        if any(kw in response_lower for kw in STOCK_KEYWORDS):
            if not self._disclaimer_added_this_session.get("stocks"):
                # Only add if response contains recommendation-like language
                rec_signals = ["recommend", "suggest", "should consider", "worth looking", "you could"]
                if any(sig in response_lower for sig in rec_signals):
                    disclaimers_to_add.append(REQUIRED_DISCLAIMERS.get("stocks", ""))
                    self._disclaimer_added_this_session["stocks"] = True

        if disclaimers_to_add:
            disclaimer_text = "\n\n_" + " ".join(disclaimers_to_add) + "_"
            response += disclaimer_text

        return response

    def validate_financial_claims(self, response: str) -> list[str]:
        """Flag unsubstantiated financial claims."""
        violations = []
        response_lower = response.lower()

        # Check for specific return guarantees
        return_patterns = [
            r"\d+%\s*(?:return|grow|gain|profit)\s*(?:guaranteed|assured|certain)",
            r"(?:guaranteed|assured|certain)\s*\d+%",
            r"(?:will|shall)\s+(?:definitely|certainly|surely)\s+(?:grow|increase|double)",
            r"(?:double|triple)\s+(?:your|the)\s+money\s+in\s+\d+",
        ]
        for pat in return_patterns:
            if re.search(pat, response_lower):
                violations.append(f"Unsubstantiated claim: matches pattern '{pat}'")

        return violations

    def check_sensitive_data_in_input(self, user_input: str) -> list[str]:
        """Check if user has shared sensitive data that should not be stored."""
        found = []
        for pattern, data_type in SENSITIVE_PATTERNS:
            if re.search(pattern, user_input):
                found.append(data_type)
        return found

    def get_sensitive_data_response(self, data_types: list[str]) -> str:
        """Generate a polite response when user shares sensitive data."""
        types_str = ", ".join(data_types)
        return (
            f"I noticed you may have shared sensitive information ({types_str}). "
            "For your security, I don't store or process personal identification numbers. "
            "Please avoid sharing PAN, Aadhaar, bank account numbers, or OTPs in our conversation. "
            "Your security is important to us."
        )

    def reset_session(self):
        """Reset per-session state."""
        self._disclaimer_added_this_session = {"mutual_funds": False, "stocks": False}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _check_stock_advice(self, text: str) -> list[str]:
        """Check for specific stock buy/sell advice."""
        violations = []
        text_lower = text.lower()

        # Patterns that indicate specific stock advice
        advice_patterns = [
            r"\b(?:buy|sell|short)\s+(?:shares?\s+of\s+)?[A-Z]{2,}(?:\s+stock)?(?:\s+now|\s+immediately|\s+today)",
            r"(?:you should|i recommend|i suggest)\s+(?:buying|selling)\s+",
        ]
        for pat in advice_patterns:
            if re.search(pat, text, re.IGNORECASE):
                violations.append("Specific stock buy/sell advice detected")
                break

        return violations

    def _check_tone(self, text: str) -> list[str]:
        """Check for inappropriate tone."""
        violations = []

        # FOMO language
        fomo_patterns = [
            r"(?:don'?t miss|act now|limited time|hurry|rush|last chance|once in a lifetime)",
            r"(?:everyone is|everybody is)\s+(?:buying|investing|doing)",
        ]
        for pat in fomo_patterns:
            if re.search(pat, text, re.IGNORECASE):
                violations.append("FOMO/urgency language detected")
                break

        # Excessive punctuation
        if text.count("!") > 2:
            violations.append("Excessive exclamation marks")

        # ALL CAPS words (more than 2 consecutive caps words)
        caps_words = re.findall(r"\b[A-Z]{4,}\b", text)
        # Filter out known abbreviations
        known_abbrevs = {"NIFTY", "SENSEX", "SIP", "ELSS", "NPS", "EMI", "SEBI", "HDFC", "ICICI", "SBI", "RERA", "JSON", "HTTP", "POST", "GET"}
        unexpected_caps = [w for w in caps_words if w not in known_abbrevs]
        if len(unexpected_caps) > 1:
            violations.append("Excessive CAPS usage")

        return violations


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = GuardrailsEngine()

    # Test competitor check
    print("=== Competitor Check ===")
    text = "You could also check Zerodha or Groww for trading"
    cleaned, violations = engine.filter_response(text)
    print(f"  Original: {text}")
    print(f"  Cleaned: {cleaned}")
    print(f"  Violations: {violations}")

    # Test forbidden patterns
    print("\n=== Forbidden Patterns ===")
    text = "This fund offers guaranteed returns of 15%"
    cleaned, violations = engine.filter_response(text)
    print(f"  Original: {text}")
    print(f"  Cleaned: {cleaned}")
    print(f"  Violations: {violations}")

    # Test disclaimers
    print("\n=== Disclaimer Injection ===")
    text = "I'd recommend looking at ET Money for starting your first SIP in mutual funds."
    result = engine.add_required_disclaimers(text, {})
    print(f"  Result: {result}")

    # Test sensitive data
    print("\n=== Sensitive Data Check ===")
    sensitive = engine.check_sensitive_data_in_input("My PAN is ABCDE1234F and my Aadhaar is 1234 5678 9012")
    print(f"  Detected: {sensitive}")
    if sensitive:
        print(f"  Response: {engine.get_sensitive_data_response(sensitive)}")

    # Test tone
    print("\n=== Tone Check ===")
    text = "DON'T MISS this AMAZING OFFER! Act NOW!!! Everyone is investing!!!"
    cleaned, violations = engine.filter_response(text)
    print(f"  Violations: {violations}")

    print("\nGuardrails OK")
