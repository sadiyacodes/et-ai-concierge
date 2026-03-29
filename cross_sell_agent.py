"""
ET AI Concierge — Cross-Sell Agent & Signal Detection
Detects life-event signals and injects contextual, non-pushy product offers.
"""

import re
import asyncio
from datetime import datetime, timezone
from typing import Optional

from tools import get_llm

# ---------------------------------------------------------------------------
# Life-Event Signal Taxonomy
# ---------------------------------------------------------------------------
LIFE_EVENT_SIGNALS = {
    "home_purchase_intent": {
        "trigger_keywords": ["home loan", "housing loan", "property", "flat", "apartment", "EMI", "down payment", "real estate", "home loan interest", "property price"],
        "triggered_products": ["et_home_loan_partner", "et_prime_realty_content", "et_wealth"],
        "tone": "helpful_advisor",
        "delay_turns": 1,
        "message_template": "Since you're exploring the property market, {personalised_insight}",
    },
    "first_investment_intent": {
        "trigger_keywords": ["start investing", "how to invest", "SIP", "mutual fund basics", "where to invest", "savings account not enough", "first time invest", "begin investing"],
        "triggered_products": ["et_money", "et_sip_calculator", "et_markets_beginner_guide", "et_masterclass_investing"],
        "tone": "encouraging_mentor",
        "delay_turns": 0,
        "message_template": "It's great that you're thinking about investing. {personalised_insight}",
    },
    "job_change_intent": {
        "trigger_keywords": ["new job", "switching company", "leaving job", "notice period", "joining bonus", "salary hike", "ESOP", "offer letter", "two weeks notice"],
        "triggered_products": ["et_prime", "et_wealth", "et_money"],
        "tone": "peer_advisor",
        "delay_turns": 1,
        "message_template": "A career move is a great time to rethink your finances. {personalised_insight}",
    },
    "market_volatility_anxiety": {
        "trigger_keywords": ["market crash", "portfolio down", "should I sell", "panic", "worried about investment", "nifty falling", "sensex crash", "market dip", "red portfolio"],
        "triggered_products": ["et_prime", "et_markets_app", "et_masterclass_investing"],
        "tone": "calm_expert",
        "delay_turns": 0,
        "message_template": "Market volatility can feel unsettling, but let's look at the bigger picture. {personalised_insight}",
    },
    "retirement_planning": {
        "trigger_keywords": ["retire", "retirement corpus", "pension", "NPS", "superannuation", "financial freedom", "retire early", "FIRE"],
        "triggered_products": ["et_prime", "et_money", "et_sip_calculator"],
        "tone": "trusted_advisor",
        "delay_turns": 0,
        "message_template": "Planning for retirement is one of the smartest financial moves you can make. {personalised_insight}",
    },
    "insurance_gap": {
        "trigger_keywords": ["insurance", "term plan", "health insurance", "cover", "dependent", "family protection", "life insurance", "medical cover"],
        "triggered_products": ["et_term_insurance_partner", "et_health_insurance_partner", "et_wealth"],
        "tone": "helpful_advisor",
        "delay_turns": 1,
        "message_template": "Making sure your family is protected is important. {personalised_insight}",
    },
    "tax_saving_intent": {
        "trigger_keywords": ["tax saving", "80C", "ELSS", "tax benefit", "tax deduction", "tax planning", "save tax"],
        "triggered_products": ["et_money", "et_wealth", "et_sip_calculator"],
        "tone": "practical_helper",
        "delay_turns": 0,
        "message_template": "Tax-saving season is always a good time to optimise. {personalised_insight}",
    },
    "child_education_planning": {
        "trigger_keywords": ["child education", "kid school", "college fund", "education plan", "child future", "school fees"],
        "triggered_products": ["et_money", "et_wealth", "et_sip_calculator"],
        "tone": "empathetic_advisor",
        "delay_turns": 0,
        "message_template": "Planning for your child's education is a wonderful priority. {personalised_insight}",
    },
}


class DetectedSignal:
    """A detected life-event signal."""
    def __init__(self, signal_type: str, confidence: float, source_text: str, turn_number: int):
        self.signal_type = signal_type
        self.confidence = confidence
        self.source_text = source_text
        self.turn_number = turn_number
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "signal_type": self.signal_type,
            "confidence": self.confidence,
            "source_text": self.source_text[:100],
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
        }


class CrossSellAgent:
    """Detects life-event signals and generates contextual cross-sell offers."""

    def __init__(self):
        self.llm = get_llm()
        self._fired_signals: set[str] = set()  # Signals already acted on this session

    def detect_signals(self, conversation_history: list, current_message: str, current_turn: int = 0) -> list[DetectedSignal]:
        """
        Scan current message and recent turns for life-event signals.
        Uses keyword matching for speed, deduplicated per session.
        """
        detected = []

        # Build text corpus: current message + last 3 turns
        texts_to_scan = [current_message]
        for turn in conversation_history[-3:]:
            if turn.get("role") == "user":
                texts_to_scan.append(turn.get("content", ""))

        combined_text = " ".join(texts_to_scan).lower()

        for signal_type, config in LIFE_EVENT_SIGNALS.items():
            # Skip already-fired signals
            if signal_type in self._fired_signals:
                continue

            keywords = config["trigger_keywords"]
            hits = 0
            matched_keyword = ""
            for kw in keywords:
                if kw.lower() in combined_text:
                    hits += 1
                    if not matched_keyword:
                        matched_keyword = kw

            if hits == 0:
                continue

            # Calculate confidence based on number of keyword hits
            confidence = min(0.5 + (hits * 0.15), 0.95)

            # Higher confidence if keyword is in current message (not just history)
            if any(kw.lower() in current_message.lower() for kw in keywords):
                confidence = min(confidence + 0.1, 0.95)

            detected.append(DetectedSignal(
                signal_type=signal_type,
                confidence=confidence,
                source_text=matched_keyword,
                turn_number=current_turn,
            ))

        # Sort by confidence descending
        detected.sort(key=lambda s: -s.confidence)
        return detected

    async def generate_cross_sell_message(
        self,
        signal: DetectedSignal,
        profile: dict,
        context: str = "",
    ) -> dict | None:
        """
        Generate a contextual, non-pushy cross-sell message.
        Returns None if confidence too low or already recommended.
        """
        if signal.confidence < 0.65:
            return None

        if signal.signal_type in self._fired_signals:
            return None

        signal_config = LIFE_EVENT_SIGNALS.get(signal.signal_type)
        if not signal_config:
            return None

        # Generate personalised insight using LLM
        persona = profile.get("persona_tag", "first_time_investor")
        experience = profile.get("investment_experience", "beginner")
        products = signal_config["triggered_products"]

        prompt = f"""Generate a brief, helpful cross-sell suggestion for the ET AI Concierge.

Signal detected: {signal.signal_type}
User persona: {persona}
Investment experience: {experience}
Triggered products: {', '.join(products)}
Tone: {signal_config['tone']}
Message template start: {signal_config['message_template']}

Conversation context: {context}

Rules:
- Maximum 2 sentences total
- Must feel helpful, not salesy
- No exclamation marks, no CAPS, no "AMAZING OFFER" language
- Reference one specific ET product that addresses their need
- Include what the product actually does for them
- Sound like a knowledgeable friend, not a salesperson

Write ONLY the personalised_insight part (to complete the template). Keep it under 30 words."""

        result = await self.llm.generate(prompt, "You are a subtle, helpful financial guide.")
        insight_text = result.get("text", "").strip()

        # Strip quotes if LLM wrapped in them
        insight_text = insight_text.strip('"\'')

        # Build full message from template
        full_message = signal_config["message_template"].replace("{personalised_insight}", insight_text)

        # Mark as fired
        self._fired_signals.add(signal.signal_type)

        # Pick primary product
        primary_product = products[0] if products else ""

        return {
            "signal_type": signal.signal_type,
            "message": full_message,
            "products": products,
            "primary_product": primary_product,
            "tone": signal_config["tone"],
            "confidence": signal.confidence,
        }

    def should_inject_now(
        self,
        signal: DetectedSignal,
        turn_number: int,
        session_state: dict,
    ) -> bool:
        """
        Decide if now is the right time to inject a cross-sell message.
        Respects delay, rate limits, and engagement signals.
        """
        signal_config = LIFE_EVENT_SIGNALS.get(signal.signal_type, {})
        delay = signal_config.get("delay_turns", 0)

        # Respect delay_turns
        if delay > 0 and turn_number <= signal.turn_number + delay:
            return False

        # Rate limit: max 1 cross-sell per 3 turns
        last_injection_turn = session_state.get("last_cross_sell_turn", -10)
        if turn_number - last_injection_turn < 3:
            return False

        # Session limit
        injection_count = session_state.get("cross_sell_injections_count", 0)
        if injection_count >= 3:
            return False

        # Disengagement detection: if last 2 user messages are very short, hold off
        turns = session_state.get("raw_turns", [])
        user_turns = [t for t in turns[-4:] if t.get("role") == "user"]
        if len(user_turns) >= 2:
            short_replies = sum(1 for t in user_turns[-2:] if len(t.get("content", "")) < 15)
            if short_replies >= 2:
                return False

        return True

    async def handle_home_loan_pivot(self, profile: dict, context: str = "") -> str:
        """
        Dedicated handler for Scenario #3: Home loan cross-sell from markets context.
        Acknowledges their trader identity, bridges naturally to home loan tools.
        """
        persona = profile.get("persona_tag", "seasoned_trader")
        is_trader = persona in ("seasoned_trader", "wealth_builder")

        if is_trader:
            message = (
                "I can see you're a regular markets follower — so you already understand "
                "how interest rate cycles affect asset classes. Since you're comparing home loan rates, "
                "ET has a comparison tool that pulls live rates from SBI, HDFC, ICICI, and 12 other lenders. "
                "You can also check out ET Prime's real estate analysis for city-wise price trends "
                "and the best time to lock in a rate. Want me to pull up the rate comparison?"
            )
        else:
            message = (
                "Since you're looking at home loan rates, ET has a handy tool that compares "
                "rates across major banks side-by-side. It also shows you your estimated EMI "
                "for different loan amounts. ET Wealth also has some excellent guides on the "
                "home buying process. Would you like to see the rate comparison tool?"
            )

        return message

    def reset(self):
        """Reset for a new session."""
        self._fired_signals.clear()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _test():
        agent = CrossSellAgent()

        # Test signal detection
        print("=== Signal Detection ===")
        history = [
            {"role": "user", "content": "What's happening with Nifty today?"},
            {"role": "assistant", "content": "Nifty is currently at 24,500..."},
        ]
        signals = agent.detect_signals(history, "What are the current home loan interest rates?", current_turn=2)
        for s in signals:
            print(f"  Signal: {s.signal_type} (confidence: {s.confidence})")

        # Test should_inject
        if signals:
            session_state = {"last_cross_sell_turn": -10, "cross_sell_injections_count": 0, "raw_turns": history}
            should = agent.should_inject_now(signals[0], 2, session_state)
            print(f"  Should inject now: {should}")

            # With delay (turn 2, detected at turn 2, delay=1)
            # Should be False for home_purchase_intent due to delay_turns=1
            print(f"  (delay_turns=1, so should wait)")

            # Test at turn 3
            should = agent.should_inject_now(signals[0], 3, session_state)
            print(f"  At turn 3: should inject = {should}")

        # Test message generation
        print("\n=== Cross-Sell Message ===")
        profile = {"persona_tag": "seasoned_trader", "investment_experience": "advanced"}
        if signals:
            msg = await agent.generate_cross_sell_message(signals[0], profile, "User is a daily markets reader")
            if msg:
                print(f"  Message: {msg['message']}")
                print(f"  Products: {msg['products']}")

        # Test home loan pivot
        print("\n=== Home Loan Pivot (Scenario 3) ===")
        pivot_msg = await agent.handle_home_loan_pivot(profile)
        print(f"  {pivot_msg}")

        # Test beginner detection
        print("\n=== Beginner Signal Detection ===")
        agent.reset()
        signals = agent.detect_signals([], "I want to start investing but don't know where to begin", current_turn=0)
        for s in signals:
            print(f"  Signal: {s.signal_type} (confidence: {s.confidence})")

    asyncio.run(_test())
