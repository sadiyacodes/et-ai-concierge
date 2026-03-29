"""
ET AI Concierge — Profiler Agent
Extracts maximum user profile information in minimum turns via progressive profiling.
"""

import asyncio
from tools import get_llm

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
PROFILER_SYSTEM_PROMPT = """You are the ET Welcome Concierge's profile intelligence engine. Your goal is to understand who the user is financially and personally — without feeling like a questionnaire. You are warm, curious, and efficient.

Rules:
1. Ask at most ONE question per turn. Make it feel like natural conversation, not a form.
2. Prioritise the highest-value missing field based on what recommendations it unlocks.
3. Infer as much as possible from what the user has already said before asking.
4. Never ask about exact income — infer from context (job type, city, investment mentions).
5. For beginners, use plain language. For experienced investors, use appropriate terminology.
6. If the user volunteers information without being asked, capture it silently.
7. Never repeat a question you've already asked.
8. Keep responses under 3 sentences. Be conversational, not robotic."""


# ---------------------------------------------------------------------------
# Next-question priority map
# ---------------------------------------------------------------------------
QUESTION_PRIORITY = [
    {
        "field": "investment_experience",
        "weight": 0.25,
        "questions": {
            "default": "Have you done any investing before, or would this be your first time?",
            "has_some_info": "It sounds like you have some experience — would you say you're comfortable with markets, or still finding your way?",
        },
    },
    {
        "field": "primary_financial_goal",
        "weight": 0.20,
        "questions": {
            "default": "What's the main financial goal on your mind right now — saving for something specific, or building long-term wealth?",
            "beginner": "What's your biggest money goal right now — could be anything from building a safety net to growing your savings?",
            "advanced": "What are you optimising for currently — tax efficiency, alpha generation, or a specific milestone?",
        },
    },
    {
        "field": "life_stage",
        "weight": 0.15,
        "questions": {
            "default": "Just to understand your situation better — are you early in your career, or a few years in?",
        },
    },
    {
        "field": "risk_appetite",
        "weight": 0.15,
        "questions": {
            "default": "When markets dip 10-15%, how would you feel — worried, okay with it, or see it as a buying opportunity?",
            "beginner": "If your investment dropped 10% in a month, would you want to pull out, wait it out, or invest more?",
        },
    },
    {
        "field": "et_products_used",
        "weight": 0.15,
        "questions": {
            "default": "Have you used any ET tools before — ET Markets for tracking stocks, ET Money for mutual funds, or ET Prime for analysis?",
        },
    },
    {
        "field": "income_band",
        "weight": 0.10,
        "questions": {
            "default": "Roughly, how much could you comfortably set aside for investing each month — just a ballpark is fine?",
        },
    },
]


# ---------------------------------------------------------------------------
# Profile field weights for completeness
# ---------------------------------------------------------------------------
FIELD_WEIGHTS = {
    "investment_experience": 0.25,
    "primary_financial_goal": 0.20,
    "life_stage": 0.15,
    "et_products_used": 0.15,
    "risk_appetite": 0.15,
    "income_band": 0.10,
}


class ProfilerAgent:
    """Extracts user profile in ≤3 turns with progressive profiling."""

    def __init__(self):
        self.llm = get_llm()
        self._asked_fields: set[str] = set()

    async def extract_profile_from_turn(self, turn: str, current_profile: dict) -> dict:
        """
        Extract profile signals from a user turn using LLM + rule-based inference.
        Returns: {updated_fields, confidence_scores, next_question, inferred_signals}
        """
        # Build context about what we already know
        known_fields = {k: v for k, v in current_profile.items()
                       if v and v not in ("", [], 0, None, 0.0, "none")
                       and k not in ("confidence_scores", "last_updated", "profile_completeness")}

        prompt = f"""Analyse this user message and extract any profile information.

Current known profile: {known_fields}

User message: "{turn}"

Extract these fields if mentioned or inferable:
- age_band: "18-25" | "26-35" | "36-45" | "46-55" | "55+"
- life_stage: "student" | "early_career" | "family_builder" | "wealth_accumulator" | "pre_retirement" | "retired"
- income_band: "below_5L" | "5-10L" | "10-25L" | "25-50L" | "50L+"
- city_tier: "metro" | "tier1" | "tier2" | "tier3"
- investment_experience: "none" | "beginner" | "intermediate" | "advanced"
- current_investments: list of ["savings_account", "fd", "sip", "stocks", "mf", "real_estate", "crypto"]
- primary_financial_goal: "wealth_creation" | "tax_saving" | "retirement" | "child_education" | "home_purchase" | "emergency_fund"
- risk_appetite: "conservative" | "moderate" | "aggressive"
- monthly_investable_surplus: "below_5k" | "5-15k" | "15-50k" | "50k+"
- et_products_used: list of ["et_prime", "et_markets", "et_money", "et_wealth", "et_masterclass"]
- et_subscription_status: "none" | "active" | "lapsed" | "trial"
- persona_tag: "first_time_investor" | "seasoned_trader" | "lapsed_subscriber" | "wealth_builder" | "nri"
- detected_life_events: list of ["home_loan_interest", "job_change", "marriage", "child_birth", "inheritance"]
- urgency_level: "browsing" | "researching" | "ready_to_act"

ONLY include fields that can be reasonably inferred from the message. Do not guess.

Respond with JSON:
{{
    "updated_fields": {{"field_name": "value", ...}},
    "confidence_scores": {{"field_name": 0.8, ...}},
    "inferred_signals": ["signal1", ...]
}}"""

        result = await self.llm.generate_structured(prompt, PROFILER_SYSTEM_PROMPT)
        parsed = result.get("parsed", {})

        updated_fields = parsed.get("updated_fields", {})
        confidence_scores = parsed.get("confidence_scores", {})
        inferred_signals = parsed.get("inferred_signals", [])

        # Determine next question
        next_question = self.get_next_best_question(current_profile, updated_fields)

        return {
            "updated_fields": updated_fields,
            "confidence_scores": confidence_scores,
            "next_question": next_question,
            "inferred_signals": inferred_signals,
            "model_used": result.get("model_used", "unknown"),
            "tokens_used": result.get("tokens_used", 0),
        }

    def calculate_profile_completeness(self, profile: dict) -> float:
        score = 0.0
        for field, weight in FIELD_WEIGHTS.items():
            val = profile.get(field, "")
            if isinstance(val, list):
                filled = len(val) > 0
            elif isinstance(val, str):
                filled = val not in ("", "none")
            else:
                filled = val is not None and val != 0
            if filled:
                conf = profile.get("confidence_scores", {}).get(field, 0.5)
                score += weight * min(conf / 0.7, 1.0)
        return round(min(score, 1.0), 2)

    def get_next_best_question(self, profile: dict, pending_updates: dict | None = None) -> str | None:
        """Return the single highest-ROI question to ask next."""
        merged = dict(profile)
        if pending_updates:
            merged.update(pending_updates)

        completeness = self.calculate_profile_completeness(merged)
        if completeness >= 0.8:
            return None  # Profile is sufficient

        # Find the highest-weight missing field we haven't asked
        for item in QUESTION_PRIORITY:
            field = item["field"]
            if field in self._asked_fields:
                continue

            val = merged.get(field, "")
            is_empty = val in ("", [], 0, None, 0.0, "none")
            if not is_empty:
                continue

            # Pick the right variant of the question
            exp = merged.get("investment_experience", "")
            questions = item["questions"]

            if exp in ("none", "beginner") and "beginner" in questions:
                q = questions["beginner"]
            elif exp in ("advanced",) and "advanced" in questions:
                q = questions["advanced"]
            elif val and "has_some_info" in questions:
                q = questions["has_some_info"]
            else:
                q = questions["default"]

            self._asked_fields.add(field)
            return q

        return None

    def reset(self):
        """Reset asked fields for a new session."""
        self._asked_fields.clear()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _test():
        agent = ProfilerAgent()
        profile = {
            "investment_experience": "",
            "primary_financial_goal": "",
            "life_stage": "",
            "risk_appetite": "",
            "et_products_used": [],
            "income_band": "",
            "confidence_scores": {},
        }

        turns = [
            "Hi, I'm 28 and work in IT. Never invested before.",
            "I want to start with something simple. My friends keep talking about SIPs.",
            "I can probably set aside 10-15k a month. I don't want to take too much risk though.",
        ]

        for turn in turns:
            print(f"\nUser: {turn}")
            result = await agent.extract_profile_from_turn(turn, profile)
            print(f"  Extracted: {result['updated_fields']}")
            print(f"  Confidence: {result['confidence_scores']}")
            if result["next_question"]:
                print(f"  Next Q: {result['next_question']}")

            # Merge updates
            for field, value in result["updated_fields"].items():
                if isinstance(profile.get(field), list) and isinstance(value, list):
                    profile[field] = list(set(profile[field] + value))
                else:
                    profile[field] = value
            profile["confidence_scores"].update(result.get("confidence_scores", {}))

            completeness = agent.calculate_profile_completeness(profile)
            print(f"  Completeness: {completeness}")

        print(f"\nFinal profile: {profile}")

    asyncio.run(_test())
