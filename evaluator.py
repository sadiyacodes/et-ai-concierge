"""
ET AI Concierge — Evaluator Agent
Self-evaluation & fallback recovery — the quality control loop.
"""

import asyncio
from tools import get_llm
from guardrails import GuardrailsEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_WEIGHTS = {
    "relevance": 0.30,
    "persona_fit": 0.25,
    "recommendation_quality": 0.25,
    "brand_safety": 0.15,
    "conciseness": 0.05,
}

PERSONA_MAX_SENTENCES = {
    "first_time_investor": 4,
    "beginner": 4,
    "seasoned_trader": 6,
    "wealth_builder": 5,
    "lapsed_subscriber": 5,
    "nri": 5,
}

PASS_THRESHOLD = 0.75
CRITICAL_THRESHOLD = 0.6


class EvaluatorAgent:
    """Scores response quality and triggers self-correction or fallback."""

    def __init__(self):
        self.llm = get_llm()
        self.guardrails = GuardrailsEngine()

    async def score_response(
        self,
        response: str,
        profile: dict,
        user_input: str,
        context: dict | None = None,
    ) -> dict:
        """
        Score a response draft on 5 dimensions.
        Returns: {overall_score, dimensions, passed, lowest_dimension, improvement_hint}
        """
        context = context or {}
        persona = profile.get("persona_tag", "first_time_investor")

        # 1. Brand Safety (rule-based — fast)
        _, violations = self.guardrails.filter_response(response)
        brand_safety_score = 1.0 if not violations else max(0.0, 1.0 - len(violations) * 0.25)

        # 2. Conciseness (rule-based)
        sentences = [s.strip() for s in response.split(".") if s.strip()]
        max_sentences = PERSONA_MAX_SENTENCES.get(persona, 5)
        if len(sentences) <= max_sentences:
            conciseness_score = 1.0
        elif len(sentences) <= max_sentences + 2:
            conciseness_score = 0.7
        else:
            conciseness_score = 0.4

        # 3. Relevance, Persona Fit, Recommendation Quality (LLM-based)
        llm_scores = await self._llm_evaluate(response, profile, user_input, context, persona)

        relevance_score = llm_scores.get("relevance", 0.7)
        persona_fit_score = llm_scores.get("persona_fit", 0.7)
        rec_quality_score = llm_scores.get("recommendation_quality", 0.7)

        # Calculate weighted overall
        dimensions = {
            "relevance": round(relevance_score, 2),
            "persona_fit": round(persona_fit_score, 2),
            "recommendation_quality": round(rec_quality_score, 2),
            "brand_safety": round(brand_safety_score, 2),
            "conciseness": round(conciseness_score, 2),
        }

        overall = sum(dimensions[k] * EVAL_WEIGHTS[k] for k in EVAL_WEIGHTS)
        overall = round(overall, 3)

        # Find lowest dimension
        lowest_dim = min(dimensions, key=lambda k: dimensions[k])

        passed = overall >= PASS_THRESHOLD

        return {
            "overall_score": overall,
            "dimensions": dimensions,
            "passed": passed,
            "lowest_dimension": lowest_dim,
            "lowest_score": dimensions[lowest_dim],
            "guardrail_violations": violations,
            "improvement_hint": self._get_improvement_hint(lowest_dim, dimensions[lowest_dim], persona),
        }

    def generate_improvement_instruction(self, eval_result: dict) -> str:
        """Generate a specific instruction for the response assembler to retry."""
        lowest = eval_result.get("lowest_dimension", "relevance")
        score = eval_result.get("lowest_score", 0.5)
        violations = eval_result.get("guardrail_violations", [])

        instructions = {
            "relevance": "IMPROVE RELEVANCE: The response does not directly address the user's question. Focus on answering exactly what was asked before adding recommendations.",
            "persona_fit": "IMPROVE PERSONA FIT: Adjust the language complexity and tone to match the user's profile. Use simpler language for beginners, more technical language for experienced investors.",
            "recommendation_quality": "IMPROVE RECOMMENDATIONS: Make recommendations more specific and personalised. Reference the user's stated goals or profile attributes. Avoid generic suggestions.",
            "brand_safety": f"FIX BRAND SAFETY: Remove these violations: {', '.join(violations)}. Never mention competitors. Never give specific buy/sell advice.",
            "conciseness": "IMPROVE CONCISENESS: The response is too long. Trim to the essential points. Maximum 4 sentences for beginners, 6 for experienced users.",
        }

        base = instructions.get(lowest, "Improve the overall quality of the response.")

        if score < 0.5:
            base = "CRITICAL — " + base

        return base

    def graceful_fallback(self, profile: dict) -> str:
        """Generate a safe fallback response when quality is too low."""
        persona = profile.get("persona_tag", "first_time_investor")

        fallbacks = {
            "first_time_investor": (
                "Let me point you to some helpful resources. "
                "You can explore ET's beginner's guide at economictimes.indiatimes.com/markets/beginners-guide, "
                "or try the SIP calculator at economictimes.indiatimes.com/wealth/calculators to see how your savings can grow. "
                "Feel free to ask me anything specific about getting started with investing."
            ),
            "seasoned_trader": (
                "For the latest market insights, check ET Markets at economictimes.indiatimes.com/markets "
                "and ET Prime at economictimes.indiatimes.com/prime for in-depth analysis. "
                "Let me know what specific market or investment topic you'd like to explore."
            ),
            "lapsed_subscriber": (
                "Welcome back. A lot has changed since you were last here — new features, new content, "
                "and a special returning-member offer on ET Prime. "
                "Check out economictimes.indiatimes.com/prime to see what's new. "
                "What were you most interested in during your last visit?"
            ),
        }

        return fallbacks.get(persona, fallbacks["first_time_investor"])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    async def _llm_evaluate(
        self,
        response: str,
        profile: dict,
        user_input: str,
        context: dict,
        persona: str,
    ) -> dict:
        """Use LLM to evaluate relevance, persona fit, and recommendation quality."""
        prompt = f"""Evaluate this AI concierge response for quality. Score each dimension 0.0 to 1.0.

USER MESSAGE: "{user_input}"

AI RESPONSE: "{response}"

USER PERSONA: {persona}
USER PROFILE: investment_experience={profile.get('investment_experience', 'unknown')}, goal={profile.get('primary_financial_goal', 'unknown')}

Score these dimensions:
1. relevance (0-1): Does the response directly address what the user asked?
2. persona_fit (0-1): Is the tone and complexity right for this persona? (beginners need simple language, traders can handle jargon)
3. recommendation_quality (0-1): Are any product recommendations specific, personalised, and not generic?

Respond ONLY with JSON: {{"relevance": 0.8, "persona_fit": 0.7, "recommendation_quality": 0.6}}"""

        try:
            result = await self.llm.generate_structured(prompt, "You are a quality evaluator. Be strict but fair.")
            parsed = result.get("parsed", {})
            return {
                "relevance": min(max(float(parsed.get("relevance", 0.7)), 0.0), 1.0),
                "persona_fit": min(max(float(parsed.get("persona_fit", 0.7)), 0.0), 1.0),
                "recommendation_quality": min(max(float(parsed.get("recommendation_quality", 0.7)), 0.0), 1.0),
            }
        except Exception:
            # If LLM eval fails, use moderate defaults
            return {"relevance": 0.7, "persona_fit": 0.7, "recommendation_quality": 0.7}

    def _get_improvement_hint(self, dimension: str, score: float, persona: str) -> str:
        hints = {
            "relevance": "Focus on answering the user's actual question first.",
            "persona_fit": f"Adjust language for '{persona}' persona — {'simpler, warmer' if persona == 'first_time_investor' else 'more technical, concise'}.",
            "recommendation_quality": "Make recommendations more specific to this user's profile.",
            "brand_safety": "Review for competitor mentions and financial advice boundaries.",
            "conciseness": "Shorten the response — focus on the key message.",
        }
        return hints.get(dimension, "Improve overall quality.")


# ---------------------------------------------------------------------------
# Self-test with sample responses
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _test():
        evaluator = EvaluatorAgent()

        test_cases = [
            # Good responses
            {
                "name": "Good beginner response",
                "response": "Great question. Starting with a SIP of even ₹500 per month is a smart first step. Try ET Money's SIP calculator to see how your money can grow over time, and then start your first SIP right through ET Money — it's free and takes just 5 minutes.",
                "profile": {"persona_tag": "first_time_investor", "investment_experience": "none", "primary_financial_goal": "wealth_creation"},
                "user_input": "I want to start investing but don't know where to begin.",
                "expected": "pass",
            },
            {
                "name": "Good trader response",
                "response": "Nifty's at a key support zone around 24,200. ET Markets Pro has a new AI-powered screener that can help you identify oversold opportunities in the current dip. Worth checking the smallcap heat map — some quality names are trading at 15-20% below their 52-week highs.",
                "profile": {"persona_tag": "seasoned_trader", "investment_experience": "advanced", "primary_financial_goal": "wealth_creation"},
                "user_input": "Markets are down today. Any opportunities?",
                "expected": "pass",
            },
            # Bad responses
            {
                "name": "Bad — mentions competitor",
                "response": "You should check Zerodha for trading. They have great tools. Also try Groww for mutual funds. ET doesn't really compare.",
                "profile": {"persona_tag": "first_time_investor", "investment_experience": "none"},
                "user_input": "Where should I invest?",
                "expected": "fail",
            },
            {
                "name": "Bad — too complex for beginner",
                "response": "Consider building a portfolio with alpha-generating strategies focused on high-CAGR smallcaps with strong P/E ratios and improving ROCE metrics. Diversify across sectors with balanced beta exposure. Track your portfolio's Sharpe ratio quarterly.",
                "profile": {"persona_tag": "first_time_investor", "investment_experience": "none"},
                "user_input": "I want to save money better.",
                "expected": "fail",
            },
            {
                "name": "Bad — generic and irrelevant",
                "response": "Welcome to Economic Times. We have many products. Please explore our website for more information. Thank you for visiting.",
                "profile": {"persona_tag": "lapsed_subscriber", "investment_experience": "intermediate"},
                "user_input": "I used to have ET Prime but my subscription lapsed. What's new?",
                "expected": "fail",
            },
        ]

        print("=== Evaluator Self-Test ===\n")
        for tc in test_cases:
            result = await evaluator.score_response(
                tc["response"], tc["profile"], tc["user_input"]
            )
            status = "PASS" if result["passed"] else "FAIL"
            expected_match = "✓" if (status == tc["expected"].upper()) else "✗"

            print(f"{expected_match} [{status}] {tc['name']}")
            print(f"  Score: {result['overall_score']} | Lowest: {result['lowest_dimension']} ({result['lowest_score']})")
            if result["guardrail_violations"]:
                print(f"  Violations: {result['guardrail_violations']}")
            if not result["passed"]:
                instruction = evaluator.generate_improvement_instruction(result)
                print(f"  Improvement: {instruction}")
            print()

        # Test fallback
        print("=== Graceful Fallback ===")
        fb = evaluator.graceful_fallback({"persona_tag": "first_time_investor"})
        print(f"  Beginner fallback: {fb}\n")

    asyncio.run(_test())
