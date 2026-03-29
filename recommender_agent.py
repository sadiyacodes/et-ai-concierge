"""
ET AI Concierge — Recommender Agent
Maps user profiles to specific ET products with scoring and personalisation.
Enriched with live ET content via Exa Search API.
"""

import asyncio
from tools import get_llm, get_exa
from et_knowledge_base import ETKnowledgeBase

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
RECOMMENDER_SYSTEM_PROMPT = """You are the ET Product Recommendation Engine. You match users to ET products with precision and honesty.

Rules:
1. Never recommend more than 3 products in a single turn.
2. Always explain WHY a product fits this specific user — reference their profile.
3. Lead with the most accessible/easiest-to-adopt product, not the most premium.
4. For returning/lapsed users: surface what's NEW since they left — don't just repeat what they already know.
5. Never mention competitor platforms (Moneycontrol, Mint, Zerodha, Groww, etc.).
6. Tone for beginners: warm, jargon-free, encouraging.
7. Tone for experienced investors: peer-to-peer, data-driven, concise.
8. Every recommendation must include a specific call-to-action (CTA) with a real ET URL."""


class RecommenderAgent:
    """Product recommendation engine with persona-specific paths."""

    def __init__(self, knowledge_base: ETKnowledgeBase | None = None):
        self.llm = get_llm()
        self.kb = knowledge_base or ETKnowledgeBase(use_chroma=True)

    async def recommend_for_beginner(self, profile: dict) -> dict:
        """
        Beginner path: 1 free tool + 1 learning resource + 1 investment tool.
        Avoids premium products and complex instruments.
        """
        # Score all products for this profile
        profile_for_scoring = {**profile, "persona_tag": "first_time_investor"}
        ranked = self.kb.get_ranked_recommendations(profile_for_scoring, limit=10)

        # Categorise into buckets
        free_tools = []
        learning = []
        invest_tools = []

        for r in ranked:
            product = self.kb.get_product(r["product_id"])
            if not product:
                continue
            cat = product["category"]
            price = product.get("price", "")

            if "Free" in price and cat == "content":
                free_tools.append((r, product))
            elif cat == "learning" or "beginner" in product["id"]:
                learning.append((r, product))
            elif cat == "investment":
                invest_tools.append((r, product))
            elif "Free" in price:
                free_tools.append((r, product))

        # Pick best from each bucket
        recs = []

        if free_tools:
            r, p = free_tools[0]
            recs.append(self._format_rec(r, p, profile, "This is completely free and takes just 2 minutes to explore."))

        if learning:
            r, p = learning[0]
            recs.append(self._format_rec(r, p, profile, "A great way to build your fundamentals at your own pace."))
        elif free_tools and len(free_tools) > 1:
            r, p = free_tools[1]
            recs.append(self._format_rec(r, p, profile, "Another free resource to build your confidence."))

        if invest_tools:
            r, p = invest_tools[0]
            recs.append(self._format_rec(r, p, profile, "When you're ready, start with as little as ₹500/month."))

        # Fallback: if we don't have enough variety, fill with top-ranked
        if len(recs) < 2:
            for r in ranked:
                if r["product_id"] not in [rec["product_id"] for rec in recs]:
                    product = self.kb.get_product(r["product_id"])
                    if product:
                        recs.append(self._format_rec(r, product, profile))
                if len(recs) >= 3:
                    break

        headline = self._generate_beginner_headline(profile)

        return {
            "recommendations": recs[:3],
            "headline": headline,
            "persona_used": "first_time_investor",
            "confidence": 0.85,
        }

    async def recommend_for_trader(self, profile: dict) -> dict:
        """
        Trader path: advanced ET Markets features + ET Prime analysis + relevant event.
        Uses market terminology, performance data references, peer comparison.
        """
        profile_for_scoring = {**profile, "persona_tag": "seasoned_trader"}
        ranked = self.kb.get_ranked_recommendations(profile_for_scoring, limit=10)

        market_tools = []
        analysis = []
        events = []

        for r in ranked:
            product = self.kb.get_product(r["product_id"])
            if not product:
                continue
            cat = product["category"]

            if "markets" in product["id"].lower() and cat != "event":
                market_tools.append((r, product))
            elif cat == "subscription" or "prime" in product["id"].lower():
                analysis.append((r, product))
            elif cat == "event":
                events.append((r, product))

        recs = []
        if market_tools:
            r, p = market_tools[0]
            recs.append(self._format_rec(r, p, profile, "Advanced charting and AI-powered screener — built for active traders."))

        if analysis:
            r, p = analysis[0]
            recs.append(self._format_rec(r, p, profile, "Institutional-grade analysis — the edge that compounds over time."))

        if events:
            r, p = events[0]
            recs.append(self._format_rec(r, p, profile, "Network with fund managers and get insights before they go mainstream."))

        # Fill if needed
        if len(recs) < 3:
            for r in ranked:
                if r["product_id"] not in [rec["product_id"] for rec in recs]:
                    product = self.kb.get_product(r["product_id"])
                    if product:
                        recs.append(self._format_rec(r, product, profile))
                if len(recs) >= 3:
                    break

        return {
            "recommendations": recs[:3],
            "headline": "Tools and insights for serious market participants:",
            "persona_used": "seasoned_trader",
            "confidence": 0.85,
        }

    async def recommend_for_lapsed(self, profile: dict, days_lapsed: int = 90) -> dict:
        """
        Lapsed subscriber path: surface what's NEW, specific to their interests.
        Frame as 'here's what you missed' — not generic 'come back' messaging.
        """
        new_content = self.kb.get_new_since(days_lapsed)
        preferred = profile.get("preferred_content_types", [])

        # Filter new content by user interests
        relevant_new = []
        for item in new_content:
            product = self.kb.get_product(item.get("product_id", ""))
            if product:
                # Check if it matches their interests
                text = f"{item['title']} {product.get('name', '')}".lower()
                if any(pref in text for pref in preferred) or not preferred:
                    relevant_new.append((item, product))

        recs = []

        # Surface 2-3 new things
        for item, product in relevant_new[:2]:
            rec = {
                "product_id": product["id"],
                "product_name": product["name"],
                "relevance_score": 0.85,
                "user_facing_reason": f"New since you last visited: {item['title']}",
                "cta_text": "Check it out",
                "cta_url": product["url"],
                "is_free": "Free" in product.get("price", ""),
                "estimated_time_to_value": "5 minutes",
            }
            recs.append(rec)

        # Add the re-engagement offer
        recs.append({
            "product_id": "et_prime",
            "product_name": "ET Prime — Returning Member Offer",
            "relevance_score": 0.9,
            "user_facing_reason": f"Returning members get ET Prime at ₹799/year (locked for life) — ₹200 less than standard price.",
            "cta_text": "Resubscribe at ₹799/year",
            "cta_url": "https://economictimes.indiatimes.com/prime",
            "is_free": False,
            "estimated_time_to_value": "immediate",
        })

        # Build headline referencing their specific interests
        interest_str = ", ".join(preferred[:2]) if preferred else "financial markets"
        headline = f"Here's what's changed in the {interest_str} space since you were last here:"

        return {
            "recommendations": recs[:3],
            "headline": headline,
            "persona_used": "lapsed_subscriber",
            "confidence": 0.8,
        }

    async def recommend(self, profile: dict, kb: ETKnowledgeBase | None = None, user_query: str = "") -> dict:
        """Unified recommend method — routes to the right path, enriched with live Exa content."""
        if kb:
            self.kb = kb

        persona = profile.get("persona_tag", "first_time_investor")
        sub_status = profile.get("et_subscription_status", "none")

        if sub_status == "lapsed" or persona == "lapsed_subscriber":
            days = profile.get("days_since_last_visit", 90)
            result = await self.recommend_for_lapsed(profile, days)
        elif persona in ("seasoned_trader", "advanced"):
            result = await self.recommend_for_trader(profile)
        else:
            result = await self.recommend_for_beginner(profile)

        # Enrich with live ET content from Exa
        if user_query:
            exa_articles = await self._fetch_exa_articles(user_query, persona)
            if exa_articles:
                result["live_articles"] = exa_articles

        return result

    async def _fetch_exa_articles(self, query: str, persona: str = "") -> list[dict]:
        """Fetch relevant live articles from ET via Exa Search."""
        try:
            exa = get_exa()
            if not exa.enabled:
                return []
            articles = await exa.search_for_topic(query, persona)
            return [
                {
                    "title": a["title"],
                    "url": a["url"],
                    "snippet": a["snippet"][:200] if a["snippet"] else "",
                    "published_date": a.get("published_date", ""),
                    "source": "exa_live",
                }
                for a in articles[:3]
            ]
        except Exception as e:
            print(f"[Recommender] Exa enrichment failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _format_rec(self, score_result: dict, product: dict, profile: dict, extra_reason: str = "") -> dict:
        persona = profile.get("persona_tag", "first_time_investor")
        goal = profile.get("primary_financial_goal", "")

        # Generate personalised reason
        reason = score_result.get("reasoning", product.get("value_proposition", ""))
        if extra_reason:
            reason = extra_reason

        if goal and goal != "":
            reason_prefix = {
                "wealth_creation": "As you build long-term wealth, ",
                "tax_saving": "For your tax planning needs, ",
                "retirement": "To prepare for your retirement, ",
                "child_education": "For your child's education planning, ",
                "home_purchase": "As you plan your home purchase, ",
                "emergency_fund": "While you build your safety net, ",
            }
            prefix = reason_prefix.get(goal, "")
            if prefix and not reason.startswith(prefix):
                reason = prefix + reason[0].lower() + reason[1:]

        return {
            "product_id": product["id"],
            "product_name": product["name"],
            "relevance_score": score_result.get("score", 0.5),
            "user_facing_reason": reason,
            "cta_text": product.get("onboarding_steps", ["Get started"])[0] if product.get("onboarding_steps") else "Get started",
            "cta_url": product["url"],
            "is_free": "Free" in product.get("price", ""),
            "estimated_time_to_value": "5 minutes" if "Free" in product.get("price", "") else "1 week",
        }

    def _generate_beginner_headline(self, profile: dict) -> str:
        goal = profile.get("primary_financial_goal", "")
        headlines = {
            "wealth_creation": "Let's start building your wealth — here are 3 easy first steps:",
            "tax_saving": "Smart tax saving starts here — 3 tools to get you going:",
            "retirement": "Your retirement planning starts today — here's where to begin:",
            "child_education": "Planning for your child's future? Start with these:",
            "home_purchase": "Saving for a home? These tools will help you plan:",
            "emergency_fund": "Building a safety net is smart — here's how to start:",
        }
        return headlines.get(goal, "Here's a personalised starting point based on what you've shared:")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _test():
        agent = RecommenderAgent()

        # Test beginner
        print("=== Beginner Recommendations ===")
        profile = {
            "persona_tag": "first_time_investor",
            "investment_experience": "none",
            "primary_financial_goal": "wealth_creation",
            "profile_completeness": 0.6,
            "et_products_used": [],
            "detected_life_events": [],
            "income_band": "5-10L",
        }
        result = await agent.recommend_for_beginner(profile)
        print(f"Headline: {result['headline']}")
        for rec in result["recommendations"]:
            print(f"  • {rec['product_name']} (score: {rec['relevance_score']})")
            print(f"    {rec['user_facing_reason']}")
            print(f"    CTA: {rec['cta_text']} → {rec['cta_url']}")

        # Test trader
        print("\n=== Trader Recommendations ===")
        trader_profile = {
            "persona_tag": "seasoned_trader",
            "investment_experience": "advanced",
            "primary_financial_goal": "wealth_creation",
            "profile_completeness": 0.7,
            "et_products_used": ["et_markets"],
            "detected_life_events": [],
        }
        result = await agent.recommend_for_trader(trader_profile)
        print(f"Headline: {result['headline']}")
        for rec in result["recommendations"]:
            print(f"  • {rec['product_name']}: {rec['user_facing_reason']}")

        # Test lapsed
        print("\n=== Lapsed Recommendations ===")
        lapsed_profile = {
            "persona_tag": "lapsed_subscriber",
            "et_subscription_status": "lapsed",
            "days_since_last_visit": 90,
            "preferred_content_types": ["markets", "smallcap"],
            "investment_experience": "intermediate",
            "et_products_used": ["et_prime", "et_markets"],
            "detected_life_events": [],
            "primary_financial_goal": "wealth_creation",
        }
        result = await agent.recommend_for_lapsed(lapsed_profile, 90)
        print(f"Headline: {result['headline']}")
        for rec in result["recommendations"]:
            print(f"  • {rec['product_name']}: {rec['user_facing_reason']}")

    asyncio.run(_test())
