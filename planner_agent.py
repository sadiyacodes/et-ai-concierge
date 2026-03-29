"""
ET AI Concierge — Planner Agent
Builds and maintains personalised ET product journeys for each user.
"""

import asyncio
from tools import get_llm

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
PLANNER_SYSTEM_PROMPT = """You are the ET Journey Architect. Given a user profile, you build the optimal path through the ET ecosystem — introducing the right products at the right time, never overwhelming, always adding value.

Rules:
1. Start with the user's most immediate need. Don't begin with a premium upsell.
2. Each journey step must deliver standalone value before asking anything of the user.
3. For beginners: start with free tools (calculators, beginner articles), then ET Money, then ET Prime.
4. For seasoned traders: start with ET Markets advanced features, then ET Prime, then Wealth Summit.
5. A journey has phases: Discover → Engage → Commit → Deepen. Don't skip phases.
6. Maximum 3 journey steps visible at a time. Reveal more as user completes steps.
7. Each step must have a clear, single action. No "explore our range" vagueness."""


# ---------------------------------------------------------------------------
# Journey Templates
# ---------------------------------------------------------------------------
JOURNEY_TEMPLATES = {
    "first_time_investor": {
        "name": "Your First Steps to Investing",
        "phases": {
            "discover": [
                {"step_id": "calc_sip", "product_id": "et_sip_calculator", "action": "Try the SIP calculator — see how ₹5,000/month can grow over 20 years", "value": "Visualise compound growth with your own numbers", "cta": "Open SIP Calculator", "url": "https://economictimes.indiatimes.com/wealth/calculators"},
                {"step_id": "read_guide", "product_id": "et_markets_beginner_guide", "action": "Read the beginner's guide — 5 minutes to understand the basics", "value": "Plain-English explanation of stocks, SIPs, and mutual funds", "cta": "Read the Guide", "url": "https://economictimes.indiatimes.com/markets/beginners-guide"},
                {"step_id": "watch_webinar", "product_id": "et_markets_webinars", "action": "Watch a free investing webinar — no jargon, real examples", "value": "Ask questions live to market experts", "cta": "Browse Webinars", "url": "https://economictimes.indiatimes.com/markets/webinars"},
            ],
            "engage": [
                {"step_id": "start_et_money", "product_id": "et_money", "action": "Start your first SIP on ET Money — even ₹500 is a great start", "value": "Zero-commission direct mutual funds, goal-based investing", "cta": "Start Investing", "url": "https://www.etmoney.com"},
                {"step_id": "setup_watchlist", "product_id": "et_markets_app", "action": "Set up a watchlist on ET Markets — track 5 stocks you hear about", "value": "See how markets move daily, learn to read price charts", "cta": "Create Watchlist", "url": "https://economictimes.indiatimes.com/markets"},
            ],
            "commit": [
                {"step_id": "try_prime", "product_id": "et_prime", "action": "Unlock ET Prime — deeper analysis to make better investment decisions", "value": "Expert insights that explain the 'why' behind market moves", "cta": "Try ET Prime", "url": "https://economictimes.indiatimes.com/prime"},
                {"step_id": "masterclass", "product_id": "et_masterclass_investing", "action": "Enrol in the Investing Masterclass for structured learning", "value": "Complete investing education with certificate", "cta": "View Masterclass", "url": "https://economictimes.indiatimes.com/masterclass"},
            ],
            "deepen": [
                {"step_id": "health_check", "product_id": "et_money", "action": "Run a portfolio health check on ET Money", "value": "See if your investments are on track for your goals", "cta": "Check Portfolio", "url": "https://www.etmoney.com"},
                {"step_id": "wealth_read", "product_id": "et_wealth", "action": "Subscribe to ET Wealth weekly newsletter", "value": "Stay updated on personal finance moves each week", "cta": "Subscribe Free", "url": "https://economictimes.indiatimes.com/wealth"},
            ],
        },
    },
    "seasoned_trader": {
        "name": "Advanced Market Intelligence",
        "phases": {
            "discover": [
                {"step_id": "markets_pro", "product_id": "et_markets_app", "action": "Explore ET Markets Pro — advanced charting and AI stock screener", "value": "Technical analysis tools and smart screeners", "cta": "Try Markets Pro", "url": "https://economictimes.indiatimes.com/markets"},
                {"step_id": "prime_analysis", "product_id": "et_prime", "action": "Read today's ET Prime market deep-dive", "value": "Institutional-grade analysis of market trends and sectors", "cta": "Read Analysis", "url": "https://economictimes.indiatimes.com/prime"},
            ],
            "engage": [
                {"step_id": "subscribe_prime", "product_id": "et_prime", "action": "Subscribe to ET Prime for daily expert analysis", "value": "The 'why' behind market moves — before the crowd knows", "cta": "Subscribe ₹999/yr", "url": "https://economictimes.indiatimes.com/prime"},
                {"step_id": "summit_register", "product_id": "et_wealth_summit", "action": "Register for the ET Wealth Summit — meet market leaders", "value": "Direct access to fund managers and market movers", "cta": "Register Now", "url": "https://economictimes.indiatimes.com/wealth-summit"},
            ],
            "commit": [
                {"step_id": "wealth_mgmt", "product_id": "et_wealth_management", "action": "Connect with a SEBI-registered wealth manager for your portfolio", "value": "Personalised advisory for portfolios above ₹50L", "cta": "Book Consultation", "url": "https://economictimes.indiatimes.com/wealth/wealth-management"},
            ],
            "deepen": [
                {"step_id": "masterclass_adv", "product_id": "et_masterclass_investing", "action": "Join the advanced investing masterclass — smallcap strategies", "value": "Learn systematic approaches from veteran fund managers", "cta": "Explore Course", "url": "https://economictimes.indiatimes.com/masterclass"},
            ],
        },
    },
    "lapsed_subscriber": {
        "name": "Welcome Back — Here's What You Missed",
        "phases": {
            "discover": [
                {"step_id": "whats_new", "product_id": "et_prime", "action": "See what's new on ET Prime since you left", "value": "New smallcap deep-dives, AI screener, portfolio X-ray, and more", "cta": "See What's New", "url": "https://economictimes.indiatimes.com/prime"},
                {"step_id": "free_stories", "product_id": "et_prime", "action": "Read 3 unlocked Prime stories — on us", "value": "See how the analysis has evolved", "cta": "Read Free Stories", "url": "https://economictimes.indiatimes.com/prime"},
            ],
            "engage": [
                {"step_id": "resubscribe", "product_id": "et_prime", "action": "Re-subscribe at the returning-member rate: ₹799/year", "value": "Locked-in price for life — lower than the standard ₹999", "cta": "Resubscribe ₹799", "url": "https://economictimes.indiatimes.com/prime"},
                {"step_id": "webinar_catch", "product_id": "et_markets_webinars", "action": "Catch the replay of the latest smallcap webinar you missed", "value": "Our most-attended webinar series is back with new speakers", "cta": "Watch Replay", "url": "https://economictimes.indiatimes.com/markets/webinars"},
            ],
            "commit": [
                {"step_id": "portfolio_xray", "product_id": "et_money", "action": "Try the new Portfolio X-Ray tool on ET Money", "value": "Free health check for your mutual fund portfolio", "cta": "Scan Portfolio", "url": "https://www.etmoney.com"},
            ],
            "deepen": [
                {"step_id": "summit_invite", "product_id": "et_wealth_summit", "action": "Priority registration for the ET Wealth Summit 2026", "value": "Returning subscribers get first access to event tickets", "cta": "Register Early", "url": "https://economictimes.indiatimes.com/wealth-summit"},
            ],
        },
    },
    "wealth_builder": {
        "name": "Build & Protect Your Wealth",
        "phases": {
            "discover": [
                {"step_id": "wealth_read", "product_id": "et_wealth", "action": "Read this week's ET Wealth top story", "value": "Actionable personal finance insights from trusted experts", "cta": "Read Now", "url": "https://economictimes.indiatimes.com/wealth"},
                {"step_id": "calc_retire", "product_id": "et_sip_calculator", "action": "Calculate your retirement corpus need", "value": "Know exactly how much you need and how to get there", "cta": "Calculate Now", "url": "https://economictimes.indiatimes.com/wealth/calculators"},
            ],
            "engage": [
                {"step_id": "et_money_goal", "product_id": "et_money", "action": "Set up goal-based investing on ET Money", "value": "Automated investing linked to your life goals", "cta": "Start Goals", "url": "https://www.etmoney.com"},
                {"step_id": "prime_wealth", "product_id": "et_prime", "action": "Subscribe to ET Prime for personal finance analysis", "value": "Tax strategies, MF analysis, insurance reviews", "cta": "Subscribe", "url": "https://economictimes.indiatimes.com/prime"},
            ],
            "commit": [
                {"step_id": "insurance_check", "product_id": "et_term_insurance_partner", "action": "Check if you have adequate life and health cover", "value": "Protect what you're building — compare plans in 2 minutes", "cta": "Compare Plans", "url": "https://economictimes.indiatimes.com/wealth/term-insurance"},
            ],
            "deepen": [
                {"step_id": "summit", "product_id": "et_wealth_summit", "action": "Attend the ET Wealth Summit for comprehensive planning", "value": "Build a complete wealth plan with expert guidance", "cta": "Register", "url": "https://economictimes.indiatimes.com/wealth-summit"},
            ],
        },
    },
}


class PlannerAgent:
    """Builds and maintains personalised ET product journeys."""

    def __init__(self):
        self.llm = get_llm()

    async def build_journey(self, profile: dict) -> dict:
        """Create initial journey based on user profile."""
        persona = profile.get("persona_tag", "first_time_investor")
        template_key = persona if persona in JOURNEY_TEMPLATES else "first_time_investor"
        template = JOURNEY_TEMPLATES[template_key]

        discover_steps = template["phases"].get("discover", [])

        journey_state = {
            "phase": "discover",
            "completed_steps": [],
            "current_step": discover_steps[0] if discover_steps else None,
            "upcoming_steps": discover_steps[1:3],  # max 3 visible
            "persona_journey": template_key,
            "journey_name": template["name"],
        }

        return journey_state

    async def advance_journey(self, state: dict, user_action: str) -> dict:
        """Progress journey based on user action."""
        persona_key = state.get("persona_journey", "first_time_investor")
        template = JOURNEY_TEMPLATES.get(persona_key, JOURNEY_TEMPLATES["first_time_investor"])

        current = state.get("current_step")
        if current:
            state["completed_steps"].append(current["step_id"])

        phase = state["phase"]
        phases_order = ["discover", "engage", "commit", "deepen"]
        phase_steps = template["phases"].get(phase, [])

        # Find next uncompleted step in current phase
        next_step = None
        upcoming = []
        for step in phase_steps:
            if step["step_id"] not in state["completed_steps"]:
                if next_step is None:
                    next_step = step
                else:
                    upcoming.append(step)

        # If current phase is exhausted, move to next
        if next_step is None:
            current_idx = phases_order.index(phase) if phase in phases_order else 0
            for next_phase in phases_order[current_idx + 1:]:
                next_phase_steps = template["phases"].get(next_phase, [])
                for step in next_phase_steps:
                    if step["step_id"] not in state["completed_steps"]:
                        if next_step is None:
                            next_step = step
                            state["phase"] = next_phase
                        else:
                            upcoming.append(step)
                if next_step:
                    break

        state["current_step"] = next_step
        state["upcoming_steps"] = upcoming[:2]  # max 2 upcoming + 1 current = 3 visible

        return state

    def get_current_cta(self, state: dict) -> str:
        """Return single clear next action."""
        current = state.get("current_step")
        if current:
            return f"{current['action']} → {current['cta']}"
        return "You've explored the key tools. Browse ET at your own pace."

    def get_journey_progress(self, state: dict) -> dict:
        """Return journey progress for UI display."""
        phases_order = ["discover", "engage", "commit", "deepen"]
        current_phase = state.get("phase", "discover")
        current_idx = phases_order.index(current_phase) if current_phase in phases_order else 0

        return {
            "phases": [
                {
                    "name": phase,
                    "status": "completed" if i < current_idx else ("active" if i == current_idx else "upcoming"),
                }
                for i, phase in enumerate(phases_order)
            ],
            "completed_count": len(state.get("completed_steps", [])),
            "current_step": state.get("current_step"),
            "upcoming_steps": state.get("upcoming_steps", []),
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _test():
        agent = PlannerAgent()

        # Test beginner journey
        profile = {"persona_tag": "first_time_investor"}
        journey = await agent.build_journey(profile)
        print("=== Beginner Journey ===")
        print(f"Phase: {journey['phase']}")
        print(f"Current: {journey['current_step']['action']}")
        print(f"CTA: {agent.get_current_cta(journey)}")

        # Advance
        journey = await agent.advance_journey(journey, "clicked_sip_calculator")
        print(f"\nAfter advance -> Phase: {journey['phase']}")
        print(f"Current: {journey['current_step']['action'] if journey['current_step'] else 'None'}")

        # Advance again
        journey = await agent.advance_journey(journey, "read_guide")
        print(f"\nAfter advance -> Phase: {journey['phase']}")
        print(f"Current: {journey['current_step']['action'] if journey['current_step'] else 'None'}")

        # Test trader journey
        print("\n=== Trader Journey ===")
        trader_profile = {"persona_tag": "seasoned_trader"}
        journey = await agent.build_journey(trader_profile)
        print(f"Phase: {journey['phase']}")
        print(f"Current: {journey['current_step']['action']}")

        # Test progress display
        progress = agent.get_journey_progress(journey)
        print(f"\nProgress: {progress}")

    asyncio.run(_test())
