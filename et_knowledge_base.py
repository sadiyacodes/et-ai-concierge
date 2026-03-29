"""
ET AI Concierge — Knowledge Base & Product Registry
Complete ET product catalog with ChromaDB vector search and recommendation scoring.
"""

import os
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
# ET Product Catalog
# ---------------------------------------------------------------------------
ET_PRODUCTS: list[dict] = [
    {
        "id": "et_prime",
        "name": "ET Prime",
        "category": "subscription",
        "description": "Premium subscription offering in-depth analysis, exclusive stories, and expert opinion from India's most trusted business publication. Unlock stories behind the headlines with ad-free reading.",
        "url": "https://economictimes.indiatimes.com/prime",
        "price": "₹999/year",
        "target_personas": ["seasoned_trader", "wealth_builder", "nri"],
        "trigger_signals": ["wants_deep_analysis", "follows_markets_regularly", "reads_et_daily"],
        "objection_handlers": {
            "too_expensive": "At ₹999/year that is less than ₹3/day — less than a cup of tea. And every insight that helps you make one better investment decision pays for itself many times over.",
            "free_content_enough": "Free headlines give you the what. ET Prime gives you the why and what-next — the analysis that actually moves your portfolio.",
            "already_have_competitor": "ET Prime's strength is the Economic Times newsroom — India's largest business journalism team with 40+ years of trust. The depth of coverage is unmatched.",
        },
        "upsell_from": ["et_markets_beginner_guide", "et_wealth"],
        "cross_sell_to": ["et_markets_app", "et_wealth_summit"],
        "value_proposition": "Get the analysis behind the headlines — expert insights that help you make smarter financial decisions.",
        "onboarding_steps": [
            "Browse 3 free Prime stories to see the depth of analysis",
            "Start your subscription at ₹999/year",
            "Set up your personalised morning brief",
            "Follow topics and authors that match your interests",
        ],
    },
    {
        "id": "et_markets_app",
        "name": "ET Markets App",
        "category": "content",
        "description": "Real-time stock market data, portfolio tracker, live Nifty/Sensex updates, expert stock picks, and interactive charts. Your complete market companion.",
        "url": "https://economictimes.indiatimes.com/markets",
        "price": "Free",
        "target_personas": ["seasoned_trader", "wealth_builder", "first_time_investor"],
        "trigger_signals": ["tracks_stocks", "wants_market_updates", "portfolio_tracking"],
        "objection_handlers": {
            "already_use_broker_app": "ET Markets complements your broker — it gives you the news and analysis layer. Know WHY stocks move, not just that they moved.",
            "too_much_info": "Start with just 5-6 stocks on your watchlist. The app learns what you follow and tailors updates.",
        },
        "upsell_from": ["et_markets_beginner_guide"],
        "cross_sell_to": ["et_prime", "et_money"],
        "value_proposition": "Track markets in real-time with expert analysis and a powerful portfolio tracker.",
        "onboarding_steps": [
            "Download ET Markets app or visit the web portal",
            "Create your watchlist with stocks you own or follow",
            "Enable push notifications for your watchlist stocks",
            "Explore expert stock picks and market analysis",
        ],
    },
    {
        "id": "et_money",
        "name": "ET Money",
        "category": "investment",
        "description": "Invest in direct mutual funds with zero commission. Features SIP setup, goal-based investing, portfolio health check, and tax-saving ELSS funds. Trusted by 10M+ Indians.",
        "url": "https://www.etmoney.com",
        "price": "Free (direct MF — zero commission)",
        "target_personas": ["first_time_investor", "wealth_builder"],
        "trigger_signals": ["wants_to_start_investing", "sip_interest", "mutual_fund_interest", "tax_saving"],
        "objection_handlers": {
            "already_invest_elsewhere": "ET Money offers direct plans which have ~1% lower expense ratio. Over 20 years, that 1% compounds into lakhs of extra returns.",
            "too_complicated": "You can start with just one SIP of ₹500/month. The app suggests funds based on your goals — no research needed.",
            "is_it_safe": "ET Money is a SEBI-registered investment advisor. Your money goes directly to the AMC, not to ET. Fully regulated and transparent.",
        },
        "upsell_from": ["et_sip_calculator", "et_markets_beginner_guide"],
        "cross_sell_to": ["et_prime", "et_masterclass_investing"],
        "value_proposition": "Start investing in mutual funds with zero commission — as little as ₹500/month.",
        "onboarding_steps": [
            "Download ET Money app or visit etmoney.com",
            "Complete quick KYC (takes 5 minutes with Aadhaar)",
            "Set your first goal (retirement, tax saving, or wealth)",
            "Start a SIP — even ₹500/month is a great beginning",
        ],
    },
    {
        "id": "et_wealth",
        "name": "ET Wealth",
        "category": "content",
        "description": "India's leading personal finance publication. Weekly insights on tax planning, mutual funds, insurance, real estate, and retirement planning from trusted ET experts.",
        "url": "https://economictimes.indiatimes.com/personal-finance",
        "price": "Free (web) / Part of ET Prime",
        "target_personas": ["wealth_builder", "first_time_investor", "family_builder"],
        "trigger_signals": ["personal_finance_interest", "tax_planning", "insurance_query", "retirement_planning"],
        "objection_handlers": {
            "too_much_reading": "Start with the weekly digest — 5-minute summary of the most important personal finance moves for the week.",
        },
        "upsell_from": [],
        "cross_sell_to": ["et_prime", "et_money", "et_wealth_summit"],
        "value_proposition": "Master your personal finances with India's most trusted personal finance experts.",
        "onboarding_steps": [
            "Visit ET Wealth section on the website",
            "Read this week's top 3 stories",
            "Subscribe to the ET Wealth weekly newsletter",
            "Use the tax planning tools and calculators",
        ],
    },
    {
        "id": "et_masterclass_investing",
        "name": "ET Masterclass — Investing",
        "category": "learning",
        "description": "Structured online masterclasses on investing and personal finance — from SIPs to stock picking, value investing, valuation, and financial freedom. Courses include Value Valuation & Sector Dynamics, Financial Freedom Masterclass, and Mutual Funds & ETFs Workshop. Taught by IIM alumni, CFA/CMT holders, and market veterans. 1,00,000+ learners trained, 4.7/5 rating.",
        "url": "https://economictimes.indiatimes.com/masterclass",
        "price": "₹2,999 (often discounted)",
        "target_personas": ["first_time_investor", "wealth_builder"],
        "trigger_signals": ["wants_to_learn_investing", "beginner_questions", "confused_about_markets"],
        "objection_handlers": {
            "free_content_available": "Free content teaches concepts. This masterclass teaches you a complete system — from analysis to execution, with assignments and mentorship.",
            "too_expensive": "One good investment decision from this course pays back the fee 100x. Think of it as investing in your investing skill.",
        },
        "upsell_from": ["et_markets_beginner_guide", "et_sip_calculator"],
        "cross_sell_to": ["et_money", "et_markets_app"],
        "value_proposition": "Learn investing from India's top market experts — structured course with a certificate.",
        "onboarding_steps": [
            "Preview the curriculum and instructor profiles",
            "Enrol and access the first module free",
            "Complete one module per week at your own pace",
            "Join the discussion community for peer learning",
        ],
    },
    {
        "id": "et_masterclass_ai_tech",
        "name": "ET Masterclass — AI & Future Tech",
        "category": "learning",
        "description": "Leverage AI and future technologies to boost efficiency, cut costs, unlock new revenue, and accelerate smarter decisions. Courses include AI Product Builder, GEO+AEO Masterclass, and AI Mastery for Students.",
        "url": "https://economictimes.indiatimes.com/masterclass",
        "price": "₹2,999 (often free introductory sessions)",
        "target_personas": ["wealth_builder", "nri"],
        "trigger_signals": ["ai_interest", "tech_skills", "future_skills", "automation"],
        "objection_handlers": {
            "already_know_ai": "Even AI practitioners discover new workflows. This masterclass covers hands-on product building and practical use cases by industry experts.",
        },
        "upsell_from": ["et_prime"],
        "cross_sell_to": ["et_prime", "et_masterclass_investing"],
        "value_proposition": "Master AI and future technologies with hands-on courses taught by industry experts.",
        "onboarding_steps": [
            "Attend a free AI Product Builder or GEO+AEO intro session",
            "Enrol in the full masterclass program",
            "Build your own AI product during the course",
            "Get certified and apply skills at work",
        ],
    },
    {
        "id": "et_wealth_summit",
        "name": "ET Wealth Summit",
        "category": "event",
        "description": "Annual flagship personal finance event — India's biggest gathering of investors, advisors, and finance leaders. Actionable insights from keynotes, panel discussions, and masterclasses.",
        "url": "https://economictimes.indiatimes.com/wealth",
        "price": "Free to ₹4,999 (tiered)",
        "target_personas": ["seasoned_trader", "wealth_builder"],
        "trigger_signals": ["event_interest", "networking", "expert_access"],
        "objection_handlers": {
            "cant_travel": "Virtual passes are available with full session access and networking rooms.",
            "not_sure_value": "Last year's attendees rated it 4.7/5 for actionable takeaways. You leave with a concrete action plan.",
        },
        "upsell_from": ["et_prime", "et_wealth"],
        "cross_sell_to": ["et_prime", "et_wealth_management"],
        "value_proposition": "India's premier personal finance event — meet the experts, build your wealth plan.",
        "onboarding_steps": [
            "Check the upcoming event dates and agenda",
            "Register for early-bird pricing",
            "Select sessions matching your interests",
            "Prepare 2-3 questions for the expert Q&A",
        ],
    },
    {
        "id": "et_markets_webinars",
        "name": "ET Markets Webinars",
        "category": "event",
        "description": "Free online webinar series on market trends, stock analysis, and investment strategies. Regular sessions featuring fund managers, analysts, and ET editors.",
        "url": "https://economictimes.indiatimes.com/markets",
        "price": "Free",
        "target_personas": ["first_time_investor", "seasoned_trader", "wealth_builder"],
        "trigger_signals": ["wants_to_learn", "event_interest", "live_expert_access"],
        "objection_handlers": {
            "no_time": "Sessions are 30-45 minutes and recordings are available. Attend live for Q&A or watch later.",
        },
        "upsell_from": ["et_markets_beginner_guide"],
        "cross_sell_to": ["et_prime", "et_masterclass_investing"],
        "value_proposition": "Free live sessions with market experts — learn, ask questions, and stay ahead.",
        "onboarding_steps": [
            "Browse upcoming webinar topics",
            "Register for one that interests you",
            "Attend live and ask a question in the Q&A",
            "Watch replays any time from the archive",
        ],
    },
    {
        "id": "et_credit_cards",
        "name": "ET Financial Services — Credit Cards",
        "category": "financial_service",
        "description": "Compare and apply for the best credit cards from HDFC, SBI, Axis, and more through ET's trusted partner network. Find cards matching your spending patterns and rewards preferences.",
        "url": "https://economictimes.indiatimes.com/wealth/spend/credit-cards",
        "price": "Varies by card",
        "target_personas": ["wealth_builder", "first_time_investor"],
        "trigger_signals": ["credit_card_interest", "rewards_optimization", "first_credit_card"],
        "objection_handlers": {
            "already_have_card": "Your spending might have changed since you last compared. Newer cards offer 2-5x better rewards for travel, dining, and online spending.",
        },
        "upsell_from": [],
        "cross_sell_to": ["et_prime", "et_wealth"],
        "value_proposition": "Find the credit card that rewards YOUR spending habits — compare India's top cards.",
        "onboarding_steps": [
            "Use the card comparison tool to see top picks for your profile",
            "Check eligibility without impacting your credit score",
            "Apply online in 5 minutes",
            "Activate and start earning rewards from day one",
        ],
    },
    {
        "id": "et_home_loan_partner",
        "name": "ET Financial Services — Home Loans",
        "category": "financial_service",
        "description": "Compare home loan rates from SBI, HDFC, ICICI, and 15+ lenders. Get pre-approved rates, EMI calculator, and expert advice on choosing the right loan through ET's partner network.",
        "url": "https://economictimes.indiatimes.com/wealth/borrow",
        "price": "Varies (current rates: 8.25%–9.5%)",
        "target_personas": ["family_builder", "wealth_builder"],
        "trigger_signals": ["home_purchase_intent", "home_loan_search", "property_interest", "emi_calculation"],
        "objection_handlers": {
            "already_have_loan": "Refinancing can save you lakhs — even a 0.25% rate drop on a ₹50L loan saves ₹2.5L over the tenure.",
            "not_ready_yet": "Getting pre-approved costs nothing and locks your eligibility. When you find the right property, you can move fast.",
        },
        "upsell_from": [],
        "cross_sell_to": ["et_prime_realty_content", "et_term_insurance_partner", "et_wealth"],
        "value_proposition": "Compare India's best home loan rates side-by-side — find the loan that fits your budget.",
        "onboarding_steps": [
            "Use the EMI calculator to determine your budget",
            "Compare rates from 15+ lenders",
            "Get pre-approved from top 2-3 lenders",
            "Read ET's home buying guides for your city",
        ],
    },
    {
        "id": "et_health_insurance_partner",
        "name": "ET Financial Services — Health Insurance",
        "category": "financial_service",
        "description": "Compare health insurance plans from top insurers. Find the right coverage for you and your family with ET's unbiased comparison tools and expert reviews.",
        "url": "https://economictimes.indiatimes.com/wealth/insure",
        "price": "Varies by plan",
        "target_personas": ["family_builder", "wealth_builder", "first_time_investor"],
        "trigger_signals": ["insurance_gap", "health_insurance_query", "family_protection"],
        "objection_handlers": {
            "already_have_employer_cover": "Employer cover ends when you leave. Personal cover is portable and premiums are lower when you're younger.",
            "too_expensive": "A ₹5L family floater costs about ₹500/month. One hospital visit without insurance can cost 10x that.",
        },
        "upsell_from": [],
        "cross_sell_to": ["et_term_insurance_partner", "et_wealth"],
        "value_proposition": "Protect your family's health and wealth — compare India's best health insurance plans.",
        "onboarding_steps": [
            "Use the coverage calculator to find your ideal sum assured",
            "Compare plans from 5+ top insurers",
            "Read ET's expert reviews on claim settlement ratio",
            "Buy online and get instant policy issuance",
        ],
    },
    {
        "id": "et_term_insurance_partner",
        "name": "ET Financial Services — Term Life Insurance",
        "category": "financial_service",
        "description": "Compare term insurance plans for maximum life cover at lowest premiums. Trusted comparison from ET featuring HDFC Life, ICICI Pru, Max Life, and more.",
        "url": "https://economictimes.indiatimes.com/wealth/insure",
        "price": "From ₹500/month for ₹1Cr cover",
        "target_personas": ["family_builder", "wealth_builder"],
        "trigger_signals": ["insurance_gap", "family_protection", "life_insurance_query", "new_parent"],
        "objection_handlers": {
            "have_lic_policy": "LIC endowment plans offer low cover at high premiums. A ₹1Cr term plan costs a fraction and provides 5-10x more cover.",
        },
        "upsell_from": [],
        "cross_sell_to": ["et_health_insurance_partner", "et_wealth"],
        "value_proposition": "₹1 Crore life cover from ₹500/month — secure your family's future.",
        "onboarding_steps": [
            "Calculate how much cover you need (10-15x annual income)",
            "Compare premiums across top insurers",
            "Complete a quick health declaration",
            "Buy online — no medical test needed for most plans under age 35",
        ],
    },
    {
        "id": "et_wealth_management",
        "name": "ET Financial Services — Wealth Management",
        "category": "financial_service",
        "description": "Connect with SEBI-registered wealth managers for portfolios of ₹50L+. Personalised advice, PMS, AIF access, and estate planning through ET's curated advisory network.",
        "url": "https://economictimes.indiatimes.com/wealth/invest",
        "price": "Customised (₹50L+ portfolio)",
        "target_personas": ["seasoned_trader", "wealth_builder", "nri"],
        "trigger_signals": ["high_net_worth", "pms_interest", "estate_planning", "large_portfolio"],
        "objection_handlers": {
            "already_have_advisor": "A second opinion never hurts — especially for portfolios above ₹50L. Different advisors bring different expertise.",
        },
        "upsell_from": ["et_prime", "et_wealth_summit"],
        "cross_sell_to": ["et_prime", "et_wealth_summit"],
        "value_proposition": "Personalised wealth management from SEBI-registered advisors for serious investors.",
        "onboarding_steps": [
            "Schedule a free portfolio review call",
            "Share your financial goals and risk profile",
            "Receive a customised investment plan",
            "Start with the advisor who best fits your needs",
        ],
    },
    {
        "id": "et_markets_beginner_guide",
        "name": "ET Markets Beginner's Guide",
        "category": "content",
        "description": "Free comprehensive guide to stock markets for complete beginners. Covers what are stocks, how markets work, how to start investing, and common mistakes to avoid.",
        "url": "https://economictimes.indiatimes.com/markets",
        "price": "Free",
        "target_personas": ["first_time_investor"],
        "trigger_signals": ["beginner_questions", "what_are_stocks", "how_to_invest", "market_basics"],
        "objection_handlers": {},
        "upsell_from": [],
        "cross_sell_to": ["et_money", "et_masterclass_investing", "et_markets_app"],
        "value_proposition": "Learn market basics in plain English — your first step to becoming an informed investor.",
        "onboarding_steps": [
            "Start with 'What are stocks?' — a 5-minute read",
            "Understand SIPs vs lump sum investing",
            "Learn the 5 mistakes every beginner makes",
            "Take the beginner quiz to test your knowledge",
        ],
    },
    {
        "id": "et_sip_calculator",
        "name": "ET SIP Calculator & Tools",
        "category": "content",
        "description": "Free interactive calculators for SIP returns, lump sum growth, retirement corpus, EMI, tax saving (80C), and more. Instant results with visual charts.",
        "url": "https://economictimes.indiatimes.com/wealth/calculators/sip-calculator",
        "price": "Free",
        "target_personas": ["first_time_investor", "wealth_builder"],
        "trigger_signals": ["sip_interest", "how_much_to_invest", "retirement_calculation", "emi_calculation"],
        "objection_handlers": {},
        "upsell_from": [],
        "cross_sell_to": ["et_money", "et_masterclass_investing"],
        "value_proposition": "See exactly how your money can grow — free calculators for every financial goal.",
        "onboarding_steps": [
            "Try the SIP calculator with your budget",
            "See the power of compounding over 10, 20, 30 years",
            "Calculate your retirement corpus need",
            "Share results and start your first SIP on ET Money",
        ],
    },
    {
        "id": "et_prime_realty_content",
        "name": "ET Prime — Real Estate Coverage",
        "category": "subscription",
        "description": "ET Prime's deep-dive real estate analysis: city-wise price trends, builder ratings, regulatory updates (RERA), and expert home-buying guides. Part of ET Prime subscription.",
        "url": "https://economictimes.indiatimes.com/wealth/real-estate",
        "price": "Part of ET Prime (₹999/year)",
        "target_personas": ["family_builder", "wealth_builder"],
        "trigger_signals": ["home_purchase_intent", "property_interest", "real_estate_analysis"],
        "objection_handlers": {
            "just_want_realty_content": "ET Prime gives you full access including markets, economy, and realty — all for ₹999/year.",
        },
        "upsell_from": [],
        "cross_sell_to": ["et_home_loan_partner", "et_prime"],
        "value_proposition": "Expert real estate analysis to help you make the biggest purchase of your life with confidence.",
        "onboarding_steps": [
            "Read the latest property market outlook for your city",
            "Use the home affordability calculator",
            "Check builder reputation and RERA compliance guides",
            "Subscribe to the weekly realty digest",
        ],
    },
    {
        "id": "et_intelligent_investing",
        "name": "ET Intelligent Investing",
        "category": "subscription",
        "description": "Part of ET Prime — deep-dive stock research series including Sectoral Deep Dives, Macro Economics analysis, Finding Red Flags in companies, and Warren Buffett investing lessons. Features the popular Multibagger or Bankrupt series.",
        "url": "https://economictimes.indiatimes.com/intelligent-investing",
        "price": "Part of ET Prime (₹999/year)",
        "target_personas": ["seasoned_trader", "wealth_builder"],
        "trigger_signals": ["stock_research", "fundamental_analysis", "red_flags", "deep_dive", "value_investing"],
        "objection_handlers": {
            "already_do_own_research": "Intelligent Investing adds a forensic lens — the Red Flags series alone has flagged companies months before trouble became public.",
        },
        "upsell_from": ["et_markets_app", "et_markets_beginner_guide"],
        "cross_sell_to": ["et_prime", "et_masterclass_investing"],
        "value_proposition": "Forensic-level stock research — spot red flags and multibaggers before the crowd.",
        "onboarding_steps": [
            "Read the latest Multibagger or Bankrupt deep-dive",
            "Explore the Finding Red Flags series",
            "Subscribe to ET Prime for full access",
            "Follow the Warren Buffett investing lessons",
        ],
    },
    {
        "id": "et_stock_report_plus",
        "name": "ET Stock Report Plus",
        "category": "subscription",
        "description": "Comprehensive stock analysis tool within ET Prime — stock scores out of 10, earnings analysis, fundamental checks, relative valuation, risk assessment, price momentum, and analyst recommendations. Features High Upside, Top Score, and Upward Momentum screeners.",
        "url": "https://economictimes.indiatimes.com/markets/benefits/stockreportsplus",
        "price": "Part of ET Prime (₹999/year)",
        "target_personas": ["seasoned_trader", "wealth_builder"],
        "trigger_signals": ["stock_screening", "stock_analysis", "stock_score", "analyst_recommendations"],
        "objection_handlers": {
            "use_other_screener": "Stock Report Plus combines 5 analytical dimensions (Earnings, Fundamentals, Valuation, Risk, Momentum) in one score — no other screener gives this holistic view.",
        },
        "upsell_from": ["et_markets_app"],
        "cross_sell_to": ["et_prime", "et_intelligent_investing"],
        "value_proposition": "Data-driven stock scores combining earnings, fundamentals, valuation, risk, and momentum in one number.",
        "onboarding_steps": [
            "Check the High Upside screener for stocks with strong analyst buy ratings",
            "Explore Top Score companies rated 10/10",
            "Use Upward Momentum screener for trending stocks",
            "Subscribe to ET Prime for full stock report access",
        ],
    },
    {
        "id": "et_epaper",
        "name": "ET ePaper",
        "category": "content",
        "description": "Read The Economic Times newspaper in its original print layout — digitally. Daily edition with city-specific content for Delhi, Mumbai, Kolkata, Bangalore, Chennai, and more.",
        "url": "https://epaper.indiatimes.com/timesepaper/publication-the-economic-times,city-delhi.cms",
        "price": "Free with ET Prime / ₹499 standalone",
        "target_personas": ["seasoned_trader", "wealth_builder", "nri"],
        "trigger_signals": ["newspaper_reader", "print_edition", "daily_news", "traditional_reader"],
        "objection_handlers": {
            "prefer_digital": "The ePaper gives you the curated editorial judgment of the print edition — every story placed by importance, plus full digital search and archive.",
        },
        "upsell_from": [],
        "cross_sell_to": ["et_prime"],
        "value_proposition": "Your daily Economic Times newspaper — anywhere, anytime, in the familiar print layout.",
        "onboarding_steps": [
            "Visit the ePaper portal and browse today's edition",
            "Select your preferred city edition",
            "Subscribe for daily delivery to your inbox",
            "Also check the weekly Wealth Edition ePaper",
        ],
    },
    {
        "id": "et_income_tax_calculator",
        "name": "ET Income Tax Calculator",
        "category": "content",
        "description": "Free comprehensive income tax calculator — compare old vs new tax regime, calculate tax liability, find optimal regime, and plan deductions under Section 80C, 80D, HRA, and more.",
        "url": "https://economictimes.indiatimes.com/wealth/calculators/income-tax-calculator",
        "price": "Free",
        "target_personas": ["first_time_investor", "wealth_builder", "family_builder"],
        "trigger_signals": ["tax_planning", "income_tax_query", "tax_saving", "itr_filing", "old_vs_new_regime"],
        "objection_handlers": {},
        "upsell_from": [],
        "cross_sell_to": ["et_wealth", "et_prime"],
        "value_proposition": "Calculate your exact tax liability in minutes — compare old vs new regime and optimize your savings.",
        "onboarding_steps": [
            "Enter your salary and income details",
            "Compare tax under old vs new regime",
            "Identify deductions you may be missing",
            "Explore linked tax-saving investment options on ET Money",
        ],
    },
    {
        "id": "et_mutual_funds",
        "name": "ET Mutual Funds",
        "category": "content",
        "description": "Comprehensive mutual fund research section — fund factsheets, NAV data, returns comparison, category-wise rankings (Large Cap, Mid Cap, Small Cap, Flexi Cap, ELSS, etc.), and expert fund picks. Powered by ET Money integration.",
        "url": "https://economictimes.indiatimes.com/mutual-funds",
        "price": "Free",
        "target_personas": ["first_time_investor", "wealth_builder"],
        "trigger_signals": ["mutual_fund_interest", "fund_comparison", "best_funds", "nav_check", "elss_funds"],
        "objection_handlers": {
            "already_invested": "Use ET Mutual Funds section to compare your holdings against top performers and discover better alternatives.",
        },
        "upsell_from": ["et_sip_calculator"],
        "cross_sell_to": ["et_money", "et_prime"],
        "value_proposition": "Research, compare, and pick the best mutual funds with data-driven insights.",
        "onboarding_steps": [
            "Browse top-performing funds by category",
            "Compare fund returns across 1Y, 3Y, 5Y periods",
            "Read expert fund picks and analysis",
            "Start investing via ET Money for zero-commission direct plans",
        ],
    },
]


# ---------------------------------------------------------------------------
# Content freshness simulation (recent additions)
# ---------------------------------------------------------------------------
NEW_CONTENT_SINCE: list[dict] = [
    {"days_ago": 7, "title": "New Masterclass: Smallcap Investing Strategies 2026", "type": "learning", "product_id": "et_masterclass_investing"},
    {"days_ago": 14, "title": "ET Money launches Portfolio X-Ray — free portfolio health check", "type": "feature", "product_id": "et_money"},
    {"days_ago": 21, "title": "ET Markets adds AI-powered stock screener", "type": "feature", "product_id": "et_markets_app"},
    {"days_ago": 30, "title": "ET Prime price lock: returning subscribers get ₹799/year (locked for life)", "type": "offer", "product_id": "et_prime"},
    {"days_ago": 35, "title": "Live webinar series: Navigating Mid & Smallcap Volatility (4-part)", "type": "event", "product_id": "et_markets_webinars"},
    {"days_ago": 45, "title": "ET Wealth Summit 2026 announced — Early bird registration open", "type": "event", "product_id": "et_wealth_summit"},
    {"days_ago": 60, "title": "New ET Prime exclusive: Monthly Smallcap Deep-Dive report", "type": "content", "product_id": "et_prime"},
    {"days_ago": 75, "title": "ET Money introduces NRI investment gateway", "type": "feature", "product_id": "et_money"},
    {"days_ago": 80, "title": "Tax Saving Season special: ELSS fund comparison tool on ET Money", "type": "feature", "product_id": "et_money"},
    {"days_ago": 90, "title": "ET Markets Pro launched — advanced charting and technical analysis tools", "type": "feature", "product_id": "et_markets_app"},
]


# ---------------------------------------------------------------------------
# ETKnowledgeBase
# ---------------------------------------------------------------------------
class ETKnowledgeBase:
    """Product registry, semantic search, and recommendation scoring."""

    def __init__(self, use_chroma: bool = True):
        self.products = {p["id"]: p for p in ET_PRODUCTS}
        self._chroma_collection = None

        if use_chroma:
            self._init_chroma()

    # ------------------------------------------------------------------
    # ChromaDB setup
    # ------------------------------------------------------------------
    def _init_chroma(self):
        try:
            import chromadb
            from chromadb.config import Settings

            persist_dir = CONFIG.get("vector_db", {}).get("persist_path", "./data/chroma")
            os.makedirs(persist_dir, exist_ok=True)

            client = chromadb.PersistentClient(path=persist_dir)
            collection_name = CONFIG.get("vector_db", {}).get("collection_name", "et_products")

            self._chroma_collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            # Upsert product embeddings if collection is empty or stale
            if self._chroma_collection.count() < len(ET_PRODUCTS):
                docs, ids, metadatas = [], [], []
                for p in ET_PRODUCTS:
                    text = f"{p['name']}. {p['description']} Target: {', '.join(p['target_personas'])}. Signals: {', '.join(p['trigger_signals'])}. {p['value_proposition']}"
                    docs.append(text)
                    ids.append(p["id"])
                    metadatas.append({"name": p["name"], "category": p["category"], "price": p.get("price", "")})

                self._chroma_collection.upsert(documents=docs, ids=ids, metadatas=metadatas)

        except Exception as e:
            print(f"[KnowledgeBase] ChromaDB init skipped: {e}")
            self._chroma_collection = None

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------
    def semantic_search(self, query: str, n: int = 3) -> list[dict]:
        if self._chroma_collection is None:
            return self._fallback_search(query, n)

        try:
            results = self._chroma_collection.query(query_texts=[query], n_results=n)
            product_ids = results["ids"][0] if results["ids"] else []
            distances = results["distances"][0] if results.get("distances") else [0.5] * len(product_ids)

            matched = []
            for pid, dist in zip(product_ids, distances):
                product = self.products.get(pid)
                if product:
                    matched.append({**product, "relevance_score": round(1.0 - dist, 3)})
            return matched
        except Exception:
            return self._fallback_search(query, n)

    def _fallback_search(self, query: str, n: int) -> list[dict]:
        """Keyword-based fallback when ChromaDB is unavailable."""
        query_lower = query.lower()
        scored: list[tuple[float, dict]] = []
        for p in ET_PRODUCTS:
            text = f"{p['name']} {p['description']} {' '.join(p['trigger_signals'])}".lower()
            words = query_lower.split()
            hits = sum(1 for w in words if w in text)
            score = hits / max(len(words), 1)
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        return [{**p, "relevance_score": round(s, 3)} for s, p in scored[:n]]

    # ------------------------------------------------------------------
    # Persona-based retrieval
    # ------------------------------------------------------------------
    def get_products_for_persona(self, persona_tag: str) -> list[dict]:
        matched = []
        for p in ET_PRODUCTS:
            if persona_tag in p["target_personas"]:
                matched.append({**p, "relevance_score": 0.8})
            elif persona_tag.replace("_", " ") in " ".join(p["target_personas"]):
                matched.append({**p, "relevance_score": 0.5})
        matched.sort(key=lambda x: -x["relevance_score"])
        return matched

    # ------------------------------------------------------------------
    # Signal-based retrieval
    # ------------------------------------------------------------------
    def get_products_for_signals(self, signals: list[str]) -> list[dict]:
        matched = []
        signals_set = set(s.lower() for s in signals)
        for p in ET_PRODUCTS:
            trigger_set = set(t.lower() for t in p["trigger_signals"])
            overlap = signals_set & trigger_set
            if overlap:
                score = len(overlap) / max(len(trigger_set), 1)
                matched.append({**p, "relevance_score": round(score, 3), "matched_signals": list(overlap)})
        matched.sort(key=lambda x: -x["relevance_score"])
        return matched

    # ------------------------------------------------------------------
    # Content freshness
    # ------------------------------------------------------------------
    def get_new_since(self, days_ago: int) -> list[dict]:
        return [item for item in NEW_CONTENT_SINCE if item["days_ago"] <= days_ago]

    # ------------------------------------------------------------------
    # Recommendation scorer
    # ------------------------------------------------------------------
    def score_recommendation(self, product_id: str, user_profile: dict) -> dict:
        product = self.products.get(product_id)
        if not product:
            return {"score": 0.0, "reasoning": "Product not found"}

        persona = user_profile.get("persona_tag", "first_time_investor")
        signals = user_profile.get("detected_life_events", []) + [user_profile.get("primary_financial_goal", "")]
        completeness = user_profile.get("profile_completeness", 0.0)

        # Persona match (40%)
        persona_score = 1.0 if persona in product["target_personas"] else 0.2
        reasons = []
        if persona_score > 0.5:
            reasons.append(f"matches your '{persona.replace('_', ' ')}' profile")

        # Signal match (30%)
        trigger_set = set(t.lower() for t in product["trigger_signals"])
        signal_set = set(s.lower() for s in signals if s)
        signal_overlap = trigger_set & signal_set
        signal_score = len(signal_overlap) / max(len(trigger_set), 1) if trigger_set else 0.0
        if signal_overlap:
            reasons.append(f"aligns with your interest in {', '.join(signal_overlap)}")

        # Profile completeness (15%)
        completeness_score = completeness

        # Past interactions (15%) — simulated
        et_used = user_profile.get("et_products_used", [])
        interaction_score = 0.5  # baseline
        if product_id in [p.replace("et_", "et_") for p in et_used]:
            interaction_score = 0.8
            reasons.append("you have used this before")
        elif any(up in product.get("upsell_from", []) for up in et_used):
            interaction_score = 0.9
            reasons.append("builds on products you already use")

        final_score = (
            0.40 * persona_score
            + 0.30 * signal_score
            + 0.15 * completeness_score
            + 0.15 * interaction_score
        )

        reasoning = f"{product['name']}: " + ("; ".join(reasons) if reasons else "general recommendation based on profile")

        return {
            "product_id": product_id,
            "score": round(final_score, 3),
            "reasoning": reasoning,
            "components": {
                "persona_match": round(persona_score, 2),
                "signal_match": round(signal_score, 2),
                "completeness": round(completeness_score, 2),
                "interaction": round(interaction_score, 2),
            },
        }

    def get_ranked_recommendations(self, user_profile: dict, limit: int = 5) -> list[dict]:
        scored = []
        for pid in self.products:
            result = self.score_recommendation(pid, user_profile)
            scored.append(result)
        scored.sort(key=lambda x: -x["score"])
        return scored[:limit]

    def get_product(self, product_id: str) -> dict | None:
        return self.products.get(product_id)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    kb = ETKnowledgeBase(use_chroma=True)
    print(f"Products loaded: {len(kb.products)}")

    # Semantic search
    results = kb.semantic_search("I want to start investing in mutual funds", n=3)
    print("\n--- Semantic Search: 'start investing in mutual funds' ---")
    for r in results:
        print(f"  {r['name']} (score: {r['relevance_score']})")

    # Persona search
    results = kb.get_products_for_persona("first_time_investor")
    print("\n--- Products for first_time_investor ---")
    for r in results[:5]:
        print(f"  {r['name']}")

    # Signal search
    results = kb.get_products_for_signals(["home_purchase_intent", "home_loan_search"])
    print("\n--- Products for home_purchase signals ---")
    for r in results:
        print(f"  {r['name']} (score: {r['relevance_score']}, signals: {r.get('matched_signals', [])})")

    # New content
    new = kb.get_new_since(90)
    print(f"\n--- New content in last 90 days: {len(new)} items ---")
    for item in new:
        print(f"  [{item['days_ago']}d ago] {item['title']}")

    # Scorer
    profile = {
        "persona_tag": "first_time_investor",
        "detected_life_events": [],
        "primary_financial_goal": "wealth_creation",
        "profile_completeness": 0.6,
        "et_products_used": [],
    }
    ranked = kb.get_ranked_recommendations(profile, limit=5)
    print("\n--- Ranked recommendations for beginner ---")
    for r in ranked:
        print(f"  {r['product_id']}: {r['score']} — {r['reasoning']}")

    print("\nKnowledge base OK")
