"""
ET AI Concierge — Memory Store & User Profile Schema
In-session + persistent user memory with progressive profiling support.
"""

import json
import uuid
import re
from datetime import datetime, timezone
from typing import Optional

import sqlalchemy as sa
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Float
from sqlalchemy.orm import declarative_base, sessionmaker

import yaml

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

CONFIG = _load_config()

# ---------------------------------------------------------------------------
# TypedDict-style schemas (plain dicts with helper constructors)
# ---------------------------------------------------------------------------

def empty_user_profile() -> dict:
    return {
        # Demographics
        "age_band": "",
        "life_stage": "",
        "income_band": "",
        "city_tier": "",

        # Financial Profile
        "investment_experience": "",
        "current_investments": [],
        "primary_financial_goal": "",
        "risk_appetite": "",
        "monthly_investable_surplus": "",

        # ET Product Usage
        "et_products_used": [],
        "et_subscription_status": "none",
        "days_since_last_visit": 0,
        "preferred_content_types": [],

        # Inferred Signals
        "detected_life_events": [],
        "urgency_level": "browsing",
        "persona_tag": "",

        # Metadata
        "profile_completeness": 0.0,
        "confidence_scores": {},
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def empty_session_memory() -> dict:
    return {
        "raw_turns": [],
        "user_profile": empty_user_profile(),
        "journey_state": {
            "phase": "discover",
            "completed_steps": [],
            "current_step": None,
            "upcoming_steps": [],
            "persona_journey": "",
        },
        "active_signals": [],
        "last_recommendation": None,
        "conversation_intent_stack": [],
        "cross_sell_injections_count": 0,
        "last_cross_sell_turn": -10,
        "disclaimer_shown": False,
    }


# ---------------------------------------------------------------------------
# Profile completeness weights
# ---------------------------------------------------------------------------
FIELD_WEIGHTS = {
    "investment_experience": 0.25,
    "primary_financial_goal": 0.20,
    "life_stage": 0.15,
    "et_products_used": 0.15,
    "risk_appetite": 0.15,
    "income_band": 0.10,
}

# ---------------------------------------------------------------------------
# Keyword-based profile inference rules  (fast, no LLM needed)
# ---------------------------------------------------------------------------
INFERENCE_RULES: list[dict] = [
    # Age / life-stage
    {"patterns": [r"\b(college|student|campus)\b"], "field": "life_stage", "value": "student", "confidence": 0.7},
    {"patterns": [r"\b(first job|just started working|fresher|early career)\b"], "field": "life_stage", "value": "early_career", "confidence": 0.75},
    {"patterns": [r"\b(kid|child|baby|school fee|child education)\b"], "field": "life_stage", "value": "family_builder", "confidence": 0.7},
    {"patterns": [r"\b(retire|retirement|pension|NPS|superannuation)\b"], "field": "life_stage", "value": "pre_retirement", "confidence": 0.65},

    # Age band hints
    {"patterns": [r"\b(i(?:'| a)?m\s*2[0-5])\b", r"\b(22|23|24|25)\s*(?:year|yr)"], "field": "age_band", "value": "18-25", "confidence": 0.8},
    {"patterns": [r"\b(i(?:'| a)?m\s*2[6-9])\b", r"\b(i(?:'| a)?m\s*3[0-5])\b", r"\b(28|29|30|32|34)\s*(?:year|yr)"], "field": "age_band", "value": "26-35", "confidence": 0.8},
    {"patterns": [r"\b(i(?:'| a)?m\s*[3][6-9])\b", r"\b(i(?:'| a)?m\s*4[0-5])\b"], "field": "age_band", "value": "36-45", "confidence": 0.8},

    # Investment experience
    {"patterns": [r"\b(never invested|no experience|don'?t know|new to invest|first time)\b"], "field": "investment_experience", "value": "none", "confidence": 0.8},
    {"patterns": [r"\b(beginner|just started|learning|basics)\b"], "field": "investment_experience", "value": "beginner", "confidence": 0.75},
    {"patterns": [r"\b(sip|mutual fund|mf|index fund)\b"], "field": "investment_experience", "value": "beginner", "confidence": 0.5},
    {"patterns": [r"\b(option|derivative|futures|f&o|swing trad|intraday|technical analysis)\b"], "field": "investment_experience", "value": "advanced", "confidence": 0.8},
    {"patterns": [r"\b(portfolio|diversif|rebalanc|asset allocat|smallcap|midcap|largecap|nifty|sensex)\b"], "field": "investment_experience", "value": "intermediate", "confidence": 0.6},

    # Current investments
    {"patterns": [r"\b(savings? account|bank account|FD|fixed deposit)\b"], "field": "current_investments", "value": "savings_account", "confidence": 0.8},
    {"patterns": [r"\bSIP\b", r"\bmutual fund\b"], "field": "current_investments", "value": "sip", "confidence": 0.8},
    {"patterns": [r"\b(stock|share|equity|nifty|sensex)\b"], "field": "current_investments", "value": "stocks", "confidence": 0.7},
    {"patterns": [r"\b(crypto|bitcoin|ethereum)\b"], "field": "current_investments", "value": "crypto", "confidence": 0.8},
    {"patterns": [r"\b(real estate|property|flat|house|apartment)\b"], "field": "current_investments", "value": "real_estate", "confidence": 0.6},

    # Goals
    {"patterns": [r"\b(tax sav|80C|ELSS|tax benefit)\b"], "field": "primary_financial_goal", "value": "tax_saving", "confidence": 0.8},
    {"patterns": [r"\b(retire|retirement|corpus|financial freedom)\b"], "field": "primary_financial_goal", "value": "retirement", "confidence": 0.75},
    {"patterns": [r"\b(child education|kid.{0,15}(school|college))\b"], "field": "primary_financial_goal", "value": "child_education", "confidence": 0.8},
    {"patterns": [r"\b(home|house|flat|property|buy|purchase|down ?payment)\b"], "field": "primary_financial_goal", "value": "home_purchase", "confidence": 0.5},
    {"patterns": [r"\b(emergency|rainy day|safety net)\b"], "field": "primary_financial_goal", "value": "emergency_fund", "confidence": 0.7},
    {"patterns": [r"\b(wealth|grow|long.?term|compound)\b"], "field": "primary_financial_goal", "value": "wealth_creation", "confidence": 0.45},

    # Risk appetite
    {"patterns": [r"\b(safe|low risk|conservative|guaranteed|no risk)\b"], "field": "risk_appetite", "value": "conservative", "confidence": 0.7},
    {"patterns": [r"\b(moderate|balanced|mix)\b"], "field": "risk_appetite", "value": "moderate", "confidence": 0.6},
    {"patterns": [r"\b(aggressive|high risk|high return|risk taker|growth)\b"], "field": "risk_appetite", "value": "aggressive", "confidence": 0.7},

    # Income hints
    {"patterns": [r"\b(IT|software|tech|developer|engineer)\b"], "field": "income_band", "value": "10-25L", "confidence": 0.4},
    {"patterns": [r"\b(CXO|director|VP|partner|founder|business owner)\b"], "field": "income_band", "value": "50L+", "confidence": 0.5},

    # ET product usage
    {"patterns": [r"\b(ET Prime|prime subscription)\b"], "field": "et_products_used", "value": "et_prime", "confidence": 0.9},
    {"patterns": [r"\b(ET Money)\b"], "field": "et_products_used", "value": "et_money", "confidence": 0.9},
    {"patterns": [r"\b(ET Markets)\b"], "field": "et_products_used", "value": "et_markets", "confidence": 0.9},

    # Life events
    {"patterns": [r"\b(home loan|housing loan|property|EMI|down ?payment)\b"], "field": "detected_life_events", "value": "home_loan_interest", "confidence": 0.7},
    {"patterns": [r"\b(new job|switching|joining|notice period|offer letter|salary hike)\b"], "field": "detected_life_events", "value": "job_change", "confidence": 0.7},
    {"patterns": [r"\b(married|marriage|wedding)\b"], "field": "detected_life_events", "value": "marriage", "confidence": 0.8},
    {"patterns": [r"\b(baby|child|kid born|expecting)\b"], "field": "detected_life_events", "value": "child_birth", "confidence": 0.8},

    # Persona tag
    {"patterns": [r"\b(lapsed|expired|used to have|subscription ended|came back)\b"], "field": "persona_tag", "value": "lapsed_subscriber", "confidence": 0.8},
    {"patterns": [r"\b(NRI|abroad|US|UK|Dubai|expat)\b"], "field": "persona_tag", "value": "nri", "confidence": 0.75},
]


# ---------------------------------------------------------------------------
# SQLAlchemy models
# ---------------------------------------------------------------------------
Base = declarative_base()


class UserRow(Base):
    __tablename__ = "users"
    user_id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    profile_json = Column(Text, default="{}")
    journey_stage = Column(String, default="discover")


class SessionRow(Base):
    __tablename__ = "sessions"
    session_id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime, nullable=True)
    summary_json = Column(Text, default="{}")


class EventRow(Base):
    __tablename__ = "events"
    event_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, index=True)
    event_type = Column(String)
    event_data = Column(Text, default="{}")
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ProductInteractionRow(Base):
    __tablename__ = "product_interactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    product_id = Column(String)
    action = Column(String)  # viewed / clicked / subscribed / lapsed
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AuditRow(Base):
    __tablename__ = "audit_log"
    audit_id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    session_id = Column(String, index=True)
    user_id = Column(String, index=True)
    turn_number = Column(Integer)
    user_input = Column(Text)
    detected_intents = Column(Text, default="[]")
    agents_called = Column(Text, default="[]")
    profile_updates = Column(Text, default="{}")
    recommendations_made = Column(Text, default="[]")
    cross_sell_signals = Column(Text, default="[]")
    response_text = Column(Text)
    evaluator_score = Column(Float, default=0.0)
    guardrail_violations = Column(Text, default="[]")
    latency_ms = Column(Integer, default=0)
    llm_tokens_used = Column(Integer, default=0)
    model_used = Column(String, default="")
    fallback_triggered = Column(Integer, default=0)


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------
class MemoryStore:
    """Manages in-session state and persistent SQLite storage."""

    def __init__(self, db_path: str | None = None):
        db_path = db_path or CONFIG.get("database", {}).get("sqlite_path", "./data/et_concierge.db")
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # In-memory session cache: session_id -> session dict
        self._sessions: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def create_session(self, user_id: str, session_id: str | None = None) -> dict:
        session_id = session_id or str(uuid.uuid4())
        mem = empty_session_memory()
        mem["session_id"] = session_id
        mem["user_id"] = user_id
        self._sessions[session_id] = mem

        # Persist
        with self.Session() as db:
            db.merge(SessionRow(session_id=session_id, user_id=user_id))
            db.commit()

        return mem

    def get_session(self, session_id: str) -> dict | None:
        return self._sessions.get(session_id)

    # ------------------------------------------------------------------
    # Profile helpers
    # ------------------------------------------------------------------
    def update_profile(self, session_id: str, field: str, value, confidence: float = 0.5):
        mem = self._sessions.get(session_id)
        if mem is None:
            return
        profile = mem["user_profile"]
        existing_conf = profile["confidence_scores"].get(field, 0.0)

        # Only overwrite if new confidence is higher or field is empty
        current_val = profile.get(field)
        is_empty = (current_val in ("", [], 0, None, 0.0))
        if confidence >= existing_conf or is_empty:
            if isinstance(current_val, list) and not isinstance(value, list):
                if value not in current_val:
                    profile[field].append(value)
            else:
                profile[field] = value
            profile["confidence_scores"][field] = confidence

        profile["last_updated"] = datetime.now(timezone.utc).isoformat()
        profile["profile_completeness"] = self.calculate_profile_completeness(profile)

    def calculate_profile_completeness(self, profile: dict) -> float:
        score = 0.0
        for field, weight in FIELD_WEIGHTS.items():
            val = profile.get(field, "")
            if isinstance(val, list):
                filled = len(val) > 0
            elif isinstance(val, str):
                filled = val != ""
            else:
                filled = val is not None and val != 0
            if filled:
                conf = profile.get("confidence_scores", {}).get(field, 0.5)
                score += weight * min(conf / 0.7, 1.0)  # Normalise confidence contribution
        return round(min(score, 1.0), 2)

    def get_missing_high_value_fields(self, session_id: str) -> list[str]:
        mem = self._sessions.get(session_id)
        if mem is None:
            return []
        profile = mem["user_profile"]
        missing = []
        for field, weight in sorted(FIELD_WEIGHTS.items(), key=lambda x: -x[1]):
            val = profile.get(field, "")
            is_empty = val in ("", [], 0, None, 0.0)
            if is_empty:
                missing.append(field)
        return missing

    def get_profile_summary_for_agent(self, session_id: str, agent_type: str) -> dict:
        mem = self._sessions.get(session_id)
        if mem is None:
            return {}
        profile = mem["user_profile"]

        if agent_type == "profiler":
            return dict(profile)  # full access
        elif agent_type == "cross_sell":
            return {
                "detected_life_events": profile["detected_life_events"],
                "urgency_level": profile["urgency_level"],
                "persona_tag": profile["persona_tag"],
                "et_products_used": profile["et_products_used"],
                "primary_financial_goal": profile["primary_financial_goal"],
                "risk_appetite": profile["risk_appetite"],
            }
        elif agent_type == "recommender":
            return {k: v for k, v in profile.items() if k not in ("confidence_scores",)}
        else:
            return {k: v for k, v in profile.items() if k not in ("confidence_scores",)}

    # ------------------------------------------------------------------
    # Conversation Tracking
    # ------------------------------------------------------------------
    def add_turn(self, session_id: str, role: str, content: str,
                 agent_name: str = "", confidence_score: float = 1.0):
        mem = self._sessions.get(session_id)
        if mem is None:
            return
        mem["raw_turns"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_name": agent_name,
            "confidence_score": confidence_score,
        })

    def get_recent_turns(self, session_id: str, n: int = 6) -> list[dict]:
        mem = self._sessions.get(session_id)
        if mem is None:
            return []
        return mem["raw_turns"][-n:]

    # ------------------------------------------------------------------
    # Rule-based profile inference from free text
    # ------------------------------------------------------------------
    def infer_from_conversation(self, session_id: str, turn_text: str) -> dict:
        updates: dict[str, tuple] = {}  # field -> (value, confidence)
        lower = turn_text.lower()

        for rule in INFERENCE_RULES:
            for pat in rule["patterns"]:
                if re.search(pat, lower, re.IGNORECASE):
                    field = rule["field"]
                    value = rule["value"]
                    conf = rule["confidence"]
                    # Keep highest confidence match per field
                    if field not in updates or conf > updates[field][1]:
                        updates[field] = (value, conf)
                    break  # first matching pattern is enough

        for field, (value, conf) in updates.items():
            self.update_profile(session_id, field, value, conf)

        return {field: val for field, (val, _) in updates.items()}

    # ------------------------------------------------------------------
    # Persistent user store
    # ------------------------------------------------------------------
    def save_user_profile(self, user_id: str, profile: dict):
        with self.Session() as db:
            row = db.get(UserRow, user_id)
            if row is None:
                row = UserRow(user_id=user_id, profile_json=json.dumps(profile))
                db.add(row)
            else:
                row.profile_json = json.dumps(profile)
                row.last_seen = datetime.now(timezone.utc)
            db.commit()

    def load_user_history(self, user_id: str) -> dict | None:
        with self.Session() as db:
            row = db.get(UserRow, user_id)
            if row is None:
                return None
            return {
                "user_id": row.user_id,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "last_seen": row.last_seen.isoformat() if row.last_seen else None,
                "profile": json.loads(row.profile_json) if row.profile_json else {},
                "journey_stage": row.journey_stage,
            }

    def save_session_summary(self, session_id: str):
        mem = self._sessions.get(session_id)
        if mem is None:
            return
        summary = {
            "turns_count": len(mem["raw_turns"]),
            "profile_completeness": mem["user_profile"]["profile_completeness"],
            "journey_phase": mem["journey_state"]["phase"],
            "signals_detected": mem["active_signals"],
            "persona_tag": mem["user_profile"]["persona_tag"],
        }
        with self.Session() as db:
            row = db.get(SessionRow, session_id)
            if row:
                row.ended_at = datetime.now(timezone.utc)
                row.summary_json = json.dumps(summary)
            db.commit()

        # Also persist the user profile
        user_id = mem.get("user_id", "")
        if user_id:
            self.save_user_profile(user_id, mem["user_profile"])

    def log_product_interaction(self, user_id: str, product_id: str, action: str):
        with self.Session() as db:
            db.add(ProductInteractionRow(user_id=user_id, product_id=product_id, action=action))
            db.commit()

    def log_event(self, session_id: str, event_type: str, event_data: dict):
        with self.Session() as db:
            db.add(EventRow(
                session_id=session_id,
                event_type=event_type,
                event_data=json.dumps(event_data),
            ))
            db.commit()

    # ------------------------------------------------------------------
    # Persona inference
    # ------------------------------------------------------------------
    def infer_persona(self, session_id: str) -> str:
        mem = self._sessions.get(session_id)
        if mem is None:
            return "first_time_investor"
        profile = mem["user_profile"]

        if profile.get("persona_tag"):
            return profile["persona_tag"]

        exp = profile.get("investment_experience", "")
        sub_status = profile.get("et_subscription_status", "none")

        if sub_status == "lapsed":
            tag = "lapsed_subscriber"
        elif exp in ("advanced",):
            tag = "seasoned_trader"
        elif exp in ("intermediate",):
            tag = "wealth_builder"
        elif exp in ("none", "beginner", ""):
            tag = "first_time_investor"
        else:
            tag = "first_time_investor"

        self.update_profile(session_id, "persona_tag", tag, 0.6)
        return tag


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    store = MemoryStore(db_path="./data/et_concierge_test.db")
    mem = store.create_session("user_001", "sess_001")
    print("Session created:", mem["session_id"])

    # Simulate conversation
    store.add_turn("sess_001", "user", "Hi, I'm 28 and work in IT. Never invested before.")
    inferred = store.infer_from_conversation("sess_001", "Hi, I'm 28 and work in IT. Never invested before.")
    print("Inferred:", inferred)

    profile = store.get_session("sess_001")["user_profile"]
    print(f"Profile completeness: {profile['profile_completeness']}")
    print(f"Missing fields: {store.get_missing_high_value_fields('sess_001')}")
    print(f"Persona: {store.infer_persona('sess_001')}")

    store.save_session_summary("sess_001")
    history = store.load_user_history("user_001")
    print(f"Loaded history: {history}")
    print("Memory store OK")
