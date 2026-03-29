"""
ET AI Concierge — Audit Logger
Structured audit trail for explainability, compliance, and analytics.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import yaml

from memory_store import Base, AuditRow

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


class AuditLogger:
    """Audit trail for every agent interaction — SQLite-backed."""

    def __init__(self, db_path: str | None = None):
        import os
        db_path = db_path or CONFIG.get("database", {}).get("sqlite_path", "./data/et_concierge.db")
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def log_turn(self, entry: dict) -> str:
        """Write a structured audit entry to SQLite. Returns audit_id."""
        audit_id = entry.get("audit_id", str(uuid.uuid4()))

        row = AuditRow(
            audit_id=audit_id,
            timestamp=datetime.now(timezone.utc),
            session_id=entry.get("session_id", ""),
            user_id=entry.get("user_id", ""),
            turn_number=entry.get("turn_number", 0),
            user_input=entry.get("user_input", ""),
            detected_intents=json.dumps(entry.get("detected_intents", [])),
            agents_called=json.dumps(entry.get("agents_called", [])),
            profile_updates=json.dumps(entry.get("profile_updates", {})),
            recommendations_made=json.dumps(entry.get("recommendations_made", [])),
            cross_sell_signals=json.dumps(entry.get("cross_sell_signals", [])),
            response_text=entry.get("response_text", ""),
            evaluator_score=entry.get("evaluator_score", 0.0),
            guardrail_violations=json.dumps(entry.get("guardrail_violations", [])),
            latency_ms=entry.get("latency_ms", 0),
            llm_tokens_used=entry.get("llm_tokens_used", 0),
            model_used=entry.get("model_used", ""),
            fallback_triggered=1 if entry.get("fallback_triggered", False) else 0,
        )

        with self.Session() as db:
            db.add(row)
            db.commit()

        return audit_id

    def get_session_audit(self, session_id: str) -> list[dict]:
        """Get all audit entries for a session."""
        with self.Session() as db:
            rows = db.query(AuditRow).filter(
                AuditRow.session_id == session_id
            ).order_by(AuditRow.turn_number).all()

            return [self._row_to_dict(r) for r in rows]

    def get_session_summary(self, session_id: str) -> dict:
        """Aggregate metrics for a session."""
        entries = self.get_session_audit(session_id)
        if not entries:
            return {"session_id": session_id, "turns": 0}

        total_tokens = sum(e.get("llm_tokens_used", 0) for e in entries)
        total_latency = sum(e.get("latency_ms", 0) for e in entries)
        scores = [e.get("evaluator_score", 0) for e in entries if e.get("evaluator_score", 0) > 0]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        all_intents = []
        for e in entries:
            all_intents.extend(e.get("detected_intents", []))

        all_recs = []
        for e in entries:
            all_recs.extend(e.get("recommendations_made", []))

        all_violations = []
        for e in entries:
            all_violations.extend(e.get("guardrail_violations", []))

        fallback_count = sum(1 for e in entries if e.get("fallback_triggered"))

        return {
            "session_id": session_id,
            "turns": len(entries),
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency,
            "avg_evaluator_score": round(avg_score, 3),
            "unique_intents": list(set(all_intents)),
            "recommendations_count": len(all_recs),
            "guardrail_violations": all_violations,
            "fallback_count": fallback_count,
            "agents_used": list(set(
                agent for e in entries for agent in e.get("agents_called", [])
            )),
        }

    def export_for_analysis(self, start_date: str | None = None, end_date: str | None = None) -> list[dict]:
        """Export audit entries for business analytics."""
        with self.Session() as db:
            query = db.query(AuditRow)
            if start_date:
                query = query.filter(AuditRow.timestamp >= start_date)
            if end_date:
                query = query.filter(AuditRow.timestamp <= end_date)

            rows = query.order_by(AuditRow.timestamp).all()
            return [self._row_to_dict(r) for r in rows]

    def get_metrics(self) -> dict:
        """Get aggregate metrics for the dashboard."""
        with self.Session() as db:
            today = datetime.now(timezone.utc).date().isoformat()

            # Total sessions today
            sessions_today = db.execute(
                text("SELECT COUNT(DISTINCT session_id) FROM audit_log WHERE date(timestamp) = :today"),
                {"today": today},
            ).scalar() or 0

            # Average evaluator score
            avg_score = db.execute(
                text("SELECT AVG(evaluator_score) FROM audit_log WHERE evaluator_score > 0"),
            ).scalar() or 0.0

            # Top recommended products
            all_recs = db.execute(
                text("SELECT recommendations_made FROM audit_log WHERE recommendations_made != '[]'"),
            ).fetchall()

            product_counts: dict[str, int] = {}
            for row in all_recs:
                recs = json.loads(row[0]) if row[0] else []
                for rec in recs:
                    pid = rec if isinstance(rec, str) else rec.get("product_id", "")
                    if pid:
                        product_counts[pid] = product_counts.get(pid, 0) + 1

            top_products = sorted(product_counts.items(), key=lambda x: -x[1])[:5]

            return {
                "sessions_today": sessions_today,
                "avg_evaluator_score": round(float(avg_score), 3),
                "top_recommended_products": [{"product_id": p, "count": c} for p, c in top_products],
            }

    def _row_to_dict(self, row: AuditRow) -> dict:
        return {
            "audit_id": row.audit_id,
            "timestamp": row.timestamp.isoformat() if row.timestamp else "",
            "session_id": row.session_id,
            "user_id": row.user_id,
            "turn_number": row.turn_number,
            "user_input": row.user_input,
            "detected_intents": json.loads(row.detected_intents) if row.detected_intents else [],
            "agents_called": json.loads(row.agents_called) if row.agents_called else [],
            "profile_updates": json.loads(row.profile_updates) if row.profile_updates else {},
            "recommendations_made": json.loads(row.recommendations_made) if row.recommendations_made else [],
            "cross_sell_signals": json.loads(row.cross_sell_signals) if row.cross_sell_signals else [],
            "response_text": row.response_text,
            "evaluator_score": row.evaluator_score,
            "guardrail_violations": json.loads(row.guardrail_violations) if row.guardrail_violations else [],
            "latency_ms": row.latency_ms,
            "llm_tokens_used": row.llm_tokens_used,
            "model_used": row.model_used,
            "fallback_triggered": bool(row.fallback_triggered),
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger = AuditLogger(db_path="./data/et_concierge_test.db")

    # Log a sample turn
    audit_id = logger.log_turn({
        "session_id": "test_session_001",
        "user_id": "test_user_001",
        "turn_number": 1,
        "user_input": "Hi, I want to start investing",
        "detected_intents": ["first_investment_intent", "greeting"],
        "agents_called": ["profiler", "recommender"],
        "profile_updates": {"investment_experience": "none"},
        "recommendations_made": [{"product_id": "et_money"}, {"product_id": "et_sip_calculator"}],
        "cross_sell_signals": [],
        "response_text": "Welcome! Let me help you get started...",
        "evaluator_score": 0.85,
        "guardrail_violations": [],
        "latency_ms": 1200,
        "llm_tokens_used": 350,
        "model_used": "gemini-1.5-flash",
        "fallback_triggered": False,
    })
    print(f"Logged audit entry: {audit_id}")

    # Get session summary
    summary = logger.get_session_summary("test_session_001")
    print(f"Session summary: {summary}")

    # Get metrics
    metrics = logger.get_metrics()
    print(f"Metrics: {metrics}")

    print("Audit logger OK")
