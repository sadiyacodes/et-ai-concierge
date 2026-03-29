"""
ET AI Concierge — Orchestrator & LangGraph State Machine
Master agent brain: routes, coordinates, assembles, and quality-checks all responses.
"""

import asyncio
import time
import uuid
from typing import Optional, TypedDict, Annotated
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END

from memory_store import MemoryStore
from et_knowledge_base import ETKnowledgeBase
from profiler_agent import ProfilerAgent
from planner_agent import PlannerAgent
from recommender_agent import RecommenderAgent
from cross_sell_agent import CrossSellAgent
from guardrails import GuardrailsEngine
from audit_logger import AuditLogger
from evaluator import EvaluatorAgent
from tools import get_llm, classify_intent

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
# Shared State Schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    session_id: str
    user_id: str
    messages: list
    current_turn: dict
    user_profile: dict
    journey_state: dict
    active_agent: str
    agent_outputs: dict
    recommendations: list
    cross_sell_signals: list
    response_draft: str
    evaluator_score: float
    needs_retry: bool
    retry_count: int
    handoff_target: str
    audit_log: list
    error_state: dict
    turn_number: int
    total_tokens: int
    model_used: str
    latency_start: float


# ---------------------------------------------------------------------------
# Singleton instances (initialised once)
# ---------------------------------------------------------------------------
_memory_store: MemoryStore | None = None
_knowledge_base: ETKnowledgeBase | None = None
_profiler: ProfilerAgent | None = None
_planner: PlannerAgent | None = None
_recommender: RecommenderAgent | None = None
_cross_sell: CrossSellAgent | None = None
_guardrails: GuardrailsEngine | None = None
_audit: AuditLogger | None = None
_evaluator: EvaluatorAgent | None = None


def _init_components():
    global _memory_store, _knowledge_base, _profiler, _planner, _recommender
    global _cross_sell, _guardrails, _audit, _evaluator

    if _memory_store is None:
        _memory_store = MemoryStore()
    if _knowledge_base is None:
        _knowledge_base = ETKnowledgeBase(use_chroma=True)
    if _profiler is None:
        _profiler = ProfilerAgent()
    if _planner is None:
        _planner = PlannerAgent()
    if _recommender is None:
        _recommender = RecommenderAgent(_knowledge_base)
    if _cross_sell is None:
        _cross_sell = CrossSellAgent()
    if _guardrails is None:
        _guardrails = GuardrailsEngine()
    if _audit is None:
        _audit = AuditLogger()
    if _evaluator is None:
        _evaluator = EvaluatorAgent()


def get_memory_store() -> MemoryStore:
    _init_components()
    return _memory_store


def get_knowledge_base() -> ETKnowledgeBase:
    _init_components()
    return _knowledge_base


# ---------------------------------------------------------------------------
# LangGraph Node Functions
# ---------------------------------------------------------------------------

async def intent_classifier_node(state: AgentState) -> AgentState:
    """Classify user input into intents."""
    try:
        user_input = state["current_turn"].get("user_input", "")
        context = ""
        if state.get("messages"):
            recent = state["messages"][-4:]
            context = " | ".join(f"{m['role']}: {m['content'][:80]}" for m in recent)

        result = await classify_intent(user_input, context)

        state["current_turn"]["detected_intent"] = result.get("primary_intent", "financial_question")
        state["current_turn"]["all_intents"] = result.get("intents", [])
        state["current_turn"]["confidence"] = result.get("confidence", 0.5)
        state["total_tokens"] = state.get("total_tokens", 0) + result.get("tokens_used", 0)

    except Exception as e:
        state["current_turn"]["detected_intent"] = "financial_question"
        state["current_turn"]["all_intents"] = ["financial_question"]
        state["current_turn"]["confidence"] = 0.3
        state["error_state"] = {"node": "intent_classifier", "error": str(e)}

    return state


async def router_node(state: AgentState) -> AgentState:
    """Route to appropriate agent(s) based on intent and journey stage."""
    intent = state["current_turn"].get("detected_intent", "financial_question")
    all_intents = state["current_turn"].get("all_intents", [intent])
    profile = state.get("user_profile", {})
    completeness = profile.get("profile_completeness", 0.0)

    agents_to_call = []

    # Always run profiler if profile is thin (unless out_of_scope)
    if completeness < 0.8 and intent != "out_of_scope":
        agents_to_call.append("profiler")

    # Route based on intent
    if intent in ("greeting", "profile_question"):
        agents_to_call.append("profiler")
    elif intent == "product_inquiry":
        agents_to_call.append("recommender")
    elif intent == "financial_question":
        if completeness >= 0.4:
            agents_to_call.append("recommender")
        agents_to_call.append("planner")
    elif intent == "re_engagement":
        agents_to_call.append("recommender")
        agents_to_call.append("planner")
    elif intent == "cross_sell_trigger":
        agents_to_call.append("cross_sell")
    elif intent == "complaint":
        agents_to_call.append("direct_response")
    elif intent == "out_of_scope":
        agents_to_call.append("direct_response")

    # Check for cross-sell signals in every turn
    if "cross_sell" not in agents_to_call:
        agents_to_call.append("cross_sell_check")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for a in agents_to_call:
        if a not in seen:
            seen.add(a)
            unique.append(a)

    state["active_agent"] = unique[0] if unique else "direct_response"
    state["agent_outputs"] = state.get("agent_outputs", {})
    state["agent_outputs"]["agents_to_call"] = unique

    return state


async def profiler_node(state: AgentState) -> AgentState:
    """Run the profiler agent."""
    try:
        _init_components()
        user_input = state["current_turn"].get("user_input", "")
        profile = state.get("user_profile", {})
        session_id = state.get("session_id", "")

        # Rule-based inference (fast)
        _memory_store.infer_from_conversation(session_id, user_input)

        # LLM-based extraction
        result = await _profiler.extract_profile_from_turn(user_input, profile)

        # Apply updates
        for field, value in result.get("updated_fields", {}).items():
            conf = result.get("confidence_scores", {}).get(field, 0.5)
            _memory_store.update_profile(session_id, field, value, conf)

        # Refresh profile from memory store
        session = _memory_store.get_session(session_id)
        if session:
            state["user_profile"] = session["user_profile"]

        state["agent_outputs"]["profiler"] = {
            "updated_fields": result.get("updated_fields", {}),
            "next_question": result.get("next_question"),
            "inferred_signals": result.get("inferred_signals", []),
        }
        state["total_tokens"] = state.get("total_tokens", 0) + result.get("tokens_used", 0)

    except Exception as e:
        state["agent_outputs"]["profiler"] = {"error": str(e)}
        state["error_state"] = {"node": "profiler", "error": str(e)}

    return state


async def planner_node(state: AgentState) -> AgentState:
    """Run the planner agent."""
    try:
        _init_components()
        profile = state.get("user_profile", {})
        journey = state.get("journey_state", {})

        if not journey or not journey.get("phase"):
            # Build initial journey
            journey = await _planner.build_journey(profile)
        else:
            # Advance existing journey
            user_action = state["current_turn"].get("user_input", "")
            journey = await _planner.advance_journey(journey, user_action)

        state["journey_state"] = journey
        state["agent_outputs"]["planner"] = {
            "phase": journey.get("phase"),
            "current_step": journey.get("current_step"),
            "cta": _planner.get_current_cta(journey),
        }

    except Exception as e:
        state["agent_outputs"]["planner"] = {"error": str(e)}

    return state


async def recommender_node(state: AgentState) -> AgentState:
    """Run the recommender agent."""
    try:
        _init_components()
        profile = state.get("user_profile", {})
        user_input = state["current_turn"].get("user_input", "")

        # Infer persona if missing
        persona = profile.get("persona_tag", "")
        if not persona:
            session_id = state.get("session_id", "")
            persona = _memory_store.infer_persona(session_id)
            profile["persona_tag"] = persona

        result = await _recommender.recommend(profile, _knowledge_base, user_query=user_input)

        state["recommendations"] = result.get("recommendations", [])
        state["agent_outputs"]["recommender"] = result

    except Exception as e:
        state["recommendations"] = []
        state["agent_outputs"]["recommender"] = {"error": str(e)}

    return state


async def cross_sell_node(state: AgentState) -> AgentState:
    """Run the cross-sell agent."""
    try:
        _init_components()
        user_input = state["current_turn"].get("user_input", "")
        turns = state.get("messages", [])
        turn_num = state.get("turn_number", 0)
        profile = state.get("user_profile", {})
        session = _memory_store.get_session(state.get("session_id", ""))

        signals = _cross_sell.detect_signals(turns, user_input, turn_num)
        state["cross_sell_signals"] = [s.to_dict() for s in signals]

        cross_sell_msg = None
        for signal in signals:
            session_state = session if session else {}
            if _cross_sell.should_inject_now(signal, turn_num, session_state):
                context = " | ".join(m.get("content", "")[:50] for m in turns[-3:])
                msg = await _cross_sell.generate_cross_sell_message(signal, profile, context)
                if msg:
                    cross_sell_msg = msg
                    # Update session injection tracking
                    if session:
                        session["last_cross_sell_turn"] = turn_num
                        session["cross_sell_injections_count"] = session.get("cross_sell_injections_count", 0) + 1
                    break

        state["agent_outputs"]["cross_sell"] = {
            "signals_detected": [s.to_dict() for s in signals],
            "message": cross_sell_msg,
        }

    except Exception as e:
        state["agent_outputs"]["cross_sell"] = {"error": str(e)}

    return state


def _find_best_product_link(user_input: str, recommendations: list, profile: dict) -> dict | None:
    """Find the single most relevant product for the user's query and return name + url."""
    _init_components()

    # If we already have ranked recommendations, use the top one
    if recommendations:
        top = recommendations[0]
        pid = top.get("product_id", "")
        product = _knowledge_base.get_product(pid)
        if product:
            return {"name": product["name"], "url": product["url"]}

    # Otherwise semantic/keyword search on the query
    results = _knowledge_base.semantic_search(user_input, n=1)
    if results:
        p = results[0]
        return {"name": p["name"], "url": p["url"]}

    # Last resort — persona default
    persona_defaults = {
        "first_time_investor": "et_money",
        "seasoned_trader": "et_markets_app",
        "lapsed_subscriber": "et_prime",
        "wealth_builder": "et_prime",
        "nri": "et_money",
    }
    persona = profile.get("persona_tag", "first_time_investor")
    default_id = persona_defaults.get(persona, "et_prime")
    product = _knowledge_base.get_product(default_id)
    if product:
        return {"name": product["name"], "url": product["url"]}

    return None


async def response_assembler_node(state: AgentState) -> AgentState:
    """Assemble all agent outputs into a single coherent response."""
    try:
        _init_components()
        llm = get_llm()
        user_input = state["current_turn"].get("user_input", "")
        profile = state.get("user_profile", {})
        persona = profile.get("persona_tag", "first_time_investor")
        outputs = state.get("agent_outputs", {})
        retry_instruction = ""

        if state.get("needs_retry"):
            eval_result = outputs.get("evaluator", {})
            retry_instruction = f"\n\nIMPROVEMENT REQUIRED: {eval_result.get('improvement_instruction', '')}\n"

        # Gather context pieces
        profiler_output = outputs.get("profiler", {})
        recommender_output = outputs.get("recommender", {})
        planner_output = outputs.get("planner", {})
        cross_sell_output = outputs.get("cross_sell", {})

        next_question = profiler_output.get("next_question", "")
        recommendations = state.get("recommendations", [])
        cross_sell_msg = cross_sell_output.get("message")
        journey_cta = planner_output.get("cta", "")

        # Build prompt for response assembly
        recs_text = ""
        if recommendations:
            recs_text = "Recommendations to weave in:\n"
            for r in recommendations[:3]:
                recs_text += f"- {r.get('product_name', '')}: {r.get('user_facing_reason', '')}\n"

        cross_sell_text = ""
        if cross_sell_msg:
            cross_sell_text = f"\nCross-sell message to include naturally: {cross_sell_msg.get('message', '')}"

        # Live articles from Exa Search
        live_articles_text = ""
        live_articles = recommender_output.get("live_articles", [])
        if live_articles:
            live_articles_text = "\nRecent ET articles relevant to this conversation (reference 1-2 if appropriate):\n"
            for art in live_articles[:3]:
                live_articles_text += f"- [{art['title']}]({art['url']})"
                if art.get("snippet"):
                    live_articles_text += f" — {art['snippet'][:100]}"
                live_articles_text += "\n"

        persona_configs = CONFIG.get("personas", {})
        persona_config = persona_configs.get(persona, persona_configs.get("first_time_investor", {}))
        tone = persona_config.get("tone", "warm, helpful")
        max_len = persona_config.get("max_response_length", 4)

        system_prompt = f"""You are the ET AI Concierge — a knowledgeable, warm assistant for the Economic Times ecosystem.

Persona: {persona}
Tone: {tone}
Max length: {max_len} sentences

Rules:
- Address the user's question directly first, then weave in recommendations naturally
- Never mention competitor platforms
- No jargon for beginners; appropriate terminology for experienced investors
- Maximum 3 product recommendations per response
- Be conversational, not robotic
- No exclamation marks overuse, no CAPS, no FOMO language
{retry_instruction}"""

        prompt = f"""User message: "{user_input}"

User profile: persona={persona}, experience={profile.get('investment_experience', 'unknown')}, goal={profile.get('primary_financial_goal', 'unknown')}

{recs_text}
{cross_sell_text}
{live_articles_text}

{"Next profiling question to ask: " + next_question if next_question else ""}
{"Journey CTA: " + journey_cta if journey_cta else ""}

Compose a natural, helpful response that:
1. Addresses the user's message directly
2. Weaves in relevant recommendations (if any) naturally
3. {"Includes the cross-sell message organically" if cross_sell_msg else ""}
4. {"Ends with the profiling question" if next_question else "Ends with a helpful prompt or CTA"}

Write ONLY the response text. No prefixes, no labels, no markdown."""

        result = await llm.generate(prompt, system_prompt)
        response_text = result.get("text", "").strip()
        state["total_tokens"] = state.get("total_tokens", 0) + result.get("tokens_used", 0)
        state["model_used"] = result.get("model_used", "unknown")

        # Apply guardrails
        cleaned, violations = _guardrails.filter_response(response_text)

        # Add disclaimers
        cleaned = _guardrails.add_required_disclaimers(cleaned, {
            "recommendations": recommendations,
            "profile": profile,
        })

        # Check for sensitive data in user input
        sensitive = _guardrails.check_sensitive_data_in_input(user_input)
        if sensitive:
            cleaned = _guardrails.get_sensitive_data_response(sensitive) + "\n\n" + cleaned

        # Always append the most relevant product hyperlink
        best_product = _find_best_product_link(user_input, recommendations, profile)
        if best_product:
            cleaned = cleaned.rstrip()
            cleaned += f"\n\n🔗 **Explore:** [{best_product['name']}]({best_product['url']})"

        state["response_draft"] = cleaned
        state["agent_outputs"]["guardrail_violations"] = violations

        if not response_text:
            # LLM returned empty — use fallback
            state["response_draft"] = _evaluator.graceful_fallback(profile)

    except Exception as e:
        state["response_draft"] = _evaluator.graceful_fallback(state.get("user_profile", {}))
        state["error_state"] = {"node": "response_assembler", "error": str(e)}

    return state


async def evaluator_node(state: AgentState) -> AgentState:
    """Score the response draft and decide on retry or acceptance."""
    try:
        _init_components()
        response = state.get("response_draft", "")
        profile = state.get("user_profile", {})
        user_input = state["current_turn"].get("user_input", "")

        eval_result = await _evaluator.score_response(response, profile, user_input)

        state["evaluator_score"] = eval_result["overall_score"]
        state["agent_outputs"]["evaluator"] = eval_result

        retry_count = state.get("retry_count", 0)

        if not eval_result["passed"] and retry_count < 2:
            state["needs_retry"] = True
            state["retry_count"] = retry_count + 1
            state["agent_outputs"]["evaluator"]["improvement_instruction"] = \
                _evaluator.generate_improvement_instruction(eval_result)
        else:
            state["needs_retry"] = False

            # If still failing after retries, use graceful fallback
            if not eval_result["passed"] and retry_count >= 2:
                if eval_result["overall_score"] < 0.6:
                    state["response_draft"] = _evaluator.graceful_fallback(profile)

    except Exception as e:
        state["evaluator_score"] = 0.7
        state["needs_retry"] = False

    return state


async def audit_node(state: AgentState) -> AgentState:
    """Log the completed turn to the audit trail."""
    try:
        _init_components()
        latency_ms = int((time.time() - state.get("latency_start", time.time())) * 1000)

        audit_entry = {
            "session_id": state.get("session_id", ""),
            "user_id": state.get("user_id", ""),
            "turn_number": state.get("turn_number", 0),
            "user_input": state["current_turn"].get("user_input", ""),
            "detected_intents": state["current_turn"].get("all_intents", []),
            "agents_called": state["agent_outputs"].get("agents_to_call", []),
            "profile_updates": state["agent_outputs"].get("profiler", {}).get("updated_fields", {}),
            "recommendations_made": [
                {"product_id": r.get("product_id", ""), "product_name": r.get("product_name", "")}
                for r in state.get("recommendations", [])
            ],
            "cross_sell_signals": state.get("cross_sell_signals", []),
            "response_text": state.get("response_draft", ""),
            "evaluator_score": state.get("evaluator_score", 0.0),
            "guardrail_violations": state["agent_outputs"].get("guardrail_violations", []),
            "latency_ms": latency_ms,
            "llm_tokens_used": state.get("total_tokens", 0),
            "model_used": state.get("model_used", ""),
            "fallback_triggered": bool(state.get("error_state")),
        }

        audit_id = _audit.log_turn(audit_entry)
        state["audit_log"] = state.get("audit_log", [])
        state["audit_log"].append(audit_id)

        # Save turn to memory
        session_id = state.get("session_id", "")
        _memory_store.add_turn(session_id, "user", state["current_turn"].get("user_input", ""))
        _memory_store.add_turn(session_id, "assistant", state.get("response_draft", ""),
                               agent_name=state.get("active_agent", ""),
                               confidence_score=state.get("evaluator_score", 0.7))

    except Exception as e:
        pass  # Audit should never block the response

    return state


# ---------------------------------------------------------------------------
# Router conditional logic
# ---------------------------------------------------------------------------
def route_after_intent(state: AgentState) -> str:
    """Decide next node after intent classification."""
    return "router"


def route_agents(state: AgentState) -> str:
    """Decide which agent pipeline to run."""
    agents = state.get("agent_outputs", {}).get("agents_to_call", [])
    if not agents:
        return "response_assembler"

    # Run profiler first if needed, then others
    if "profiler" in agents:
        return "profiler"
    elif "recommender" in agents:
        return "recommender"
    elif "cross_sell" in agents:
        return "cross_sell"
    elif "planner" in agents:
        return "planner"
    else:
        return "response_assembler"


def route_after_profiler(state: AgentState) -> str:
    """After profiler, continue to other agents."""
    agents = state.get("agent_outputs", {}).get("agents_to_call", [])
    if "recommender" in agents:
        return "recommender"
    if "planner" in agents:
        return "planner"
    return "response_assembler"


def route_after_recommender(state: AgentState) -> str:
    agents = state.get("agent_outputs", {}).get("agents_to_call", [])
    if "cross_sell" in agents or "cross_sell_check" in agents:
        return "cross_sell"
    if "planner" in agents:
        return "planner"
    return "response_assembler"


def route_after_cross_sell(state: AgentState) -> str:
    agents = state.get("agent_outputs", {}).get("agents_to_call", [])
    if "planner" in agents:
        return "planner"
    return "response_assembler"


def route_after_planner(state: AgentState) -> str:
    return "response_assembler"


def route_after_evaluator(state: AgentState) -> str:
    """After evaluation, retry or proceed to audit."""
    if state.get("needs_retry"):
        return "response_assembler"
    return "audit"


# ---------------------------------------------------------------------------
# Build the Graph
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("router", router_node)
    graph.add_node("profiler", profiler_node)
    graph.add_node("planner", planner_node)
    graph.add_node("recommender", recommender_node)
    graph.add_node("cross_sell", cross_sell_node)
    graph.add_node("response_assembler", response_assembler_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("audit", audit_node)

    # Set entry point
    graph.set_entry_point("intent_classifier")

    # Edges
    graph.add_edge("intent_classifier", "router")

    graph.add_conditional_edges("router", route_agents, {
        "profiler": "profiler",
        "recommender": "recommender",
        "cross_sell": "cross_sell",
        "planner": "planner",
        "response_assembler": "response_assembler",
    })

    graph.add_conditional_edges("profiler", route_after_profiler, {
        "recommender": "recommender",
        "planner": "planner",
        "response_assembler": "response_assembler",
    })

    graph.add_conditional_edges("recommender", route_after_recommender, {
        "cross_sell": "cross_sell",
        "planner": "planner",
        "response_assembler": "response_assembler",
    })

    graph.add_conditional_edges("cross_sell", route_after_cross_sell, {
        "planner": "planner",
        "response_assembler": "response_assembler",
    })

    graph.add_edge("planner", "response_assembler")
    graph.add_edge("response_assembler", "evaluator")

    graph.add_conditional_edges("evaluator", route_after_evaluator, {
        "response_assembler": "response_assembler",
        "audit": "audit",
    })

    graph.add_edge("audit", END)

    return graph


# Compile the graph
_compiled_graph = None

def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def run_turn(user_input: str, session_id: str, user_id: str) -> dict:
    """Main interface — runs a single conversational turn through the entire pipeline."""
    _init_components()

    # Ensure session exists
    session = _memory_store.get_session(session_id)
    if session is None:
        session = _memory_store.create_session(user_id, session_id)

    turn_number = len(session.get("raw_turns", [])) // 2  # Approximate turn count

    # Build initial state
    initial_state: AgentState = {
        "session_id": session_id,
        "user_id": user_id,
        "messages": session.get("raw_turns", []),
        "current_turn": {"user_input": user_input},
        "user_profile": session.get("user_profile", {}),
        "journey_state": session.get("journey_state", {}),
        "active_agent": "",
        "agent_outputs": {},
        "recommendations": [],
        "cross_sell_signals": [],
        "response_draft": "",
        "evaluator_score": 0.0,
        "needs_retry": False,
        "retry_count": 0,
        "handoff_target": "",
        "audit_log": [],
        "error_state": {},
        "turn_number": turn_number,
        "total_tokens": 0,
        "model_used": "",
        "latency_start": time.time(),
    }

    # Run the graph
    graph = get_graph()
    final_state = await graph.ainvoke(initial_state)

    # Update session memory with journey state
    if final_state.get("journey_state"):
        session["journey_state"] = final_state["journey_state"]

    return {
        "response": final_state.get("response_draft", "I'm here to help. What would you like to know about ET's products and services?"),
        "recommendations": final_state.get("recommendations", []),
        "journey_step": final_state.get("journey_state", {}).get("current_step"),
        "journey_phase": final_state.get("journey_state", {}).get("phase", "discover"),
        "profile_completeness": final_state.get("user_profile", {}).get("profile_completeness", 0.0),
        "evaluator_score": final_state.get("evaluator_score", 0.0),
        "audit_id": final_state.get("audit_log", [""])[0] if final_state.get("audit_log") else "",
        "intents": final_state.get("current_turn", {}).get("all_intents", []),
        "agents_called": final_state.get("agent_outputs", {}).get("agents_to_call", []),
        "cross_sell_signals": final_state.get("cross_sell_signals", []),
        "session_id": session_id,
    }


# ---------------------------------------------------------------------------
# Quick start / session management helpers
# ---------------------------------------------------------------------------
async def start_session(user_id: str | None = None, returning_user_hint: bool = False) -> dict:
    """Start a new session and return greeting."""
    _init_components()
    user_id = user_id or str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    # Check for returning user
    history = _memory_store.load_user_history(user_id) if user_id else None

    if history and returning_user_hint:
        user_type = "returning"
        past_profile = history.get("profile", {})
        sub_status = past_profile.get("et_subscription_status", "none")
        if sub_status == "lapsed":
            user_type = "lapsed"

        session = _memory_store.create_session(user_id, session_id)
        # Merge past profile
        for field, value in past_profile.items():
            if value and value not in ("", [], 0, None, 0.0):
                _memory_store.update_profile(session_id, field, value, 0.6)

        if user_type == "lapsed":
            greeting = (
                "Welcome back to Economic Times. It's good to see you again. "
                "A lot has changed since your last visit — new tools, new analysis, and some features "
                "I think you'll find interesting. What brings you back today?"
            )
        else:
            greeting = (
                "Welcome back. Good to see you again at Economic Times. "
                "How can I help you today?"
            )
    else:
        user_type = "new"
        _memory_store.create_session(user_id, session_id)
        greeting = (
            "Welcome to Economic Times. I'm your AI concierge — I can help you "
            "discover the right tools, content, and financial products across the ET ecosystem. "
            "Whether you're just starting your investment journey or looking for advanced market analysis, "
            "I'm here to help. What's on your mind?"
        )

    return {
        "session_id": session_id,
        "user_id": user_id,
        "greeting": greeting,
        "user_type": user_type,
    }


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _test():
        print("=" * 60)
        print("ET AI Concierge — Orchestrator Test")
        print("=" * 60)

        # Scenario 1: Cold start
        print("\n--- Scenario 1: Cold Start ---")
        session = await start_session()
        print(f"Greeting: {session['greeting'][:100]}...")

        result = await run_turn(
            "Hi, I came across Economic Times. I'm not sure where to start.",
            session["session_id"],
            session["user_id"],
        )
        print(f"Response: {result['response'][:150]}...")
        print(f"Intents: {result['intents']}")
        print(f"Profile completeness: {result['profile_completeness']}")

        # Turn 2
        result = await run_turn(
            "I'm 28, work in IT. Never invested before. Friends keep talking about SIPs.",
            session["session_id"],
            session["user_id"],
        )
        print(f"\nTurn 2 Response: {result['response'][:150]}...")
        print(f"Profile completeness: {result['profile_completeness']}")
        print(f"Recommendations: {[r.get('product_name') for r in result['recommendations']]}")

        # Scenario 2: Lapsed subscriber
        print("\n\n--- Scenario 2: Lapsed Subscriber ---")
        session2 = await start_session(returning_user_hint=True)
        # Manually set lapsed status
        _memory_store.update_profile(session2["session_id"], "et_subscription_status", "lapsed", 0.9)
        _memory_store.update_profile(session2["session_id"], "persona_tag", "lapsed_subscriber", 0.9)
        _memory_store.update_profile(session2["session_id"], "preferred_content_types", ["markets", "smallcap"], 0.8)

        result = await run_turn(
            "Hey, I used to have ET Prime but my subscription lapsed a while back. Just browsing again.",
            session2["session_id"],
            session2["user_id"],
        )
        print(f"Response: {result['response'][:150]}...")

        # Scenario 3: Home loan cross-sell
        print("\n\n--- Scenario 3: Home Loan Cross-Sell ---")
        session3 = await start_session()
        _memory_store.update_profile(session3["session_id"], "persona_tag", "seasoned_trader", 0.8)
        _memory_store.update_profile(session3["session_id"], "investment_experience", "advanced", 0.8)

        result = await run_turn(
            "What are the current home loan interest rates? Nifty is volatile today too.",
            session3["session_id"],
            session3["user_id"],
        )
        print(f"Response: {result['response'][:150]}...")
        print(f"Cross-sell signals: {result['cross_sell_signals']}")

        print("\n" + "=" * 60)
        print("Orchestrator test complete.")

    asyncio.run(_test())
