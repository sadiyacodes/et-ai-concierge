"""
ET AI Concierge — LLM Tool Definitions
Shared utility for calling LLMs (Gemini primary, NVIDIA Nemotron fallback, Ollama last resort).
"""

import os
import json
import asyncio
import time
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

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
# LLM Wrapper
# ---------------------------------------------------------------------------
class LLMClient:
    """Unified LLM client with Gemini primary and Ollama fallback."""

    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", CONFIG.get("llm", {}).get("ollama", {}).get("base_url", "http://localhost:11434"))
        self.primary_model = CONFIG.get("llm", {}).get("primary", {}).get("model", "gemini-2.0-flash")
        self.ollama_model = CONFIG.get("llm", {}).get("ollama", {}).get("model", "mistral:7b")
        self.temperature = CONFIG.get("llm", {}).get("primary", {}).get("temperature", 0.3)
        self.max_tokens = CONFIG.get("llm", {}).get("primary", {}).get("max_tokens", 512)
        self._client = None

    def _init_gemini(self):
        if self._client is None and self.google_api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.google_api_key)
            except Exception as e:
                print(f"[LLM] Gemini init failed: {e}")
                self._client = None

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> dict:
        """Generate text from LLM. Returns {text, model_used, tokens_used}."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        # 1. Try Ollama first
        result = await self._try_ollama(prompt, system_prompt, temp, max_tok, json_mode)
        if result:
            return result

        # 2. Fallback to Gemini
        result = await self._try_gemini(prompt, system_prompt, temp, max_tok, json_mode)
        if result:
            return result

        # All providers failed
        return {
            "text": "",
            "model_used": "none",
            "tokens_used": 0,
            "fallback": True,
        }

    async def _try_gemini(self, prompt, system_prompt, temperature, max_tokens, json_mode) -> dict | None:
        self._init_gemini()
        if self._client is None:
            return None

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            if system_prompt:
                config.system_instruction = system_prompt
            if json_mode:
                config.response_mime_type = "application/json"

            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self.primary_model,
                contents=prompt,
                config=config,
            )

            text = response.text if response.text else ""
            tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens = getattr(response.usage_metadata, "total_token_count", 0)

            return {
                "text": text,
                "model_used": self.primary_model,
                "tokens_used": tokens,
                "fallback": False,
            }
        except Exception as e:
            print(f"[LLM] Gemini call failed: {e}")
            return None



    async def _try_ollama(self, prompt, system_prompt, temperature, max_tokens, json_mode) -> dict | None:
        try:
            import httpx

            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            if json_mode:
                payload["format"] = "json"

            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{self.ollama_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()

            return {
                "text": data.get("response", ""),
                "model_used": self.ollama_model,
                "tokens_used": data.get("eval_count", 0),
                "fallback": True,
            }
        except Exception as e:
            print(f"[LLM] Ollama call failed: {e}")
            return None

    async def generate_structured(
        self,
        prompt: str,
        system_prompt: str = "",
        schema_hint: str = "",
    ) -> dict:
        """Generate JSON-structured output from LLM."""
        json_prompt = prompt
        if schema_hint:
            json_prompt += f"\n\nRespond ONLY with valid JSON matching this schema:\n{schema_hint}"
        json_prompt += "\n\nIMPORTANT: Output ONLY valid JSON, no markdown fences, no explanation."

        result = await self.generate(json_prompt, system_prompt, json_mode=True)

        # Parse JSON from text
        text = result["text"].strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            parsed = json.loads(text)
            result["parsed"] = parsed
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    parsed = json.loads(match.group())
                    result["parsed"] = parsed
                except json.JSONDecodeError:
                    result["parsed"] = {}
            else:
                result["parsed"] = {}

        return result


# ---------------------------------------------------------------------------
# Shared LLM instance
# ---------------------------------------------------------------------------
_llm_client: LLMClient | None = None

def get_llm() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
async def classify_intent(user_input: str, context: str = "") -> dict:
    """Classify user intent into categories."""
    llm = get_llm()
    system = """You are an intent classifier for the Economic Times AI Concierge.
Classify the user's message into one or more intents.

Valid intents:
- profile_question: User is sharing personal/financial info or answering a profile question
- product_inquiry: User asks about an ET product or service
- financial_question: User asks a general financial/market question
- complaint: User expresses dissatisfaction
- re_engagement: User is returning after absence, mentions lapsed subscription
- cross_sell_trigger: Message contains life-event signals (home loan, job change, etc.)
- greeting: Simple hello/hi
- out_of_scope: Completely unrelated to finance or ET

Respond with JSON: {"intents": ["intent1", "intent2"], "primary_intent": "intent1", "confidence": 0.9}"""

    prompt = f"Context: {context}\n\nUser message: {user_input}"

    result = await llm.generate_structured(prompt, system)
    parsed = result.get("parsed", {})

    if not parsed or "intents" not in parsed:
        # Fallback classification
        return {
            "intents": ["financial_question"],
            "primary_intent": "financial_question",
            "confidence": 0.5,
            "model_used": result.get("model_used", "fallback"),
        }

    parsed["model_used"] = result.get("model_used", "unknown")
    return parsed


async def generate_response(
    system_prompt: str,
    user_message: str,
    context: str = "",
    persona: str = "first_time_investor",
) -> dict:
    """Generate a conversational response."""
    llm = get_llm()
    prompt = f"""Persona context: {persona}
Conversation context: {context}

User: {user_message}

Respond naturally and helpfully. Keep within ET brand guidelines."""

    result = await llm.generate(prompt, system_prompt)
    return result


# ---------------------------------------------------------------------------
# Exa Search — Real-time ET Content
# ---------------------------------------------------------------------------
class ExaSearchClient:
    """Wraps the Exa Search API to find live ET content."""

    def __init__(self):
        self.api_key = os.getenv("EXA_API_KEY", "")
        exa_cfg = CONFIG.get("exa", {})
        self.enabled = bool(self.api_key) and exa_cfg.get("enabled", True)
        self.site_filter = exa_cfg.get("site_filter", "economictimes.indiatimes.com")
        self.max_results = exa_cfg.get("max_results", 5)
        self.use_highlights = exa_cfg.get("use_highlights", True)
        self._client = None

    def _init_client(self):
        if self._client is None and self.api_key:
            try:
                from exa_py import Exa
                self._client = Exa(api_key=self.api_key)
            except Exception as e:
                print(f"[Exa] Init failed: {e}")
                self._client = None
                self.enabled = False

    async def search(self, query: str, num_results: int | None = None) -> list[dict]:
        """Search ET website for relevant articles/content.

        Returns list of {title, url, snippet, published_date}.
        """
        if not self.enabled:
            return []

        self._init_client()
        if self._client is None:
            return []

        n = num_results or self.max_results

        try:
            results = await asyncio.to_thread(
                self._client.search_and_contents,
                query,
                num_results=n,
                include_domains=[self.site_filter],
                highlights=self.use_highlights,
                summary=True,
            )

            items = []
            for r in results.results:
                item = {
                    "title": getattr(r, "title", "") or "",
                    "url": getattr(r, "url", "") or "",
                    "snippet": "",
                    "published_date": getattr(r, "published_date", "") or "",
                }
                # Prefer highlight, fall back to summary
                highlights = getattr(r, "highlights", None)
                if highlights:
                    item["snippet"] = " ".join(highlights[:2])
                elif getattr(r, "summary", None):
                    item["snippet"] = r.summary
                items.append(item)

            return items

        except Exception as e:
            print(f"[Exa] Search failed: {e}")
            return []

    async def search_for_topic(self, topic: str, persona: str = "") -> list[dict]:
        """Search with ET-context-aware query enrichment."""
        query_parts = [topic]
        if persona in ("seasoned_trader", "advanced"):
            query_parts.append("markets analysis trading")
        elif persona == "first_time_investor":
            query_parts.append("beginner guide how to invest")
        elif persona == "lapsed_subscriber":
            query_parts.append("latest new today")

        enriched_query = " ".join(query_parts)
        return await self.search(enriched_query)


# Shared Exa instance
_exa_client: ExaSearchClient | None = None

def get_exa() -> ExaSearchClient:
    global _exa_client
    if _exa_client is None:
        _exa_client = ExaSearchClient()
    return _exa_client


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _test():
        llm = get_llm()
        print(f"Google API key set: {bool(llm.google_api_key)}")

        result = await classify_intent("I want to start investing in mutual funds")
        print(f"Intent classification: {result}")

        result = await llm.generate("Say hello in one sentence.", "You are a friendly assistant.")
        print(f"Generation: {result['text'][:100]}... (model: {result['model_used']})")

    asyncio.run(_test())
