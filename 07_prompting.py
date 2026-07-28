"""
07_prompting.py
----------------
Stage 7 of the RAG pipeline: turning (query, context) into a prompt, and
talking to the local LLM (Ollama).

Moved from rag_engine.py, unchanged, per the refactor mapping:
    build_weak_prompt, build_better_prompt, build_strict_prompt,
    PROMPT_BUILDERS, ask_ollama, ask_ollama_stream, extract_final_answer,
    parse_cited_sources
"""

import json
import re

import requests

# --------------------------------------------------------------------------- #
# OpenRouter client config. The API key is read from Streamlit secrets at app
# startup (see streamlit_app.py) and written into OPENROUTER_API_KEY. When
# empty, the app falls back to local Ollama. The default model is a free one
# so the deployment works without adding credits; swap to gpt-4o-mini (or any
# other OpenRouter model id) in st.secrets once the account has credits.
# --------------------------------------------------------------------------- #
OPENROUTER_API_KEY = ""
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --------------------------------------------------------------------------- #
# 8. Prompt writing
# --------------------------------------------------------------------------- #

def build_weak_prompt(query, context_text):
    return f"""Answer the question using the context.

Question:
{query}

Context:
{context_text}
"""


def build_better_prompt(query, context_text):
    return f"""You are a careful vehicle-listings assistant.

Answer using only the provided context.

Rules:
1. Do not use outside knowledge about cars, brands, or prices.
2. If the context is not enough to answer, say so clearly.
3. If sources disagree, prefer the CURRENT source and mention the conflict.
4. Cite the source number(s) you used.

Question:
{query}

Context:
{context_text}
"""


def build_strict_prompt(query, context_text):
    return f"""You are a vehicle-listings assistant operating under strict grounding rules.

STRICT RULES:
- Use ONLY the context below. Never use prior knowledge about car prices or specs.
- If sources conflict, prefer CURRENT over OUTDATED/SUPERSEDED and say so explicitly.
- If the context does not fully answer the question, say exactly what is missing.

Aggregation questions (cheapest / most expensive / lowest / highest / oldest / newest):
- Only answer an aggregation if a "[STRUCTURED RESULT]" line appears in the context,
  or if the sources unambiguously cover every row that could match the question.
- Otherwise, do NOT pick the min/max from a small sample and present it as the
  global min/max. Say "I can't determine that from these N sources alone —
  the corpus may contain rows I didn't retrieve."
- When a "[STRUCTURED RESULT]" line is present, use it as the authoritative answer
  and cite its doc id(s) in SOURCES USED.

User pushback ("is it not X?", "I thought Y was cheaper"):
- If your retrieved context does not support the user's claim, say so plainly.
  Do not change your answer to match the user's assertion unless a source supports it.
- If prior assistant turns appear in the conversation, treat them as your own
  previous grounded answers; you may stand by them when the context still supports
  them, or correct yourself when a new retrieval shows they were wrong.

Output format:
  1) ANSWER: a short, direct answer.
  2) SOURCES USED: the source numbers you relied on.

Question:
{query}

Context:
{context_text}
"""


PROMPT_BUILDERS = {
    "Weak": build_weak_prompt,
    "Better": build_better_prompt,
    "Strict": build_strict_prompt,
}


def build_strict_system_prompt():
    """Rules-only version of the Strict prompt, used as the `system` message
    when multi-turn chat history is threaded through the /chat endpoints.
    The per-turn `user` message then only carries the question + current
    retrieval context (see build_user_turn_message below), avoiding the
    rules being repeated once per turn.

    Keep the rule text in sync with build_strict_prompt above."""
    return """You are a vehicle-listings assistant operating under strict grounding rules.

STRICT RULES:
- Use ONLY the context provided in the current user message. Never use prior knowledge about car prices or specs.
- If sources conflict, prefer CURRENT over OUTDATED/SUPERSEDED and say so explicitly.
- If the context does not fully answer the question, say exactly what is missing.

Aggregation questions (cheapest / most expensive / lowest / highest / oldest / newest):
- Only answer an aggregation if a "[STRUCTURED RESULT]" line appears in the context,
  or if the sources unambiguously cover every row that could match the question.
- Otherwise, do NOT pick the min/max from a small sample and present it as the
  global min/max. Say "I can't determine that from these N sources alone —
  the corpus may contain rows I didn't retrieve."
- When a "[STRUCTURED RESULT]" line is present, use it as the authoritative answer
  and cite its doc id(s) in SOURCES USED.

User pushback ("is it not X?", "I thought Y was cheaper"):
- If your retrieved context does not support the user's claim, say so plainly.
  Do not change your answer to match the user's assertion unless a source supports it.
- Prior assistant turns in the conversation are your own previous grounded answers;
  stand by them when the context still supports them, or correct yourself when a
  new retrieval shows they were wrong.

Output format:
  1) ANSWER: a short, direct answer.
  2) SOURCES USED: the source numbers you relied on.
"""


def build_user_turn_message(query, context_text):
    """The user-role message for one RAG turn in a multi-turn chat. Carries
    just the question + the current retrieval context; the rules live in the
    system message (see build_strict_system_prompt)."""
    return f"""Question:
{query}

Context:
{context_text}
"""


# --------------------------------------------------------------------------- #
# 9. LLM connection (local Ollama, same pattern as Lab 9)
# --------------------------------------------------------------------------- #

def ollama_status(host: str):
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return True, models
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def ask_ollama(prompt: str, host: str, model: str, temperature: float = 0.0) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(f"{host}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"__ERROR__: Ollama request failed: {exc}"
    return resp.json().get("response", "")


def ask_ollama_stream(prompt: str, host: str, model: str, temperature: float = 0.0):
    """Generator version of ask_ollama for live token-by-token rendering
    (st.write_stream on the UI side). Yields text pieces as they arrive.

    If the connection drops mid-stream, whatever was already received is
    still yielded, followed by a final "__ERROR__: ..." sentinel chunk so the
    caller can tell the stream ended abnormally rather than just finishing."""
    import json as _json

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(f"{host}/api/generate", json=payload, timeout=120, stream=True)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        yield f"__ERROR__: Ollama request failed: {exc}"
        return

    try:
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = _json.loads(line)
            except ValueError:
                continue
            piece = chunk.get("response", "")
            if piece:
                yield piece
            if chunk.get("done"):
                return
    except Exception as exc:  # noqa: BLE001
        yield f"__ERROR__: Stream interrupted: {exc}"


def ask_ollama_chat_stream(messages, host: str, model: str, temperature: float = 0.0):
    """Multi-turn version of ask_ollama_stream. Uses Ollama's /api/chat
    endpoint, which accepts a messages list (system + user/assistant turns)
    instead of a single prompt. Same __ERROR__ sentinel contract.

    `messages` must be a list of {"role": "system"|"user"|"assistant",
    "content": str} dicts. The caller is responsible for capping history
    length before calling this."""
    import json as _json

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=120, stream=True)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        yield f"__ERROR__: Ollama request failed: {exc}"
        return

    try:
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = _json.loads(line)
            except ValueError:
                continue
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
            if chunk.get("done"):
                return
    except Exception as exc:  # noqa: BLE001
        yield f"__ERROR__: Stream interrupted: {exc}"


# --------------------------------------------------------------------------- #
# 9b. LLM connection (OpenRouter -- used in deployment when st.secrets holds
# OPENROUTER_API_KEY). Mirrors the Ollama functions above: same __ERROR__
# sentinel contract for the streaming variant so the UI's dispatch in
# stream_or_fallback can treat both providers identically.
# --------------------------------------------------------------------------- #

def openrouter_status(api_key: str):
    """Quick connectivity + key check. Returns (ok, info) where info is a
    list of model ids on success or an error string on failure."""
    if not api_key:
        return False, "no API key configured"
    try:
        resp = requests.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        models = [m.get("id", "") for m in resp.json().get("data", [])]
        return True, models
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def ask_openrouter(prompt: str, api_key: str, model: str, temperature: float = 0.0) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    }
    try:
        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"__ERROR__: OpenRouter request failed: {exc}"
    try:
        return resp.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError) as exc:
        return f"__ERROR__: OpenRouter response parse failed: {exc}"


def ask_openrouter_stream(prompt: str, api_key: str, model: str, temperature: float = 0.0):
    """Generator version of ask_openrouter. Yields content deltas as they
    arrive from OpenRouter's SSE stream. Same __ERROR__ sentinel contract
    as ask_ollama_stream so the UI can detect interruption."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": True,
    }
    try:
        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        yield f"__ERROR__: OpenRouter request failed: {exc}"
        return

    try:
        for line in resp.iter_lines():
            if not line:
                continue
            # SSE frames arrive as `data: {json}` (and a terminal `data: [DONE]`).
            try:
                text = line.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if not text.startswith("data:"):
                continue
            payload_str = text[len("data:"):].strip()
            if payload_str == "[DONE]":
                return
            try:
                chunk = json.loads(payload_str)
            except ValueError:
                continue
            try:
                delta = chunk["choices"][0].get("delta", {}).get("content", "")
            except (KeyError, IndexError):
                delta = ""
            if delta:
                yield delta
    except Exception as exc:  # noqa: BLE001
        yield f"__ERROR__: Stream interrupted: {exc}"


def ask_openrouter_chat_stream(messages, api_key: str, model: str, temperature: float = 0.0):
    """Multi-turn version of ask_openrouter_stream. Same SSE parsing, same
    __ERROR__ sentinel contract, but the payload takes the messages list
    directly so prior user/assistant turns are visible to the model. Caller
    is responsible for capping history length before calling this.

    `messages` must be a list of {"role": "system"|"user"|"assistant",
    "content": str} dicts. The first message should typically be a system
    message carrying the rules (see build_strict_system_prompt)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    try:
        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        yield f"__ERROR__: OpenRouter request failed: {exc}"
        return

    try:
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                text = line.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if not text.startswith("data:"):
                continue
            payload_str = text[len("data:"):].strip()
            if payload_str == "[DONE]":
                return
            try:
                chunk = json.loads(payload_str)
            except ValueError:
                continue
            try:
                delta = chunk["choices"][0].get("delta", {}).get("content", "")
            except (KeyError, IndexError):
                delta = ""
            if delta:
                yield delta
    except Exception as exc:  # noqa: BLE001
        yield f"__ERROR__: Stream interrupted: {exc}"


def extract_final_answer(raw_text: str) -> str:
    if not raw_text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def parse_cited_sources(answer_text: str, num_sources: int):
    """Pulls the source numbers out of a Strict-prompt answer's trailing
    "SOURCES USED: 1, 3 and 5" line. Only meaningful for the Strict prompt
    style -- Weak/Better answers aren't asked to produce this line and are
    rendered as-is. Returns a de-duplicated, order-preserving list of ints,
    each clamped to the valid 1..num_sources range so a hallucinated source
    number can't produce a citation marker that points nowhere."""
    match = re.search(r"SOURCES USED:?\s*(.+)", answer_text, flags=re.IGNORECASE)
    if not match:
        return []
    tail = match.group(1)
    seen = []
    for n_str in re.findall(r"\d+", tail):
        n = int(n_str)
        if 1 <= n <= num_sources and n not in seen:
            seen.append(n)
    return seen
