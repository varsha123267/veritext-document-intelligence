"""
Thin wrapper around the Gemini API for three jobs:

1. Grounded answer generation: answer a question using ONLY the retrieved
   passages, citing them inline as [1], [2], etc.
2. Optional hybrid re-ranking: ask the model to re-order a shortlist of
   vector-search candidates by true relevance before generation.
3. One-sentence document summaries for the index cards.

Reliability
-----------
Gemini model names change over time, and busy models sometimes answer
"503 UNAVAILABLE" or "429 RESOURCE_EXHAUSTED". So instead of trusting one
hard-coded model, every call works through a chain of models:

  your GEMINI_MODEL (default "gemini-flash-latest")
  -> a few well-known names
  -> every Flash model that YOUR API KEY can actually use (asked from Google)

A busy model is retried once, then skipped. A model that doesn't exist for
your key (404) or is out of quota (429) is skipped immediately. An invalid
API key is reported immediately, since no other model could fix that. If
every model fails, the error lists what each one said, so the real cause is
visible instead of hidden.

If GEMINI_API_KEY is not set, everything degrades gracefully to
retrieval-only behaviour.
"""
import json
import os
import re
import time
from typing import Dict, List, Optional

_client = None
_discovered: Optional[List[str]] = None

_STATIC_FALLBACKS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
_MAX_CHAIN = 6

# Models that can't answer plain-text questions well (or aren't text models).
_EXCLUDE_TOKENS = (
    "image", "tts", "live", "audio", "embed", "vision", "robotics",
    "computer-use", "imagen", "veo", "gemma", "native", "thinking",
)

_TRANSIENT_MARKERS = (
    "503", "unavailable", "500", "internal", "504", "deadline",
    "overloaded", "high demand", "timed out", "timeout",
)


def _get_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    from google import genai  # imported lazily so the app still starts without a key
    _client = genai.Client(api_key=api_key)
    return _client


def is_available() -> bool:
    return _get_client() is not None


def _discover_models() -> List[str]:
    """Ask Google which Flash text models this API key can use (cached)."""
    global _discovered
    if _discovered is not None:
        return _discovered
    found: List[str] = []
    try:
        client = _get_client()
        for m in client.models.list():
            name = (getattr(m, "name", "") or "").replace("models/", "")
            low = name.lower()
            actions = getattr(m, "supported_actions", None) or []
            if actions and "generateContent" not in actions:
                continue
            if not low.startswith("gemini") or "flash" not in low:
                continue
            if any(tok in low for tok in _EXCLUDE_TOKENS):
                continue
            found.append(name)
    except Exception:  # discovery is best-effort
        found = []
    found.sort(reverse=True)  # newest version numbers first
    # Stable models before previews/experiments, full models before "lite".
    found.sort(key=lambda n: ("preview" in n or "exp" in n, "lite" in n))
    _discovered = found
    return found


def _model_chain(max_models: Optional[int] = None) -> List[str]:
    primary = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    chain: List[str] = []
    for m in [primary] + _STATIC_FALLBACKS + _discover_models():
        if m and m not in chain:
            chain.append(m)
    limit = max_models if max_models else _MAX_CHAIN
    return chain[:limit]


def _is_transient(err: Exception) -> bool:
    text = str(err).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _skip_model(err: Exception) -> bool:
    """Errors that mean 'this particular model won't work, try another'."""
    text = str(err).lower()
    return (
        "404" in text or "not_found" in text or "not found" in text
        or "429" in text or "resource_exhausted" in text or "quota" in text
    )


def _short(err: Exception, limit: int = 140) -> str:
    text = " ".join(str(err).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _generate(prompt: str, attempts_per_model: int = 2, max_models: Optional[int] = None) -> str:
    """Call Gemini with retries and model fallback. Returns the response text,
    or raises an error listing what every model said."""
    client = _get_client()
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    problems: List[str] = []
    for model in _model_chain(max_models):
        for attempt in range(attempts_per_model):
            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                text = (resp.text or "").strip()
                if text:
                    return text
                problems.append(f"{model}: empty response")
                break
            except Exception as e:  # noqa: BLE001 - classified below
                if _skip_model(e):
                    problems.append(f"{model}: {_short(e)}")
                    break  # try the next model right away
                if _is_transient(e):
                    if attempt < attempts_per_model - 1:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    problems.append(f"{model}: {_short(e)}")
                    break
                raise  # e.g. invalid API key: no other model would help
    raise RuntimeError("; ".join(problems) if problems else "No Gemini model could be reached")


def _extract_json(text: str):
    """Tolerant JSON extraction: bare value, fenced code block, or a value
    with a stray sentence before/after it."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No parseable JSON found in model response")


def rerank(question: str, candidates: List[Dict], top_k: int) -> List[Dict]:
    if not is_available() or len(candidates) <= top_k:
        return candidates[:top_k]

    listing = "\n".join(
        f"({i + 1}) [{c['doc_name']}] {c['text'][:220].strip()}"
        for i, c in enumerate(candidates)
    )
    prompt = (
        f'Question: "{question}"\n\n'
        f"Candidate passages:\n{listing}\n\n"
        f"Return ONLY a JSON array of the {top_k} most relevant candidate "
        f"numbers, most relevant first, e.g. [3,1,7]. No other text."
    )
    try:
        # Re-ranking is an optimisation, so keep it quick: one attempt, three models.
        order = _extract_json(_generate(prompt, attempts_per_model=1, max_models=3))
        picked = [candidates[i - 1] for i in order if isinstance(i, int) and 1 <= i <= len(candidates)]
        return picked if picked else candidates[:top_k]
    except Exception:
        return candidates[:top_k]


def generate_answer(
    question: str, sources: List[Dict], history: Optional[List[Dict]] = None
) -> Optional[str]:
    """Returns the answer text, or None if no API key is configured.
    Raises if every model fails (the caller shows a friendly message).

    `history` is a list of {"question": ..., "answer": ...} dicts for the
    most recent prior turns, oldest first. Including it lets the model
    resolve follow-ups like "what about the duration?" that only make sense
    with the previous exchange in mind — without it, every question is
    answered in total isolation."""
    if not is_available():
        return None

    context = "\n\n".join(
        f"[{s['rank']}] (from \"{s['doc_name']}\")\n{s['text']}" for s in sources
    )

    history_block = ""
    if history:
        turns = "\n".join(
            f"User: {h['question']}\nAssistant: {h['answer']}" for h in history
        )
        history_block = (
            "Recent conversation so far (for context only — do not cite or "
            f"quote from it, only from the numbered passages below):\n{turns}\n\n"
        )

    prompt = (
        "You are answering a question using ONLY the numbered passages below, "
        "drawn from the user's own uploaded documents. Answer clearly and "
        "directly. After any sentence that relies on a passage, cite it "
        "inline like [1] or [2]. If the passages don't contain enough "
        "information to answer, say so plainly rather than guessing. If the "
        "question is a follow-up (e.g. uses 'it', 'that', 'what about...'), "
        "use the recent conversation to understand what is being asked, but "
        "still answer strictly from the passages.\n\n"
        f"{history_block}"
        f"Passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return _generate(prompt, attempts_per_model=2)


def summarize(text: str) -> Optional[str]:
    if not is_available():
        return None
    try:
        summary = _generate(
            "In exactly one plain sentence (no preamble, no quotes), "
            f"summarize what this document is about:\n\n{text[:3000]}",
            attempts_per_model=1,
            max_models=3,
        )
        return summary.strip().strip('"')
    except Exception:
        return None
