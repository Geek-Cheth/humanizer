"""
OpenRouter client — replaces both cerebras_client.py and anthropic_client.py.

Each pipeline role has a ranked fallback chain of free models.
On any error (429 rate-limit, 404, 5xx, timeout) the next model in the list
is tried automatically. If every model in the chain fails, the original text
is returned unchanged so the pipeline never crashes.

Fingerprint mixing is preserved: translation uses Meta/Llama models first;
intelligence passes use Qwen first. Even when falling back, the two chains
use intentionally different model families.

Rate limits on free tier: ~20 req/min, ~200 req/day per model.
Multiple models in each chain effectively multiplies available capacity.
"""

import os
import re
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ─── FALLBACK CHAINS ─────────────────────────────────────────────────────────────
# Models tried in order on any error. IDs from openrouter.ai/collections/free-models
# Active parameter counts matter for speed — MoE models skip most of their weights.
#
# TRANSLATION chain — needs multilingual support (EN, ES, FR at minimum)
# Primary is GPT-OSS-120B: 5.1B active params, OpenAI quality, strong multilingual.
# Llama 3.3 70B is the proven explicit-multilingual fallback (supports ES+FR by spec).
# TRANSLATION: needs multilingual (EN, ES, FR). 4 families = strong fingerprint mixing.
# Note: Gemma free tier routes via Google AI Studio which occasionally rejects system
# prompts — the fallback chain catches this and moves to the next model automatically.
TRANSLATION_MODELS = [
    "openai/gpt-oss-120b:free",                  # primary   — 5.1B active, OpenAI, multilingual
    "meta-llama/llama-3.3-70b-instruct:free",    # fallback1 — Meta, explicitly supports ES+FR
    "google/gemma-3-27b-it:free",                # fallback2 — Google, 140+ languages, 128K ctx
    "mistralai/mistral-small-3.1-24b-instruct:free", # fallback3 — Mistral, 24B, multilingual
    "stepfun/step-3.5-flash:free",               # fallback4 — 11B active, multilingual, fast
    "qwen/qwen3-next-80b-a3b-instruct:free",     # fallback5 — 3B active, multilingual, 262K ctx
]

# INTELLIGENCE: speed + instruction following + language quality. 5 model families.
INTELLIGENCE_MODELS = [
    "stepfun/step-3.5-flash:free",               # primary   — 11B active, fastest, no thinking
    "openai/gpt-oss-120b:free",                  # fallback1 — 5.1B active, OpenAI quality
    "google/gemma-3-27b-it:free",                # fallback2 — Google/DeepMind, 27B, 128K ctx
    "arcee-ai/trinity-large-preview:free",       # fallback3 — 13B active, creative writing
    "mistralai/mistral-small-3.1-24b-instruct:free", # fallback4 — Mistral, 24B, strong instruct
    "google/gemma-3-12b-it:free",                # fallback5 — Google, 12B, fast, 128K ctx
    "qwen/qwen3-next-80b-a3b-instruct:free",     # fallback6 — 3B active, no thinking, 262K ctx
    "openai/gpt-oss-20b:free",                   # fallback7 — 3.6B active, emergency fastest
]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://text-humanizer.app",
        "X-Title": "Text Humanizer",
    }
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _tokens_for(text: str, factor: float = 2.0) -> int:
    """
    Dynamic max_tokens based on input length.
    Hardcoding 4096 cuts off 10k-character texts because:
      10 000 chars / 4 chars-per-token = 2 500 input tokens
      x2 expansion + 1 024 buffer = 6 024 — safely covers the full output.
    factor=2.0 for transformation passes (text may grow).
    factor=1.4 for structure/format passes (length stays roughly equal).
    Capped at 32 000 — beyond that, free-tier models truncate anyway.
    """
    return min(32_000, max(4_096, int(len(text) / 4 * factor) + 1_024))


def _thinking_extra(model: str) -> dict:
    """Suppress chain-of-thought for models that support it."""
    thinking_models = ("qwen3", "qwen3.6", "step-3", "thinking", "glm-4.5", "gpt-oss")
    if any(k in model for k in thinking_models):
        return {"reasoning": {"enabled": False}}
    return {}


def _call_one(model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
    """Single model call — raises on any error."""
    extra = _thinking_extra(model)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra if extra else None,
    )
    content = response.choices[0].message.content or ""
    return _THINK_RE.sub("", content).strip()


def _call(models, system, user, temperature=0.7, max_tokens=4096, label="pass"):
    """
    Try each model in order with per-error delays, then retry the whole chain
    once after a longer pause if every model is rate-limited simultaneously.

    Retry schedule:
      429 rate-limit  - wait RATE_WAIT seconds, try next model
      404 not found   - skip immediately (model unavailable)
      other error     - skip immediately
      all exhausted   - wait CHAIN_RETRY_WAIT seconds, retry full chain once
      still all fail  - return None (caller keeps original text)
    """
    RATE_WAIT        = 3   # seconds between models after a 429
    CHAIN_RETRY_WAIT = 20  # seconds before retrying the full chain

    def _attempt(attempt_num):
        for i, model in enumerate(models):
            try:
                result = _call_one(model, system, user, temperature, max_tokens)
                if i > 0 or attempt_num > 1:
                    print(f"[openrouter] {label}: OK (attempt {attempt_num}, "
                          f"model #{i} - {model})")
                return result
            except Exception as e:
                err = str(e)
                is_rate  = "429" in err or "rate" in err.lower()
                is_404   = "404" in err or "No endpoints" in err
                has_next = i < len(models) - 1
                reason   = "rate-limited" if is_rate else ("not found" if is_404 else "error")
                wait     = RATE_WAIT if is_rate else 0
                if has_next:
                    print(f"[openrouter] {label}: {model} {reason}"
                          + (f" (waiting {wait}s)" if wait else "")
                          + f" -> trying {models[i+1]}")
                    if wait:
                        time.sleep(wait)
                else:
                    print(f"[openrouter] {label}: all {len(models)} models exhausted "
                          f"(attempt {attempt_num}). Last: {e}")
        return None

    result = _attempt(1)
    if result is not None:
        return result

    print(f"[openrouter] {label}: waiting {CHAIN_RETRY_WAIT}s before retrying chain...")
    time.sleep(CHAIN_RETRY_WAIT)
    return _attempt(2)


# ─── PROMPTS ──────────────────────────────────────────────────────────────────────

TRANSLATE_PROMPTS = {
    "en_to_es": "You are a precise English to Spanish translator. Translate the text into natural, fluent Spanish. Preserve meaning and tone exactly. Output only the Spanish text, nothing else.",
    "es_to_fr": "You are a precise Spanish to French translator. Translate the text into natural, fluent French. Preserve meaning and tone exactly. Output only the French text, nothing else.",
    "fr_to_en": "You are a precise French to English translator. Translate the text into clear, natural English that reads well — comfortable and readable, not the most cliched phrasing, but not rare or overly formal words either. Output only the English text, nothing else.",
}

SPLIT_MERGE_PROMPT = """You are a sentence-length disruptor. AI detectors flag UNIFORM sentence lengths. Your only job is to vary sentence boundaries aggressively.

INSTRUCTIONS:
1. SPLIT: Any sentence over 25 words containing "and", "but", "which", "who", or "because" — split into two complete sentences.
2. MERGE: Any two consecutive sentences each under 12 words — fuse with a semicolon (;) or em-dash (—).
3. FRAGMENT: Occasionally leave a short sentence as a punchy fragment after a long one: "This matters." / "Worth noting."
4. VARY OPENERS: If three consecutive sentences start with the same word, change the third.

HARD RULES:
- Do NOT change any vocabulary, meaning, or phrasing.
- Do NOT add markdown.
- Mix lengths: some 5 words, some 35+, most in-between.

Output only the restructured text."""

PERPLEXITY_BOOST_PROMPT = """You are a vocabulary disruptor. AI detectors flag text with LOW PERPLEXITY — the words chosen are the most statistically expected choices. Your job is to raise perplexity by swapping predictable words for less-expected but still natural alternatives.

NATURALNESS IS NON-NEGOTIABLE. Every replacement must sound like something a real person would write. No archaic words, no formal words that feel out of place, nothing that would make a reader pause. If it sounds stiff or Victorian — DO NOT USE IT.

WHAT TO DO:
1. Find adjectives, adverbs, verbs, and nouns that are the "obvious" choice in their sentence.
2. Replace them with words that mean the same thing but are slightly less predictable — while remaining completely natural.
3. Target ~15-20% of content words. A few well-chosen swaps beat many awkward ones.
4. Never change proper nouns, numbers, or technical terms.

GOOD SWAPS:
- "demonstrates that" -> "points to" / "suggests"
- "it is clear that" -> "evidently," / "clearly,"
- "due to the fact that" -> "given that" / "since"
- "a number of" -> "several" / "a handful of"
- "important" -> "key" / "central" / "pressing"
- "shows" -> "reveals" / "indicates"
- "utilize" -> "use"

BAD SWAPS (NEVER):
- "provide" -> "furnish" (archaic)
- "shows" -> "betrays" (pretentious)
- "important" -> "non-trivial" (jargon)

RULES:
- Swap individual words only. Never rephrase sentences.
- Do NOT add or remove sentences.
- Do NOT use markdown.
- When in doubt — choose natural over unexpected.

Output only the modified text."""

HUMANITY_INJECTION_PROMPT = """You are inserting authentic human writing patterns into text that sounds too polished. Real human writing has quirks AI almost never produces spontaneously.

YOUR TASK: Inject 2-4 of the following patterns. Pick locations that feel natural — if a location feels forced, skip it. Fewer well-placed injections beat many awkward ones.

PATTERNS (choose most natural-fitting):

1. PARENTHETICAL ASIDES:
   "The results were striking (though not entirely surprising given earlier findings)."
   "This approach — borrowed loosely from manufacturing — proved unexpectedly useful."

2. HEDGED CLAIMS:
   "This seems to suggest..." / "One might argue..." / "To some extent, at least..."

3. SELF-CORRECTION:
   "The pattern, or rather the absence of one, was the key insight."
   "Three — actually, four — distinct phases were identified."

4. ABRUPT SHORT SENTENCES after a long one:
   "...which depends on assumptions that rarely hold in practice. That matters."
   "The gap is real."

5. CASUAL CONNECTIVES (casual mode only):
   "And yet..." / "Even so," / "Which is odd, because..."

RULES:
- Maximum 1 injection per paragraph.
- Every injection must feel like it belongs.
- Do NOT use markdown formatting.
- Do NOT add complex vocabulary.
- Preserve all facts and meaning exactly.

Output only the modified text."""

NATURALNESS_PROMPT = """You are a naturalness editor. Your ONLY job is to find word choices that no real person would naturally write — archaic, overly formal, stilted, or jarring — and replace them with simple natural equivalents.

WHAT TO FIX:
- Archaic: "furnish" -> "provide", "heretofore" -> "until now"
- Stilted machine-translation output: "effectuate a resolution" -> "reach a solution"

WHAT TO LEAVE ALONE (intentional human patterns):
- Sentence fragments, em-dashes, parentheticals
- Hedging language ("arguably", "to some extent")
- Mixed US/UK spellings, short punchy sentences

RULES:
- Swap individual words/short phrases only. Never rephrase whole sentences.
- Do NOT add or remove sentences. Do NOT use markdown.
- If unsure -> leave it alone.

Output only the corrected text."""

FORMAT_ONLY_PROMPT = """You are a mechanical text formatter. Fix ONLY spacing and punctuation. Words and sentences must be word-for-word identical.

ALLOWED: Fix double spaces, add space after period/comma/semicolon/colon, remove space before punctuation, fix capitalisation after a period.

FORBIDDEN: Changing any word, structure, rephrasing, adding/removing sentences, markdown.

Output only the corrected text."""


# ─── PIPELINE FUNCTIONS ───────────────────────────────────────────────────────────

def triple_translation(text: str) -> str:
    """EN -> ES -> FR -> EN using the translation fallback chain."""
    try:
        mt = _tokens_for(text, factor=2.0)
        es = _call(TRANSLATION_MODELS, TRANSLATE_PROMPTS["en_to_es"], text, temperature=0.6, max_tokens=mt, label="EN->ES")
        if not es:
            return text
        fr = _call(TRANSLATION_MODELS, TRANSLATE_PROMPTS["es_to_fr"], es,   temperature=0.6, max_tokens=mt, label="ES->FR")
        if not fr:
            return text
        en = _call(TRANSLATION_MODELS, TRANSLATE_PROMPTS["fr_to_en"], fr,   temperature=0.7, max_tokens=mt, label="FR->EN")
        return en if en else text
    except Exception as e:
        print(f"[openrouter] triple_translation unexpected error: {e}")
        return text


def structural_restructure(text: str) -> str:
    """Vary sentence lengths aggressively (burstiness)."""
    result = _call(
        INTELLIGENCE_MODELS, SPLIT_MERGE_PROMPT,
        f"Restructure sentence boundaries for extreme length variation:\n\n{text}",
        temperature=0.3, max_tokens=_tokens_for(text, factor=1.4), label="structural_restructure",
    )
    return result if result else text


def perplexity_boost(text: str) -> str:
    """Replace predictable words with natural but less-expected alternatives."""
    result = _call(
        INTELLIGENCE_MODELS, PERPLEXITY_BOOST_PROMPT,
        f"Boost perplexity of this text:\n\n{text}",
        temperature=0.72, max_tokens=_tokens_for(text, factor=1.6), label="perplexity_boost",
    )
    return result if result else text


def humanity_injection(text: str, style: str = "academic") -> str:
    """Inject parentheticals, hedging, self-corrections, punchy fragments."""
    style_note = (
        "ACADEMIC mode: Do NOT use casual connectives (pattern 5). Focus on parentheticals, hedging, self-correction."
        if style == "academic" else
        "CASUAL mode: Use all pattern types including casual connectives."
    )
    result = _call(
        INTELLIGENCE_MODELS,
        HUMANITY_INJECTION_PROMPT + f"\n\nStyle: {style_note}",
        f"Inject human writing patterns:\n\n{text}",
        temperature=0.75, max_tokens=_tokens_for(text, factor=1.6), label="humanity_injection",
    )
    return result if result else text


def naturalness_smoother(text: str) -> str:
    """Fix archaic/stilted words from translation without reconverging to AI patterns."""
    result = _call(
        INTELLIGENCE_MODELS, NATURALNESS_PROMPT,
        f"Fix genuinely unnatural word choices:\n\n{text}",
        temperature=0.1, max_tokens=_tokens_for(text, factor=1.4), label="naturalness_smoother",
    )
    return result if result else text


def format_only(text: str) -> str:
    """Final pass: spacing and punctuation only. Zero content changes."""
    result = _call(
        INTELLIGENCE_MODELS, FORMAT_ONLY_PROMPT,
        f"Apply formatting corrections only:\n\n{text}",
        temperature=0.0, max_tokens=_tokens_for(text, factor=1.4), label="format_only",
    )
    return result if result else text