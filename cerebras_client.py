"""
Cerebras AI Client for Text Humanization
Improved pipeline: perplexity boost → humanity injection → structural burst → format fix.

Key changes vs original:
- Removed the 'repair' pass (it was converging text back toward AI patterns)
- Added perplexity_boost() — directly targets predictable word sequences
- Added humanity_injection() — injects authentic human writing quirks
- Triple-hop translation for stronger lexical disruption
- Higher temperatures on creative passes
"""

import os
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

load_dotenv()

API_KEY = os.getenv("CEREBRAS_API_KEY")
MODEL = os.getenv("CEREBRAS_MODEL", "qwen-3-235b-a22b-instruct-2507")

client = Cerebras(api_key=API_KEY)


# ─── PERPLEXITY BOOST ────────────────────────────────────────────────────────────
# The #1 signal AI detectors use. Targets predictable/expected word choices.
PERPLEXITY_BOOST_PROMPT = """You are a vocabulary disruptor. AI detectors flag text with LOW PERPLEXITY — meaning the words chosen are the most statistically expected choices given the context. Your job is to raise perplexity by swapping predictable words for valid but less-expected alternatives.

NATURALNESS IS NON-NEGOTIABLE. Every replacement must sound like something a real person would write in a good essay or article. No archaic words. No formal-sounding words that feel out of place. No words that would make a reader pause and think "that's a strange choice." If it sounds stiff, formal, or Victorian — DO NOT USE IT.

WHAT TO DO:
1. Identify adjectives, adverbs, verbs, and nouns that are the "obvious" choice in that sentence.
2. Replace them with words that mean the same thing but are slightly less predictable — while remaining completely natural and readable.
3. Target around 15–20% of content words. Do NOT over-swap — a few well-chosen swaps beat many awkward ones.
4. Never change proper nouns, numbers, or technical/domain-specific terms.

GOOD SWAPS (natural but unexpected):
- "demonstrates that" → "points to" / "suggests"
- "it is clear that" → "evidently," / "clearly,"
- "due to the fact that" → "given that" / "since"
- "a number of" → "several" / "a handful of"
- "conducted a study" → "ran an experiment" / "carried out a study"
- "provides support for" → "backs" / "supports"
- "important" → "key" / "central" / "pressing"
- "shows" → "reveals" / "indicates"
- "utilize" → "use"
- "in the field of" → "in" / "within"

BAD SWAPS — DO NOT DO THIS:
- "provide" → "furnish" (archaic, unnatural)
- "shows" → "betrays" / "surfaces" (sounds pretentious)
- "important" → "non-trivial" (academic jargon)
- "bearing" instead of "impact" (overly formal)
- Any word that sounds like it's from a legal document or Victorian novel

ABSOLUTE RULES:
- Do NOT rephrase whole sentences. Swap individual words WITHIN existing structures.
- Do NOT add or remove sentences.
- Do NOT use markdown or formatting.
- When in doubt between natural and unexpected — CHOOSE NATURAL.

Output only the modified text, nothing else."""


# ─── HUMANITY INJECTION ──────────────────────────────────────────────────────────
# Adds authentic human writing patterns that AI almost never produces naturally.
HUMANITY_INJECTION_PROMPT = """You are inserting authentic human writing patterns into text that sounds too polished. Real human writing has specific quirks that AI almost never produces spontaneously.

YOUR TASK: Inject 2–4 of the following patterns into the text. Pick the locations that feel the most natural — if a location feels forced, skip it. Fewer well-placed injections are better than many awkward ones.

PATTERNS TO INJECT (choose the most natural-fitting ones):

1. PARENTHETICAL ASIDES — a brief digression in parentheses or em-dashes:
   "The results were striking (though not entirely surprising given earlier findings)."
   "This approach — borrowed loosely from manufacturing — proved unexpectedly useful."

2. HEDGED CLAIMS — humans rarely assert things absolutely:
   "This seems to suggest..." / "One might argue..." / "To some extent, at least..."
   "This is arguably the strongest..." / "It's worth asking whether..."

3. SELF-CORRECTION — catching oneself mid-thought:
   "The pattern, or rather the absence of one, was the key insight."
   "Three — actually, four — distinct phases were identified."

4. ABRUPT SHORT SENTENCES after a long complex one:
   "...which itself depends on a chain of assumptions that rarely hold in practice. That matters."
   "The gap is real."

5. CASUAL CONNECTIVES (casual mode only):
   "And yet..." / "Even so," / "Which is odd, because..."

RULES:
- Maximum 1 injection per paragraph — do not cluster them.
- Every injection must feel like it belongs. If it doesn't fit naturally, don't add it.
- Do NOT use markdown formatting.
- Preserve all original facts and meaning exactly.
- Keep the overall reading difficulty the same — do NOT add complex vocabulary.

Output only the modified text, nothing else."""


# ─── TRIPLE-HOP TRANSLATION ──────────────────────────────────────────────────────
# Three language hops provide much stronger lexical disruption than a single hop.
TRANSLATE_PROMPT = {
    "en_to_es": "You are a precise English to Spanish translator. Translate the following text into natural, fluent Spanish. Preserve meaning and tone exactly. Output only the Spanish text.",
    "es_to_fr": "You are a precise Spanish to French translator. Translate the following text into natural, fluent French. Preserve meaning and tone exactly. Output only the French text.",
    "fr_to_en": "You are a precise French to English translator. Translate the following text into clear, natural English that reads well. Choose words that feel comfortable and readable — not the most common clichéd phrasing, but not rare or formal words either. Aim for the kind of English a thoughtful writer would use. Output only the English text.",
}


# ─── STRUCTURAL SPLIT/MERGE ──────────────────────────────────────────────────────
SPLIT_MERGE_PROMPT = """You are a sentence-length disruptor. AI detectors flag UNIFORM sentence lengths (low burstiness). Your only job is to aggressively vary sentence boundaries.

INSTRUCTIONS:
1. SPLIT: Any sentence over 25 words that contains "and", "but", "which", "who", or "because" — split it into two complete sentences.
2. MERGE: Any two consecutive sentences each under 12 words — fuse them with a semicolon (;) or em-dash (—).
3. CREATE FRAGMENTS: Occasionally, when a short sentence follows a long one, leave it as a punchy fragment: "This matters." / "Worth noting." / "The exception, not the rule."
4. VARY OPENERS: If three consecutive sentences start with the same word or structure, change the third.

HARD RULES:
- Do NOT change any vocabulary, meaning, or phrasing.
- Do NOT add markdown.
- Aim for a mix: some sentences 5 words, some 35 words, most in-between.

Output only the restructured text."""


# ─── FORMAT-ONLY PROMPT ──────────────────────────────────────────────────────────
FORMAT_ONLY_PROMPT = """You are a mechanical text formatter. Fix ONLY spacing and punctuation. The words and sentences must be word-for-word identical.

ALLOWED: Fix double spaces, add space after period/comma/semicolon/colon, remove space before punctuation, fix capitalisation after a period, fix paragraph spacing.

FORBIDDEN: Changing any word, changing sentence structure, rephrasing, adding/removing sentences, adding markdown.

If uncertain whether a change is purely formatting → DO NOT make it.
Output only the corrected text. No markdown."""


def perplexity_boost(text: str) -> str:
    """
    Raises perplexity by replacing predictable word choices with less-expected
    but still valid alternatives. This directly attacks the primary AI detection signal.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": PERPLEXITY_BOOST_PROMPT},
                {"role": "user", "content": f"Boost the perplexity of this text by swapping predictable words:\n\n{text}"}
            ],
            max_completion_tokens=4096,
            temperature=0.72,   # Balanced: creative word choices without going obscure
            top_p=0.92
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text


def humanity_injection(text: str, style: str = "academic") -> str:
    """
    Injects authentic human writing patterns: parentheticals, hedging,
    self-corrections, concrete specificity, abrupt short sentences.
    """
    style_note = (
        "This is ACADEMIC writing. Do NOT inject casual connectives (rule 6). "
        "Focus on parentheticals, hedged claims, self-correction, and concrete specificity."
        if style == "academic"
        else
        "This is CASUAL writing. Use all pattern types including casual connectives."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": HUMANITY_INJECTION_PROMPT + f"\n\nStyle note: {style_note}"},
                {"role": "user", "content": f"Inject human writing patterns into this text:\n\n{text}"}
            ],
            max_completion_tokens=4096,
            temperature=0.75,
            top_p=0.95
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text


def triple_translation(text: str) -> str:
    """
    Translates EN → ES → FR → EN.
    Three hops provide far stronger lexical disruption than a single EN→DE→EN pass,
    because each language has different collocations and phrase structures,
    so the final English output uses genuinely different token sequences.
    """
    try:
        # Step 1: EN → ES
        r1 = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": TRANSLATE_PROMPT["en_to_es"]},
                {"role": "user", "content": text}
            ],
            max_completion_tokens=4096,
            temperature=0.65,
        )
        es_text = r1.choices[0].message.content.strip()

        # Step 2: ES → FR
        r2 = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": TRANSLATE_PROMPT["es_to_fr"]},
                {"role": "user", "content": es_text}
            ],
            max_completion_tokens=4096,
            temperature=0.65,
        )
        fr_text = r2.choices[0].message.content.strip()

        # Step 3: FR → EN (with instruction to avoid most-common English words)
        r3 = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": TRANSLATE_PROMPT["fr_to_en"]},
                {"role": "user", "content": fr_text}
            ],
            max_completion_tokens=4096,
            temperature=0.75,   # Slightly higher on final hop for lexical variety
        )
        return r3.choices[0].message.content.strip()

    except Exception:
        return text   # Graceful fallback


def structural_restructure(text: str) -> str:
    """
    Varies sentence lengths aggressively (burstiness injection).
    Splits long sentences, merges short ones, creates occasional fragments.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SPLIT_MERGE_PROMPT},
                {"role": "user", "content": f"Restructure sentence boundaries to create extreme length variation:\n\n{text}"}
            ],
            max_completion_tokens=4096,
            temperature=0.3,    # Low — structural changes only, content must stay identical
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text


def naturalness_smoother(text: str) -> str:
    """
    A targeted readability pass — different from the old 'repair' pass.

    The repair pass asked AI to 'fix broken text', which made it smooth out
    humanised phrasing back toward AI norms. This pass does the opposite:
    it only hunts for word choices that sound STILTED, ARCHAIC, or UNNATURALLY
    FORMAL, and swaps them for normal, comfortable equivalents.

    It does NOT improve flow, does NOT make sentences better, does NOT touch
    anything that already sounds natural — even if it's imperfect.
    """
    NATURALNESS_PROMPT = """You are a naturalness editor. Your ONLY job is to find word choices that no real person would naturally write — words that are archaic, overly formal, stilted, or jarring to read — and replace them with the simple, natural equivalent.

THIS IS NOT ABOUT IMPROVING THE TEXT. You are looking for specific unnatural word choices.

WHAT TO FIX:
- Archaic words: "furnish" → "provide", "heretofore" → "until now", "hitherto" → "so far"
- Overly formal where out of place: "endeavour to ascertain" → "try to find out"  
- Words that sound like a legal document or Victorian novel
- Unnatural collocations produced by machine translation, e.g. "effectuate a resolution" → "reach a solution"
- Stilted phrasing: "it is the case that" → "it's true that"

WHAT TO LEAVE ALONE:
- Anything that sounds like a real person wrote it, even if imperfect or unusual
- Sentence fragments, em-dashes, parentheticals — these are intentional human patterns
- Hedging language ("arguably", "to some extent") — intentional
- Mixed US/UK spellings — intentional
- Short punchy sentences after long ones — intentional
- Anything awkward-but-readable

RULES:
- Only swap individual words or short phrases. Never rephrase whole sentences.
- Do NOT add or remove sentences.  
- Do NOT make the text sound more polished or AI-like.
- Do NOT use markdown.
- If you are not sure a word is unnatural, leave it alone.

Output only the corrected text. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": NATURALNESS_PROMPT},
                {"role": "user", "content": f"Fix only genuinely unnatural or archaic word choices in this text:\n\n{text}"}
            ],
            max_completion_tokens=4096,
            temperature=0.1,   # Very low — conservative, specific fixes only
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text



def format_only(text: str) -> str:
    """
    Final formatting pass — spacing and punctuation only. Zero content changes.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": FORMAT_ONLY_PROMPT},
                {"role": "user", "content": f"Apply formatting corrections only:\n\n{text}"}
            ],
            max_completion_tokens=4096,
            temperature=0.0,
            top_p=1.0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text


# ─── KEPT FOR BACKWARD COMPATIBILITY (minimised role in new pipeline) ────────────
def polish_iteration(text: str, style: str = "academic") -> str:
    """
    Minimal grammar-only fix. Only called if text has genuinely broken fragments.
    NOT called per-pass in the new pipeline — only as emergency fallback.
    """
    MINIMAL_GRAMMAR_PROMPT = """Fix ONLY sentences that are grammatically broken to the point of being unreadable.
Do NOT change vocabulary, rephrase, improve, or smooth anything.
If a sentence is awkward but understandable: LEAVE IT EXACTLY AS IS.
Do NOT use markdown. Output only the text."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": MINIMAL_GRAMMAR_PROMPT},
                {"role": "user", "content": text}
            ],
            max_completion_tokens=4096,
            temperature=0.05,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text


# ─── KEPT FOR BACKWARD COMPATIBILITY ─────────────────────────────────────────────
def translation_shuffle(text: str) -> str:
    """Legacy single-hop translation. Prefer triple_translation() instead."""
    return triple_translation(text)