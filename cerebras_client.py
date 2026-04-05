"""
Cerebras AI Client for Text Humanization
Multi-pass pipeline: NLP variation → minimal AI repair → repeat → format only.
"""

import os
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

load_dotenv()

API_KEY = os.getenv("CEREBRAS_API_KEY")
MODEL = os.getenv("CEREBRAS_MODEL", "qwen-3-235b-a22b-instruct-2507")

client = Cerebras(api_key=API_KEY)


REPAIR_PROMPT = """You are an EXTREMELY STRICT AND MINIMAL text corrector. Automated NLP processing has been applied to this text. 

YOUR ONLY JOB: Fix things ONLY if they are absolutely, fundamentally broken and make zero logical sense.
IF IT MAKES SENSE, EVEN IF IT SOUNDS WEIRD OR CLUNKY, YOU MUST NOT TOUCH IT.

STRICT INSTRUCTIONS (READ CAREFULLY):
1. ONLY replace a word if it is grammatically invalid (e.g. "he runned") or completely destroys the sentence's meaning.
2. DO NOT REPLACE, REPHRASE, OR REWRITE IF IT IS ALREADY UNDERSTANDABLE.
3. DO NOT smooth out the flow. Do not make it sound better. Do not fix awkwardness.
4. I repeat: ONLY replace if it is absolutely necessary and affects the core meaning, else DO NOT TOUCH IT!
5. KEEP exactly the same sentence structure, length, and transition words.
6. If a sentence has a weird choice of words but you can still understand the point: LEAVE IT ALONE.
7. ABSOLUTELY NO MARKDOWN: Do not return any text wrapping like **bold**, *italics*, or ### headers. Return raw text only.

If you rewrite sentences to sound better, you will fail your core objective. You must return the text exactly as provided, fixing ONLY broken fragments.

Output the text with ONLY the absolute minimum fixes applied. No explanations, no labels, no markdown formatting."""


# ─── TRANSLATION SHUFFLE PROMPT ─────────────────────────────────────────────────
TRANSLATE_TO_GERMAN_PROMPT = """You are a highly accurate English to German translator. Translate the following text into fluent, natural German. Capture the exact original meaning and tone, but use natural German phrasing and sentence structures. Output only the German text."""

TRANSLATE_TO_ENGLISH_PROMPT = """You are a highly accurate German to English translator. Translate the following text into fluent English. Output only the English text."""


# ─── FORMAT-ONLY PROMPT ─────────────────────────────────────────────────────────
# Final pass. Zero content changes. Only mechanical formatting corrections.
FORMAT_ONLY_PROMPT = """You are a text formatter. Your job is EXCLUSIVELY to fix mechanical spacing and punctuation formatting. The words, sentences, and content must remain word-for-word identical.

ALLOWED CORRECTIONS (and ONLY these):
- Fix double spaces → single space
- Add a space after a period, comma, semicolon, or colon if one is missing
- Remove a space immediately before a period, comma, semicolon, or colon
- Fix capitalisation of the first word after a period (.)
- Fix paragraph spacing (one blank line between paragraphs if missing)

STRICTLY FORBIDDEN:
- Changing any word, even a single character
- Changing sentence structure or order
- Adding or removing any sentence
- Rephrasing or improving any phrasing
- Making any change you are not 100% certain is purely a spacing/punctuation fix
- ADDING MARKDOWN: You must not use **bold**, *italics*, or any markdown formatting.

Rule of thumb: if you are unsure whether a change is purely formatting → DO NOT make it.

Output the text with ONLY mechanical formatting corrections applied. Nothing else. No markdown."""


# ─── ACADEMIC HUMANIZE PROMPT (for AI-only / first-pass AI mode) ────────────────
ACADEMIC_HUMANIZE_PROMPT = """You are an expert academic editor who rewrites AI-generated text to be undetectable by AI detectors, without ever making the text casual or informal.

AI detectors flag two signals: low perplexity (predictable word choices) and low burstiness (uniform sentence lengths). Fix both.

WHAT TO CHANGE:
1. **Burstiness**: Vary sentence lengths dramatically. Short. Then long and complex. Then medium. Humans write in bursts, not uniformly.
2. **Perplexity**: Use unexpected-but-correct word choices. Instead of "significant impact" → "considerable bearing." Instead of "demonstrates" → "points to." Avoid the most obvious synonym.
3. **Kill AI clichés**: Replace "Furthermore," "Moreover," "Additionally," "In conclusion," "It is important to note that," "Notably," "It is worth noting" with varied alternatives: "That said," "Even so," "Yet," "On reflection," "In practice," "To this end."
4. **Academic hedging**: Add "This suggests," "The evidence points to," "One might argue," "It seems reasonable to conclude."
5. **Embedded clauses**: "The results, though preliminary, suggest a strong correlation."
6. **Varied paragraph density**: Let one paragraph be one sentence. Let another be four.

RULES:
- Never use contractions, slang, or casual language
- Preserve all facts and meaning exactly
- Output ONLY the rewritten text"""


CASUAL_HUMANIZE_PROMPT = """You are an expert text humanizer. Rewrite this AI-generated text to sound like a real person — natural, conversational, genuine.

Apply: contractions freely, varied sentence lengths, informal transitions ("Plus," "Thing is," "Look,"), start sentences with "And" or "But" occasionally, natural imperfections, idiomatic language, varied paragraph lengths.

RULES:
- Keep the core meaning intact
- Do NOT add new facts
- Output ONLY the rewritten text"""


def polish_iteration(text: str, style: str = "academic") -> str:
    """
    Minimal repair pass after NLP processing.
    Fixes only genuinely broken words/phrases. Leaves everything else verbatim.

    Args:
        text: NLP-processed text that may have some broken substitutions
        style: "academic" or "casual" (affects system context only)

    Returns:
        Text with only broken parts corrected
    """
    style_note = (
        "The text is academic writing — maintain formal register in any fix."
        if style == "academic"
        else "The text is casual writing — keep the conversational tone in any fix."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": REPAIR_PROMPT + f"\n\nContext: {style_note}"},
                {"role": "user", "content": f"Fix only what is broken in this text:\n\n{text}"}
            ],
            max_completion_tokens=4096,
            temperature=0.15,   # Very low temperature — we want conservative, minimal changes
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return text  # If API fails, return original unchanged


def format_only(text: str) -> str:
    """
    Final formatting pass. Corrects ONLY mechanical spacing and punctuation.
    Zero content changes permitted.

    Args:
        text: The fully humanized text

    Returns:
        Text with only formatting corrections applied
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": FORMAT_ONLY_PROMPT},
                {"role": "user", "content": f"Apply formatting corrections only to this text:\n\n{text}"}
            ],
            max_completion_tokens=4096,
            temperature=0.0,    # Zero temperature — purely deterministic formatting corrections
            top_p=1.0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text  # If API fails, return unchanged


def translation_shuffle(text: str) -> str:
    """
    Translates text to German and then back to English.
    This forcibly restructures sentences and erases original AI token sequences.
    """
    try:
        # Step 1: English -> German
        de_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": TRANSLATE_TO_GERMAN_PROMPT},
                {"role": "user", "content": f"Translate this text to German:\n\n{text}"}
            ],
            max_completion_tokens=4096,
            temperature=0.7,
        )
        german_text = de_response.choices[0].message.content.strip()

        # Step 2: German -> English
        en_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": TRANSLATE_TO_ENGLISH_PROMPT},
                {"role": "user", "content": f"Translate this text to English:\n\n{german_text}"}
            ],
            max_completion_tokens=4096,
            temperature=0.7,
        )
        english_text = en_response.choices[0].message.content.strip()

        return english_text
    except Exception:
        return text  # Fallback to original text if API fails



