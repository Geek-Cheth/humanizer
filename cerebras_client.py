"""
Cerebras AI Client for Text Humanization
Uses the Cerebras API to rewrite text in a more human-like manner.
"""

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_URL = "https://api.cerebras.ai/v1/chat/completions"
API_KEY = os.getenv("CEREBRAS_API_KEY")
MODEL = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")

HUMANIZE_SYSTEM_PROMPT = """You are an expert text humanizer. Your job is to rewrite AI-generated text to make it sound naturally human-written while preserving the original meaning.

Apply these humanization techniques:

1. **Sentence Variation**: Mix short, punchy sentences with longer, flowing ones. Humans don't write uniformly.

2. **Contractions**: Use contractions naturally (don't, it's, we're, they've). AI tends to avoid them.

3. **Informal Transitions**: Use casual connectors like "Plus," "Thing is," "Here's the deal," "Look," occasionally.

4. **Natural Redundancies**: Humans sometimes restate things slightly differently for emphasis.

5. **Conversational Tone**: Add occasional personal touches, rhetorical questions, or asides.

6. **Imperfect Structure**: Start some sentences with "And" or "But". Use fragments occasionally.

7. **Active Voice**: Prefer active voice but mix in passive occasionally for variety.

8. **Idiomatic Expressions**: Sprinkle in common idioms and colloquialisms where appropriate.

9. **Varied Paragraph Lengths**: Some paragraphs can be just one sentence. Others longer.

10. **Reduce Formality**: Avoid overly formal constructions that sound robotic.

IMPORTANT RULES:
- Keep the core meaning and information intact
- Don't add new facts or claims
- Don't make it too casual if the original is academic/professional
- Match the general tone but make it feel human
- Output ONLY the rewritten text, no explanations or meta-commentary"""


def humanize_with_ai(text: str, intensity: str = "medium") -> str:
    """
    Use Cerebras AI to humanize the given text.
    
    Args:
        text: The text to humanize
        intensity: How aggressively to humanize ("light", "medium", "heavy")
    
    Returns:
        Humanized text from the AI
    """
    
    intensity_prompts = {
        "light": "Make subtle changes to sound more natural. Keep most of the original structure.",
        "medium": "Rewrite to sound genuinely human while keeping the meaning. Apply moderate changes.",
        "heavy": "Significantly rewrite to sound completely human-written. Be creative with structure and phrasing."
    }
    
    user_prompt = f"""{intensity_prompts.get(intensity, intensity_prompts["medium"])}

Text to humanize:
\"\"\"
{text}
\"\"\""""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": HUMANIZE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 4096,
        "temperature": 0.8,  # Higher temperature for more creative/varied output
        "top_p": 0.95
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            raise Exception("No response content from Cerebras API")
            
    except requests.exceptions.Timeout:
        raise Exception("Cerebras API request timed out")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Cerebras API error: {str(e)}")


def polish_with_ai(text: str) -> str:
    """
    Light polish pass to clean up text after NLP processing.
    """
    
    polish_prompt = """Lightly polish this text for readability. Fix any awkward phrasing from automated processing, but keep the content and style intact. Only make minimal necessary corrections.

Text:
\"\"\"
{text}
\"\"\""""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a text editor. Make minimal corrections for readability. Output only the polished text."},
            {"role": "user", "content": polish_prompt.format(text=text)}
        ],
        "max_tokens": 4096,
        "temperature": 0.3  # Lower temperature for conservative edits
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            return text  # Return original if API fails
            
    except Exception:
        return text  # Return original if any error


if __name__ == "__main__":
    # Test the client
    test_text = """Artificial intelligence has revolutionized numerous industries. It has enabled unprecedented advancements in healthcare, finance, and transportation. The implementation of machine learning algorithms has facilitated the automation of complex tasks. Furthermore, natural language processing has enhanced human-computer interaction significantly."""
    
    print("Original:")
    print(test_text)
    print("\n" + "="*50 + "\n")
    print("Humanized:")
    print(humanize_with_ai(test_text, "medium"))
