"""
NLP-based Text Humanizer
Implements programmatic techniques to make text appear more human-written.
"""

import os
import random
import re
import nltk

# Set NLTK data path to /tmp for serverless environments (Vercel, AWS Lambda, etc.)
# /tmp is the only writable directory on these platforms
NLTK_DATA_DIR = '/tmp/nltk_data'
if not os.path.exists(NLTK_DATA_DIR):
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.insert(0, NLTK_DATA_DIR)

# Download required NLTK data to /tmp
def ensure_nltk_data():
    """Download NLTK data if not present."""
    packages = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
    ]
    
    for path, package in packages:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, download_dir=NLTK_DATA_DIR, quiet=True)
            except Exception:
                pass  # Silently continue if download fails

# Initialize NLTK data
ensure_nltk_data()

from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag


# Words to avoid replacing (common, important, or structural)
PROTECTED_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
    'because', 'until', 'while', 'although', 'though', 'i', 'you', 'he',
    'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my',
    'your', 'his', 'its', 'our', 'their', 'this', 'that', 'these', 'those'
}

# Contraction mappings
CONTRACTIONS = {
    'do not': "don't",
    'does not': "doesn't",
    'did not': "didn't",
    'is not': "isn't",
    'are not': "aren't",
    'was not': "wasn't",
    'were not': "weren't",
    'have not': "haven't",
    'has not': "hasn't",
    'had not': "hadn't",
    'will not': "won't",
    'would not': "wouldn't",
    'could not': "couldn't",
    'should not': "shouldn't",
    'cannot': "can't",
    'can not': "can't",
    'must not': "mustn't",
    'it is': "it's",
    'it has': "it's",
    'that is': "that's",
    'there is': "there's",
    'here is': "here's",
    'what is': "what's",
    'who is': "who's",
    'how is': "how's",
    'I am': "I'm",
    'I have': "I've",
    'I will': "I'll",
    'I would': "I'd",
    'you are': "you're",
    'you have': "you've",
    'you will': "you'll",
    'you would': "you'd",
    'we are': "we're",
    'we have': "we've",
    'we will': "we'll",
    'we would': "we'd",
    'they are': "they're",
    'they have': "they've",
    'they will': "they'll",
    'they would': "they'd",
    'he is': "he's",
    'he has': "he's",
    'he will': "he'll",
    'he would': "he'd",
    'she is': "she's",
    'she has': "she's",
    'she will': "she'll",
    'she would': "she'd",
    'let us': "let's",
}

# Informal transitions to inject (casual mode only)
INFORMAL_TRANSITIONS = [
    "Plus, ",
    "Thing is, ",
    "Here's the deal: ",
    "Look, ",
    "Honestly, ",
    "The reality is, ",
    "Truth be told, ",
    "Interestingly enough, ",
    "What's more, ",
    "On top of that, ",
]

# Filler phrases for natural redundancy (casual mode only)
FILLER_PHRASES = [
    "basically",
    "essentially",
    "in a way",
    "kind of",
    "pretty much",
    "actually",
    "really",
]

# AI clichés that detectors are trained to recognise — replace with varied alternatives
AI_CLICHES = {
    r'\bFurthermore\b': ['That said,', 'Even so,', 'Beyond this,', 'In addition,'],
    r'\bfurthermore\b': ['that said,', 'beyond this,', 'in addition,', 'what is more,'],
    r'\bMoreover\b': ['Still,', 'On reflection,', 'As it stands,', 'Of note,'],
    r'\bmoreover\b': ['still,', 'on reflection,', 'as it stands,', 'worth noting,'],
    r'\bAdditionally\b': ['Yet,', 'To this end,', 'Along these lines,', 'In turn,'],
    r'\badditionally\b': ['yet,', 'to this end,', 'along these lines,', 'in turn,'],
    r'\bIn conclusion\b': ['In sum,', 'On balance,', 'Taken together,', 'All things considered,'],
    r'\bin conclusion\b': ['in sum,', 'on balance,', 'taken together,', 'all things considered,'],
    r'\bIt is important to note that\b': ['It is worth considering that', 'One should bear in mind that', 'Crucially,'],
    r'\bit is important to note that\b': ['it is worth considering that', 'one should bear in mind that', 'crucially,'],
    r'\bIt is worth noting that\b': ['Of particular relevance,', 'Notably,', 'This is significant because'],
    r'\bit is worth noting that\b': ['of particular relevance,', 'here,', 'this matters because'],
    r'\bNotably\b': ['Of interest,', 'Tellingly,', 'Here,'],
    r'\bnotably\b': ['of interest,', 'tellingly,', 'here,'],
    r'\bSignificantly\b': ['Strikingly,', 'Tellingly,', 'Of note,'],
    r'\bsignificantly\b': ['strikingly,', 'tellingly,', 'to a marked degree,'],
    r'\bIn summary\b': ['In short,', 'On balance,', 'Taken together,'],
    r'\bin summary\b': ['in short,', 'on balance,', 'taken together,'],
    r'\bTo summarize\b': ['In brief,', 'Stepping back,', 'To draw this together,'],
    r'\bto summarize\b': ['in brief,', 'stepping back,', 'to draw this together,'],
    r'\bIn order to\b': ['To', 'So as to', 'With the aim of'],
    r'\bin order to\b': ['to', 'so as to', 'with the aim of'],
    r'\bplays a crucial role\b': ['is central to', 'matters greatly for', 'underpins'],
    r'\bplays a significant role\b': ['bears heavily on', 'shapes', 'is central to'],
    r'\bhas the potential to\b': ['can', 'may well', 'could'],
    r'\bIt is clear that\b': ['Evidently,', 'The data suggest that', 'This points to'],
    r'\bit is clear that\b': ['evidently,', 'the evidence suggests that', 'this points to'],
    r'\boverarching\b': ['broader', 'general', 'governing'],
    r'\brobust\b': ['strong', 'reliable', 'sound'],
    r'\bseamless\b': ['smooth', 'fluid', 'uninterrupted'],
    r'\bdelve\b': ['examine', 'explore', 'look into'],
    r'\bunderscores\b': ['reinforces', 'confirms', 'supports'],
    r'\belucidates\b': ['clarifies', 'explains', 'sheds light on'],
    r'\bdemonstrates\b': ['shows', 'reveals', 'points to', 'indicates'],
    r'\bfacilitates\b': ['enables', 'supports', 'helps', 'makes possible'],
    r'\bfacilitates\b': ['enables', 'supports', 'helps', 'makes possible'],
}

SPELLING_PAIRS = {
    "analyze": "analyse",
    "analyzes": "analyses",
    "color": "colour",
    "colors": "colours",
    "behavior": "behaviour",
    "behaviors": "behaviours",
    "organize": "organise",
    "organizes": "organises",
    "recognize": "recognise",
    "recognizes": "recognises",
    "realize": "realise",
    "realizes": "realises",
    "optimization": "optimisation",
    "optimize": "optimise",
    "center": "centre",
    "centers": "centres",
    "defense": "defence",
    "license": "licence",
    "traveling": "travelling",
    "fueled": "fuelled",
    "program": "programme",
    "programs": "programmes",
}

# Add reverse mappings
spelling_keys = list(SPELLING_PAIRS.keys())
for k in spelling_keys:
    SPELLING_PAIRS[SPELLING_PAIRS[k]] = k


def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tag to WordNet POS tag."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def get_synonyms(word, pos=None):
    """Get synonyms for a word from WordNet."""
    synonyms = set()
    
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower() and len(synonym.split()) == 1:
                synonyms.add(synonym)
    
    return list(synonyms)


def synonym_swap(text, swap_rate=0.15):
    """
    Replace some words with synonyms to increase lexical variety.
    
    Args:
        text: Input text
        swap_rate: Fraction of eligible words to swap (0.0 to 1.0)
    
    Returns:
        Text with some words replaced by synonyms
    """
    sentences = sent_tokenize(text)
    result_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged = pos_tag(words)
        
        new_words = []
        for word, tag in tagged:
            # Skip protected words and short words
            if word.lower() in PROTECTED_WORDS or len(word) < 4:
                new_words.append(word)
                continue
            
            # Random chance to swap
            if random.random() > swap_rate:
                new_words.append(word)
                continue
            
            # Get WordNet POS
            wn_pos = get_wordnet_pos(tag)
            if wn_pos is None:
                new_words.append(word)
                continue
            
            # Get synonyms
            synonyms = get_synonyms(word.lower(), wn_pos)
            
            if synonyms:
                # Pick a random synonym
                synonym = random.choice(synonyms[:5])  # Limit to top 5 common ones
                
                # Preserve capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                if word.isupper():
                    synonym = synonym.upper()
                    
                new_words.append(synonym)
            else:
                new_words.append(word)
        
        # Reconstruct sentence
        result = ""
        for i, word in enumerate(new_words):
            if i == 0:
                result = word
            elif word in '.,!?;:\'")':
                result += word
            elif new_words[i-1] in '("\'':
                result += word
            else:
                result += " " + word
        
        result_sentences.append(result)
    
    return " ".join(result_sentences)


def add_contractions(text, rate=0.7):
    """
    Convert formal word pairs to contractions.
    
    Args:
        text: Input text
        rate: Probability of converting each instance
    
    Returns:
        Text with contractions added
    """
    result = text
    
    for formal, contraction in CONTRACTIONS.items():
        if random.random() < rate:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(formal), re.IGNORECASE)
            
            def replace_match(match):
                original = match.group(0)
                if original[0].isupper():
                    return contraction.capitalize()
                return contraction
            
            result = pattern.sub(replace_match, result)
    
    return result


def vary_sentence_length(text):
    """
    Add variation to sentence lengths for burstiness.
    Occasionally splits long sentences or combines short ones.
    """
    sentences = sent_tokenize(text)
    result = []
    i = 0
    
    while i < len(sentences):
        sentence = sentences[i]
        words = word_tokenize(sentence)
        
        # Long sentence - maybe split it
        if len(words) > 25 and random.random() < 0.3:
            # Look for a good split point (comma, semicolon, or conjunction)
            split_points = []
            for j, word in enumerate(words):
                if word in [',', ';'] and 8 < j < len(words) - 8:
                    split_points.append(j)
                elif word.lower() in ['and', 'but', 'so', 'yet'] and 8 < j < len(words) - 5:
                    split_points.append(j - 1)
            
            if split_points:
                split_at = random.choice(split_points)
                first_part = words[:split_at + 1]
                second_part = words[split_at + 1:]
                
                # Clean up
                if first_part[-1] == ',':
                    first_part[-1] = '.'
                if second_part and second_part[0].lower() in ['and', 'but', 'so']:
                    second_part[0] = second_part[0].capitalize()
                elif second_part:
                    second_part[0] = second_part[0].capitalize()
                
                result.append(" ".join(first_part))
                result.append(" ".join(second_part))
                i += 1
                continue
        
        # Short consecutive sentences - maybe combine them
        if len(words) < 10 and i + 1 < len(sentences):
            next_sentence = sentences[i + 1]
            next_words = word_tokenize(next_sentence)
            
            if len(next_words) < 12 and random.random() < 0.25:
                # Combine with a connector
                connectors = [" — ", ", and ", "; ", " — plus, "]
                connector = random.choice(connectors)
                
                # Remove period from first sentence
                if sentence.rstrip().endswith('.'):
                    sentence = sentence.rstrip()[:-1]
                
                # Lowercase the start of next sentence
                next_sentence = next_sentence[0].lower() + next_sentence[1:]
                
                combined = sentence + connector + next_sentence
                result.append(combined)
                i += 2
                continue
        
        result.append(sentence)
        i += 1
    
    return " ".join(result)


def inject_informal_elements(text, rate=0.1):
    """
    Add informal transitions and filler words occasionally.
    """
    sentences = sent_tokenize(text)
    result = []
    
    for i, sentence in enumerate(sentences):
        # Skip first sentence
        if i == 0:
            result.append(sentence)
            continue
        
        # Maybe add informal transition at the start
        if random.random() < rate and not sentence.startswith(tuple(INFORMAL_TRANSITIONS)):
            transition = random.choice(INFORMAL_TRANSITIONS)
            # Lowercase the first letter of the original sentence
            sentence = transition + sentence[0].lower() + sentence[1:]
        
        # Maybe add a filler phrase
        elif random.random() < rate * 0.5:
            words = sentence.split()
            if len(words) > 5:
                # Insert filler after 2-4 words
                insert_pos = random.randint(2, min(4, len(words) - 2))
                filler = random.choice(FILLER_PHRASES)
                words.insert(insert_pos, filler)
                sentence = " ".join(words)
        
        result.append(sentence)
    
    return " ".join(result)


def add_sentence_starters(text, rate=0.08):
    """
    Occasionally start sentences with 'And' or 'But' for a more casual feel.
    """
    sentences = sent_tokenize(text)
    result = []
    starters = ['And ', 'But ', 'So ', 'Now, ']
    
    for i, sentence in enumerate(sentences):
        # Skip first couple sentences
        if i < 2:
            result.append(sentence)
            continue
        
        # Check if sentence already starts with these
        first_word = sentence.split()[0].lower() if sentence.split() else ""
        if first_word in ['and', 'but', 'so', 'now', 'however', 'therefore']:
            result.append(sentence)
            continue
        
        if random.random() < rate:
            starter = random.choice(starters)
            sentence = starter + sentence[0].lower() + sentence[1:]
        
        result.append(sentence)
    
    return " ".join(result)


def mix_spellings(text):
    """Randomly swaps US/UK spellings across the text to inject intentional slight inconsistencies."""
    words = word_tokenize(text)
    result_words = []
    
    for word in words:
        lower_word = word.lower()
        if lower_word in SPELLING_PAIRS and random.random() < 0.5:
            # Swap spelling
            swapped = SPELLING_PAIRS[lower_word]
            # Preserve capitalization
            if word.istitle():
                swapped = swapped.capitalize()
            elif word.isupper():
                swapped = swapped.upper()
            result_words.append(swapped)
        else:
            result_words.append(word)
            
    # Re-join intelligently to handle punctuation
    result = " ".join(result_words)
    result = re.sub(r'\s+([.,;:?!\'"])', r'\1', result)
    result = result.replace(" 's", "'s").replace(" n't", "n't")
    return result


def flip_clauses(text):
    """
    Finds standard SVO sentences joined by 'because', 'although', or 'if'
    and flips the clause order. (e.g. "A because B" -> "Because B, A")
    """
    sentences = sent_tokenize(text)
    new_sentences = []
    
    flip_keywords = ["because", "although", "whereas", "while", "since"]
    
    for sentence in sentences:
        # Only try to flip if the sentence is relatively simple (no complex inner punctuation)
        if "," in sentence or ";" in sentence:
            new_sentences.append(sentence)
            continue
            
        words = word_tokenize(sentence.lower())
        found_keyword = None
        
        for k in flip_keywords:
            if k in words and words[0] != k:
                found_keyword = k
                break
                
        if found_keyword and random.random() < 0.7:  # 70% chance to flip if eligible
            # We use regex to split cleanly around the keyword, preserving case of the keyword if any
            pattern = re.compile(rf'\b({found_keyword})\b', re.IGNORECASE)
            parts = pattern.split(sentence, maxsplit=1)
            
            if len(parts) == 3:
                clause1 = parts[0].strip()
                kw = parts[1].strip()
                clause2 = parts[2].strip()
                
                # Remove trailing punctuation from clause2 if present to re-add at the end
                end_punct = ""
                if clause2 and clause2[-1] in ".!?":
                    end_punct = clause2[-1]
                    clause2 = clause2[:-1].strip()
                
                # Lowercase the first letter of clause1
                if clause1:
                    clause1 = clause1[0].lower() + clause1[1:]
                
                # Uppercase the keyword as it's now starting the sentence
                kw = kw.capitalize()
                
                flipped = f"{kw} {clause2}, {clause1}{end_punct}"
                new_sentences.append(flipped)
            else:
                new_sentences.append(sentence)
        else:
            new_sentences.append(sentence)
            
    return " ".join(new_sentences)


def remove_ai_cliches(text):
    """
    Replace common AI-generated cliché phrases with varied human alternatives.
    This targets the exact patterns AI detectors are trained on.
    """
    result = text
    for pattern, replacements in AI_CLICHES.items():
        matches = list(re.finditer(pattern, result))
        for match in reversed(matches):  # reverse to preserve indices
            replacement = random.choice(replacements)
            result = result[:match.start()] + replacement + result[match.end():]
    return result


def humanize_text(text, options=None):
    """
    Apply NLP humanization techniques — casual mode.
    Includes informal transitions, contractions, and casual starters.

    Args:
        text: Input text to humanize
        options: Dict of options to control which techniques to apply

    Returns:
        Humanized text
    """
    if options is None:
        options = {}

    result = text

    # Always strip AI clichés first
    result = remove_ai_cliches(result)

    # Forced new techniques
    result = flip_clauses(result)
    result = mix_spellings(result)

    if options.get('synonyms', True):
        swap_rate = options.get('synonym_rate', 0.15)
        result = synonym_swap(result, swap_rate)

    if options.get('contractions', True):
        result = add_contractions(result)

    # Vary length unconditionally
    result = vary_sentence_length(result)

    if options.get('informal', True):
        rate = options.get('informal_rate', 0.1)
        result = inject_informal_elements(result, rate)

    if options.get('casual_starters', True):
        result = add_sentence_starters(result)

    return result


def humanize_text_academic(text, options=None):
    """
    Apply NLP humanization techniques — academic mode.
    Strips AI clichés and varies structure WITHOUT any informal language.
    Never adds contractions, casual starters, or filler phrases.

    Args:
        text: Input text to humanize
        options: Dict of options to control which techniques to apply

    Returns:
        Humanized text safe for academic use
    """
    if options is None:
        options = {}

    result = text

    # Step 1: Kill AI clichés — highest impact for detector bypass
    result = remove_ai_cliches(result)

    # Step 2: Structural randomization
    result = flip_clauses(result)
    
    # Step 3: Spelling inconsistencies
    result = mix_spellings(result)

    # Step 4: Synonym variation
    swap_rate = options.get('synonym_rate', 0.12)
    result = synonym_swap(result, swap_rate)

    # Step 5: Sentence length variation
    result = vary_sentence_length(result)

    return result


if __name__ == "__main__":
    # Test the humanizer
    test_text = """Artificial intelligence has revolutionized numerous industries. It has enabled unprecedented advancements in healthcare, finance, and transportation. The implementation of machine learning algorithms has facilitated the automation of complex tasks. Furthermore, natural language processing has enhanced human-computer interaction significantly. These technological developments have created new opportunities for businesses and individuals alike."""
    
    print("Original:")
    print(test_text)
    print("\n" + "="*50 + "\n")
    print("Humanized (NLP only):")
    print(humanize_text(test_text))
