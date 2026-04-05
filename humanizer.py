"""
NLP-based Text Humanizer — Improved
Expanded AI cliché list, more aggressive synonym replacement,
and new inject_human_quirks() for pattern-level human signals.
"""

import os
import random
import re
import nltk

NLTK_DATA_DIR = '/tmp/nltk_data'
if not os.path.exists(NLTK_DATA_DIR):
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.insert(0, NLTK_DATA_DIR)

def ensure_nltk_data():
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
                pass

ensure_nltk_data()

from nltk.tokenize import sent_tokenize, word_tokenize

# ─── PROTECTED WORDS ─────────────────────────────────────────────────────────────
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

# ─── CONTRACTIONS ────────────────────────────────────────────────────────────────
CONTRACTIONS = {
    'do not': "don't", 'does not': "doesn't", 'did not': "didn't",
    'is not': "isn't", 'are not': "aren't", 'was not': "wasn't",
    'were not': "weren't", 'have not': "haven't", 'has not': "hasn't",
    'had not': "hadn't", 'will not': "won't", 'would not': "wouldn't",
    'could not': "couldn't", 'should not': "shouldn't", 'cannot': "can't",
    'can not': "can't", 'it is': "it's", 'it has': "it's",
    'that is': "that's", 'there is': "there's", 'I am': "I'm",
    'I have': "I've", 'I will': "I'll", 'you are': "you're",
    'you have': "you've", 'we are': "we're", 'we have': "we've",
    'they are': "they're", 'they have': "they've", 'he is': "he's",
    'she is': "she's", 'let us': "let's",
}

# ─── INFORMAL TRANSITIONS ────────────────────────────────────────────────────────
INFORMAL_TRANSITIONS = [
    "Plus, ", "Thing is, ", "Look, ", "Honestly, ",
    "The reality is, ", "What's more, ", "On top of that, ",
    "Here's the thing: ", "Worth noting: ", "Interestingly, ",
]

FILLER_PHRASES = [
    "basically", "essentially", "arguably", "in practice",
    "for the most part", "at least in principle", "to some degree",
]

# ─── EXPANDED AI CLICHÉS ─────────────────────────────────────────────────────────
# These are the exact phrases AI detectors are trained on.
# Each maps to multiple alternatives so consecutive passes vary.
AI_CLICHES = {
    # Transition words
    r'\bFurthermore\b': ['That said,', 'Beyond this,', 'Alongside this,', 'On top of that,', 'What is more,'],
    r'\bfurthermore\b': ['beyond this,', 'alongside this,', 'on top of that,', 'what is more,', 'relatedly,'],
    r'\bMoreover\b': ['Still,', 'On reflection,', 'Worth adding,', 'Of note,', 'Alongside this,'],
    r'\bmoreover\b': ['still,', 'on reflection,', 'worth adding,', 'alongside this,'],
    r'\bAdditionally\b': ['Yet,', 'To this end,', 'Along these lines,', 'In turn,', 'Beyond that,'],
    r'\badditionally\b': ['yet,', 'to this end,', 'along these lines,', 'in turn,', 'beyond that,'],
    r'\bIn conclusion\b': ['In sum,', 'On balance,', 'Taken together,', 'All things considered,', 'Stepping back,'],
    r'\bin conclusion\b': ['in sum,', 'on balance,', 'taken together,', 'all things considered,'],
    r'\bIn summary\b': ['In short,', 'Briefly,', 'To draw this together,', 'Looking at the whole,'],
    r'\bin summary\b': ['in short,', 'briefly,', 'to draw this together,'],
    r'\bTo summarize\b': ['In brief,', 'Stepping back,', 'The upshot is,', 'To pull this together,'],
    r'\bto summarize\b': ['in brief,', 'stepping back,', 'the upshot is,'],
    r'\bIn order to\b': ['To', 'So as to', 'With the goal of', 'Aiming to'],
    r'\bin order to\b': ['to', 'so as to', 'with the goal of', 'aiming to'],
    r'\bHowever\b': ['Even so,', 'That said,', 'Yet,', 'Still,', 'All the same,'],
    r'\bhowever\b': ['even so,', 'that said,', 'yet,', 'still,', 'all the same,'],
    r'\bTherefore\b': ['As a result,', 'So,', 'Hence,', 'This is why', 'Consequently,'],
    r'\btherefore\b': ['as a result,', 'so,', 'hence,', 'this is why', 'consequently,'],
    r'\bConsequently\b': ['As a result,', 'So,', 'This led to', 'It follows that'],
    r'\bconsequently\b': ['as a result,', 'so,', 'this led to', 'it follows that'],
    r'\bNevertheless\b': ['Even so,', 'All the same,', 'That said,', 'Despite this,'],
    r'\bnevertheless\b': ['even so,', 'all the same,', 'that said,', 'despite this,'],
    r'\bNotwithstanding\b': ['Despite this,', 'Even so,', 'That said,'],
    r'\bnotwithstanding\b': ['despite this,', 'even so,', 'that said,'],

    # Importance phrases
    r'\bIt is important to note that\b': ['It is worth considering that', 'Crucially,', 'One should bear in mind that', 'Notably,'],
    r'\bit is important to note that\b': ['it is worth considering that', 'crucially,', 'one should bear in mind that'],
    r'\bIt is worth noting that\b': ['Of particular relevance,', 'Here,', 'Pertinently,'],
    r'\bit is worth noting that\b': ['of particular relevance,', 'here,', 'pertinently,'],
    r'\bIt is important to\b': ['It matters to', 'One should', 'The priority is to'],
    r'\bit is important to\b': ['it matters to', 'one should', 'the priority is to'],
    r'\bIt is essential to\b': ['One must', 'The key is to', 'Critically, one should'],
    r'\bit is essential to\b': ['one must', 'the key is to', 'critically, one should'],

    # Flagged adverbs/adjectives
    r'\bNotably\b': ['Of interest,', 'Tellingly,', 'Here,', 'Strikingly,'],
    r'\bnotably\b': ['of interest,', 'tellingly,', 'strikingly,', 'here,'],
    r'\bSignificantly\b': ['Strikingly,', 'To a marked degree,', 'Tellingly,'],
    r'\bsignificantly\b': ['strikingly,', 'to a marked degree,', 'tellingly,', 'materially,'],
    r'\bSubstantially\b': ['Considerably,', 'By a wide margin,', 'Markedly,'],
    r'\bsubstantially\b': ['considerably,', 'by a wide margin,', 'markedly,'],
    r'\bCrucially\b': ['What matters here is that', 'Centrally,', 'The key point is that'],
    r'\bcrucially\b': ['what matters here is that', 'centrally,', 'the key point is that'],

    # AI-tell phrases
    r'\bplays a crucial role\b': ['is central to', 'matters greatly for', 'underpins', 'drives'],
    r'\bplays a significant role\b': ['bears heavily on', 'shapes', 'is central to', 'influences'],
    r'\bplays an important role\b': ['has a hand in', 'contributes to', 'shapes'],
    r'\bhas the potential to\b': ['can', 'may well', 'could', 'stands to'],
    r'\bhave the potential to\b': ['can', 'may well', 'could', 'stand to'],
    r'\bIt is clear that\b': ['Evidently,', 'The data suggest that', 'This points to', 'Plainly,'],
    r'\bit is clear that\b': ['evidently,', 'the evidence suggests that', 'this points to', 'plainly,'],
    r'\bIt becomes clear that\b': ['One sees that', 'Evidently,', 'It emerges that'],
    r'\bit becomes clear that\b': ['one sees that', 'evidently,', 'it emerges that'],
    r'\bIt is evident that\b': ['Plainly,', 'Evidently,', 'One can see that'],
    r'\bit is evident that\b': ['plainly,', 'evidently,', 'one can see that'],
    r'\bThis highlights\b': ['This points to', 'This reveals', 'This confirms'],
    r'\bthis highlights\b': ['this points to', 'this reveals', 'this confirms'],
    r'\bThis demonstrates\b': ['This shows', 'This reveals', 'This indicates'],
    r'\bthis demonstrates\b': ['this shows', 'this reveals', 'this indicates'],
    r'\bThis suggests\b': ['This implies', 'This hints that', 'The implication is'],
    r'\bthis suggests\b': ['this implies', 'this hints that', 'the implication is'],
    r'\bThis underscores\b': ['This reinforces', 'This confirms', 'This backs'],
    r'\bthis underscores\b': ['this reinforces', 'this confirms', 'this backs'],

    # Overused AI vocabulary
    r'\boverall\b': ['on the whole', 'in total', 'across the board', 'broadly speaking'],
    r'\bOverall\b': ['On the whole,', 'In total,', 'Broadly speaking,', 'Across the board,'],
    r'\boverall,\b': ['on the whole,', 'in total,', 'broadly speaking,'],
    r'\boverarching\b': ['broader', 'general', 'governing', 'central'],
    r'\brobust\b': ['strong', 'reliable', 'solid', 'sound', 'dependable'],
    r'\bseamless\b': ['smooth', 'fluid', 'uninterrupted', 'frictionless'],
    r'\bseamlessly\b': ['smoothly', 'without friction', 'fluidly'],
    r'\bdelve\b': ['examine', 'explore', 'look into', 'dig into'],
    r'\bdelves\b': ['examines', 'explores', 'looks into'],
    r'\bunderscores\b': ['reinforces', 'confirms', 'supports', 'backs'],
    r'\belsucidates\b': ['clarifies', 'explains', 'sheds light on'],
    r'\belucidates\b': ['clarifies', 'explains', 'sheds light on'],
    r'\bdemonstrates\b': ['shows', 'reveals', 'points to', 'indicates'],
    r'\bdemonstrate\b': ['show', 'reveal', 'point to', 'indicate'],
    r'\bfacilitates\b': ['enables', 'supports', 'helps', 'makes possible'],
    r'\bfacilitate\b': ['enable', 'support', 'help', 'allow'],
    r'\butilizes\b': ['uses', 'employs', 'draws on'],
    r'\butilize\b': ['use', 'employ', 'draw on'],
    r'\butilization\b': ['use', 'usage', 'application'],
    r'\benables\b': ['lets', 'allows', 'makes it possible for', 'opens the door to'],
    r'\benable\b': ['let', 'allow', 'make it possible to'],
    r'\bensures\b': ['guarantees', 'makes certain', 'keeps'],
    r'\bensure\b': ['guarantee', 'make certain', 'keep'],
    r'\benhances\b': ['improves', 'boosts', 'sharpens', 'strengthens'],
    r'\benhance\b': ['improve', 'boost', 'sharpen', 'strengthen'],
    r'\bleverages\b': ['uses', 'draws on', 'taps into', 'builds on'],
    r'\bleverage\b': ['use', 'draw on', 'tap into', 'build on'],
    r'\bmitigates\b': ['reduces', 'limits', 'dampens', 'eases'],
    r'\bmitigate\b': ['reduce', 'limit', 'dampen', 'ease'],
    r'\boptimizes\b': ['improves', 'fine-tunes', 'refines'],
    r'\boptimize\b': ['improve', 'fine-tune', 'refine'],
    r'\bparadigm\b': ['model', 'framework', 'approach', 'way of thinking'],
    r'\blandscape\b': ['field', 'environment', 'arena', 'terrain'],
    r'\becosystem\b': ['environment', 'network', 'system', 'web of relationships'],
    r'\bstakeholders\b': ['those involved', 'the parties concerned', 'the people affected'],
    r'\bsynergies\b': ['combined effects', 'overlaps', 'mutual benefits'],
    r'\bharnessing\b': ['using', 'drawing on', 'tapping into'],
    r'\bharness\b': ['use', 'draw on', 'tap into'],
    r'\bpivotal\b': ['central', 'key', 'decisive', 'critical'],
    r'\btransformative\b': ['significant', 'far-reaching', 'reshaping'],
    r'\bgroundbreaking\b': ['novel', 'unprecedented', 'new ground'],
    r'\bcutting-edge\b': ['leading', 'advanced', 'at the forefront'],
    r'\bstate-of-the-art\b': ['advanced', 'current', 'leading'],
    r'\binnovative\b': ['novel', 'new', 'fresh', 'original'],
    r'\binnovation\b': ['novelty', 'new development', 'advance', 'new approach'],
    r'\btailored\b': ['customised', 'adapted', 'fitted', 'designed'],
    r'\bcomprehensive\b': ['thorough', 'wide-ranging', 'full', 'broad'],
    r'\brobust\b': ['strong', 'solid', 'reliable'],
    r'\bnuanced\b': ['subtle', 'layered', 'complex', 'detailed'],
    r'\bvalidate\b': ['confirm', 'verify', 'check', 'back up'],
    r'\bvalidates\b': ['confirms', 'verifies', 'backs up'],
    r'\bstreamline\b': ['simplify', 'speed up', 'reduce friction in'],
    r'\bstreamlines\b': ['simplifies', 'speeds up', 'cuts friction in'],
    r'\bscalable\b': ['able to grow', 'flexible', 'adaptable'],
    r'\bscalability\b': ['growth capacity', 'flexibility', 'adaptability'],
    r'\bholistic\b': ['whole-system', 'broad', 'full-picture', 'integrated'],
    r'\bproactive\b': ['ahead of the curve', 'anticipatory', 'forward-looking'],
    r'\bsustainable\b': ['lasting', 'long-term', 'durable', 'viable over time'],
    r'\baccelerates\b': ['speeds up', 'hastens', 'pushes forward'],
    r'\baccelerating\b': ['speeding up', 'hastening', 'pushing forward'],
    r'\bprecision\b': ['accuracy', 'exactness', 'care'],
    r'\binsightful\b': ['perceptive', 'revealing', 'illuminating'],
    r'\bimpactful\b': ['effective', 'consequential', 'meaningful', 'significant in effect'],
    r'\bsophisticated\b': ['complex', 'advanced', 'developed', 'nuanced'],
    r'\bseamless integration\b': ['smooth connection', 'clean combination', 'unified approach'],
    r'\breal-world\b': ['practical', 'actual', 'on-the-ground', 'in practice'],
    r'\bin the realm of\b': ['in', 'within', 'in the area of'],
    r'\bin the context of\b': ['in', 'within', 'given'],
    r'\bin terms of\b': ['regarding', 'when it comes to', 'on', 'for'],
    r'\bwith respect to\b': ['regarding', 'on', 'about', 'concerning'],
    r'\bwith regard to\b': ['regarding', 'on', 'about', 'concerning'],
    r'\bdue to the fact that\b': ['because', 'given that', 'since'],
    r'\bthe fact that\b': ['that', 'the reality that', 'the point that'],
    r'\ba wide range of\b': ['many', 'various', 'a variety of', 'numerous'],
    r'\ba wide variety of\b': ['many', 'various', 'diverse', 'numerous'],
    r'\ba number of\b': ['several', 'a handful of', 'a few', 'some'],
    r'\ba plethora of\b': ['many', 'a great many', 'no shortage of'],
    r'\bvast majority\b': ['most', 'the bulk of', 'nearly all'],
    r'\bsignificant amount\b': ['substantial', 'considerable', 'much'],
    r'\bin light of\b': ['given', 'considering', 'in view of'],
    r'\bin the light of\b': ['given', 'considering', 'in view of'],
}

# ─── US/UK SPELLING PAIRS ────────────────────────────────────────────────────────
SPELLING_PAIRS = {
    "analyze": "analyse", "analyzes": "analyses", "color": "colour",
    "colors": "colours", "behavior": "behaviour", "behaviors": "behaviours",
    "organize": "organise", "organizes": "organises", "recognize": "recognise",
    "recognizes": "recognises", "realize": "realise", "realizes": "realises",
    "optimization": "optimisation", "optimize": "optimise", "center": "centre",
    "centers": "centres", "defense": "defence", "license": "licence",
    "traveling": "travelling", "fueled": "fuelled", "program": "programme",
    "programs": "programmes", "labeled": "labelled", "labeling": "labelling",
    "modeling": "modelling", "modeled": "modelled", "fulfill": "fulfil",
    "fulfills": "fulfils", "canceled": "cancelled", "canceling": "cancelling",
    "enrollment": "enrolment", "focused": "focussed", "gray": "grey",
    "grays": "greys", "humor": "humour", "honor": "honour", "flavor": "flavour",
    "neighbor": "neighbour", "neighbors": "neighbours",
}
# Add reverse mappings
for k in list(SPELLING_PAIRS.keys()):
    SPELLING_PAIRS[SPELLING_PAIRS[k]] = k


def remove_ai_cliches(text: str) -> str:
    """Replace AI cliché phrases with varied human alternatives."""
    result = text
    for pattern, replacements in AI_CLICHES.items():
        matches = list(re.finditer(pattern, result))
        for match in reversed(matches):
            replacement = random.choice(replacements)
            result = result[:match.start()] + replacement + result[match.end():]
    return result


def add_contractions(text: str, rate: float = 0.7) -> str:
    """Convert formal word pairs to contractions."""
    result = text
    for formal, contraction in CONTRACTIONS.items():
        if random.random() < rate:
            pattern = re.compile(re.escape(formal), re.IGNORECASE)
            def replace_match(match, c=contraction):
                return c.capitalize() if match.group(0)[0].isupper() else c
            result = pattern.sub(replace_match, result)
    return result


def vary_sentence_length(text: str) -> str:
    """Split long sentences or merge short ones for burstiness."""
    sentences = sent_tokenize(text)
    result = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        words = word_tokenize(sentence)

        # Long sentence — maybe split
        if len(words) > 25 and random.random() < 0.35:
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
                if first_part and first_part[-1] == ',':
                    first_part[-1] = '.'
                if second_part:
                    second_part[0] = second_part[0].capitalize()
                result.append(" ".join(first_part))
                result.append(" ".join(second_part))
                i += 1
                continue

        # Short consecutive sentences — maybe merge
        if len(words) < 10 and i + 1 < len(sentences):
            next_words = word_tokenize(sentences[i + 1])
            if len(next_words) < 12 and random.random() < 0.3:
                connectors = [" — ", "; ", ", and ", " — though "]
                connector = random.choice(connectors)
                s = sentence.rstrip()
                if s.endswith('.'):
                    s = s[:-1]
                next_s = sentences[i + 1][0].lower() + sentences[i + 1][1:]
                result.append(s + connector + next_s)
                i += 2
                continue

        result.append(sentence)
        i += 1

    return " ".join(result)


def mix_spellings(text: str) -> str:
    """Swap US/UK spellings randomly to inject subtle inconsistencies."""
    words = word_tokenize(text)
    result_words = []
    for word in words:
        lower = word.lower()
        if lower in SPELLING_PAIRS and random.random() < 0.5:
            swapped = SPELLING_PAIRS[lower]
            if word.istitle():
                swapped = swapped.capitalize()
            elif word.isupper():
                swapped = swapped.upper()
            result_words.append(swapped)
        else:
            result_words.append(word)
    result = " ".join(result_words)
    result = re.sub(r'\s+([.,;:?!\'"])', r'\1', result)
    result = result.replace(" 's", "'s").replace(" n't", "n't")
    return result


def flip_clauses(text: str) -> str:
    """Flip subordinate clause order for structural variation."""
    sentences = sent_tokenize(text)
    new_sentences = []
    flip_keywords = ["because", "although", "whereas", "while", "since"]

    for sentence in sentences:
        if "," in sentence or ";" in sentence:
            new_sentences.append(sentence)
            continue
        words = word_tokenize(sentence.lower())
        found_keyword = None
        for k in flip_keywords:
            if k in words and words[0] != k:
                found_keyword = k
                break
        if found_keyword and random.random() < 0.65:
            pattern = re.compile(rf'\b({found_keyword})\b', re.IGNORECASE)
            parts = pattern.split(sentence, maxsplit=1)
            if len(parts) == 3:
                clause1 = parts[0].strip()
                kw = parts[1].strip()
                clause2 = parts[2].strip()
                end_punct = ""
                if clause2 and clause2[-1] in ".!?":
                    end_punct = clause2[-1]
                    clause2 = clause2[:-1].strip()
                if clause1:
                    clause1 = clause1[0].lower() + clause1[1:]
                kw = kw.capitalize()
                new_sentences.append(f"{kw} {clause2}, {clause1}{end_punct}")
                continue
        new_sentences.append(sentence)

    return " ".join(new_sentences)


def inject_human_quirks(text: str, academic: bool = True) -> str:
    """
    NLP-level injection of human writing patterns.
    Adds parenthetical asides, hedging language, and abrupt short sentences.
    Works alongside the AI-based humanity_injection() for multi-level coverage.
    """
    sentences = sent_tokenize(text)
    result = []

    # Hedging phrases for academic mode
    academic_hedges = [
        "it seems,", "arguably,", "to some degree,",
        "at least in principle,", "in most cases,",
    ]
    # Parenthetical inserts
    parentheticals = [
        "(though not universally)",
        "(with some exceptions)",
        "(if only partially)",
        "(at least in this context)",
        "(a distinction worth keeping)",
        "(the difference is subtle but real)",
    ]

    for i, sentence in enumerate(sentences):
        words = sentence.split()

        # Add hedging to long assertions (not first/last sentence)
        if 1 < i < len(sentences) - 1 and len(words) > 15 and random.random() < 0.18:
            hedge = random.choice(academic_hedges if academic else [
                "honestly,", "in practice,", "for the most part,", "as far as it goes,"
            ])
            # Insert after first 2-3 words
            insert_pos = min(3, len(words) - 2)
            words.insert(insert_pos, hedge)
            sentence = " ".join(words)

        # Add parenthetical to some medium-length sentences
        elif len(words) > 12 and len(words) < 30 and random.random() < 0.12:
            paren = random.choice(parentheticals)
            # Insert before the last 4 words
            insert_pos = len(words) - 4
            if insert_pos > 4:
                words.insert(insert_pos, paren)
                sentence = " ".join(words)

        result.append(sentence)

        # Occasionally add a short punchy follow-up sentence
        if i < len(sentences) - 1 and len(words) > 25 and random.random() < 0.10:
            punchy = random.choice([
                "That distinction matters.",
                "Worth sitting with.",
                "The implications run deeper than they first appear.",
                "This is not a minor point.",
                "The gap is real.",
            ])
            result.append(punchy)

    return " ".join(result)


def inject_informal_elements(text: str, rate: float = 0.1) -> str:
    """Add informal transitions and filler words (casual mode)."""
    sentences = sent_tokenize(text)
    result = []
    for i, sentence in enumerate(sentences):
        if i == 0:
            result.append(sentence)
            continue
        if random.random() < rate and not sentence.startswith(tuple(INFORMAL_TRANSITIONS)):
            transition = random.choice(INFORMAL_TRANSITIONS)
            sentence = transition + sentence[0].lower() + sentence[1:]
        elif random.random() < rate * 0.5:
            words = sentence.split()
            if len(words) > 5:
                pos = random.randint(2, min(4, len(words) - 2))
                words.insert(pos, random.choice(FILLER_PHRASES))
                sentence = " ".join(words)
        result.append(sentence)
    return " ".join(result)


def add_sentence_starters(text: str, rate: float = 0.08) -> str:
    """Start some sentences with And/But/So for casual feel."""
    sentences = sent_tokenize(text)
    result = []
    starters = ['And ', 'But ', 'So ', 'Now, ']
    for i, sentence in enumerate(sentences):
        if i < 2:
            result.append(sentence)
            continue
        first_word = sentence.split()[0].lower() if sentence.split() else ""
        if first_word in ['and', 'but', 'so', 'now', 'however', 'therefore']:
            result.append(sentence)
            continue
        if random.random() < rate:
            starter = random.choice(starters)
            sentence = starter + sentence[0].lower() + sentence[1:]
        result.append(sentence)
    return " ".join(result)


def humanize_text(text: str, options: dict = None) -> str:
    """NLP humanization — casual mode."""
    if options is None:
        options = {}
    result = remove_ai_cliches(text)
    result = flip_clauses(result)
    result = mix_spellings(result)
    if options.get('contractions', True):
        result = add_contractions(result)
    result = vary_sentence_length(result)
    if options.get('informal', True):
        result = inject_informal_elements(result, options.get('informal_rate', 0.1))
    if options.get('casual_starters', True):
        result = add_sentence_starters(result)
    return result


def humanize_text_academic(text: str, options: dict = None) -> str:
    """NLP humanization — academic mode. No contractions or casual language."""
    if options is None:
        options = {}
    result = remove_ai_cliches(text)
    result = flip_clauses(result)
    result = mix_spellings(result)
    result = inject_human_quirks(result, academic=True)
    result = vary_sentence_length(result)
    return result


if __name__ == "__main__":
    test = """Artificial intelligence has revolutionized numerous industries. Furthermore, it has enabled unprecedented advancements. The implementation of machine learning algorithms has facilitated the automation of complex tasks. Additionally, natural language processing has enhanced human-computer interaction significantly. These transformative developments have created new opportunities for businesses and individuals alike. It is important to note that this is a significant and impactful trend."""
    print("Academic humanized:")
    print(humanize_text_academic(test))