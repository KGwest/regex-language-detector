"""
Regex-based Romance language detector.

"""

import re  # Python’s built-in regular-expression library

# Each language maps to a list of regex patterns (clues)
CLUE_PATTERNS = {
    "Portuguese": [r"ção", r"ç"],
    "Spanish":    [r"ñ",   r"ción"],
    "Italian":    [r"\\bche\\b", r"zione"]
}

def extract_language_clues(text: str) -> dict:
    """
    Scan the text for orthographic/morphological features.
    Returns a dict: { "Portuguese": count, "Spanish": count, "Italian": count }.

    """
    scores = {}
    for lang, patterns in CLUE_PATTERNS.items():
        total = 0
        for pat in patterns:
            matches = re.findall(pat, text, flags=re.IGNORECASE)
            total += len(matches)
        scores[lang] = total
    return scores

def detect_language(scores: dict) -> str:
    """
    Chooses the language with the highest count.
    If there’s a tie or zero matches, returns "Unknown".
    
    """
    best = max(scores, key=scores.get)
    best_score = scores[best]
    # tie or no evidence?
    if best_score == 0 or list(scores.values()).count(best_score) > 1:
        return "Unknown"
    return best
