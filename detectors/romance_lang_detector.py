"""
Regex-based Romance language & dialect detector.
"""
#  ***Tightening Central Italian Accent on 7/1/2025****

import re
from typing import Dict

# ─── CORE LANGUAGE PATTERNS ────────────────────────────────────────────────────

CLUE_PATTERNS = {
    "Portuguese": [r"ção", r"ç"],
    "Spanish":    [r"ñ",   r"ción"],
    "Italian":    [r"\bche\b", r"zione"]
}

def extract_language_clues(text: str) -> Dict[str,int]:
    """
    Scan the text for orthographic/morphological features.
    Returns a dict: { "Portuguese": count, "Spanish": count, "Italian": count }.
    """
    scores = {}
    for lang, patterns in CLUE_PATTERNS.items():
        total = 0
        for pat in patterns:
            total += len(re.findall(pat, text, flags=re.IGNORECASE))
        scores[lang] = total
    return scores

def detect_language(scores: Dict[str,int]) -> str:
    """
    Chooses the language with the highest count.
    If there’s a tie or zero matches, returns "Unknown".
    """
    best = max(scores, key=scores.get)
    best_score = scores[best]
    if best_score == 0 or list(scores.values()).count(best_score) > 1:
        return "Unknown"
    return best

# ─── DIALECT PATTERNS ──────────────────────────────────────────────────────────

DIALECT_PATTERNS = {
    # Only applied when core detector says “Spanish”
    "Spanish": {
        "Andalusian": [
            r"(?<=\w)ao\b",   # “-ado” → “-ao”
            r"[aeiou]h\b",    # s-aspiration (estas → estah)
        ],
        "Caribbean": [
            r"[aeiou]l\b",    # r → l at word end (comer → comel)
            r"[aeiou]r\b",    # l → r at word end (caldo → cardo)
            r"[aeiou]h\b",    # s-aspiration
            r"\bpa\b",        # para → pa
            r"\bustedes\b",   # ustedes pronoun
        ],
        "CentralAmerican": [
            r"\bvos\b",       # voseo pronoun
            r"\bustedes\b",   # ustedes instead of vosotros
        ]
    },
    # Only applied when core detector says “Portuguese”
    "Portuguese": {
        "European": [ r"\bol\b" ],     # silent final “l”
        "Brazilian": [ r"ão\b", r"ãe\b" ],
        "African":   [ r"\bng\b" ],    # typical loanword clusters
    },
    # Only applied when core detector says “Italian”
    "Italian": {
        "Northern": [ r"\bpiz[za]\b" ],       # pizza /pidza/
        "Central": [  r"(?<=[aeiou])h(?=[aeiou])",    # catches eha, oho, etc.
                      r"\bporhè\b",                   # if “perché” shows as “porhè”
                    # …any other concrete orthographic differences?
                    ],
        "Central": [
                    # Tuscan “gorgia”: h between vowels (e.g. eha for eta) 
                   #r"(?<=[aeiou])h(?=[aeiou])",
                    # Vowel-drop observed in “ondra” → “ondr”
                   #r"\bondr\b",
                    # Any cafè/cafe spelling variation 
                    r"caf[èe]\b",  ],
        "Southern": [ r"sciòla", r"\bra\b" ],  # local lexemes (sciòla, ’a)
    }
}

def extract_dialect_clues(text: str) -> Dict[str,Dict[str,int]]:
    """
    Run through DIALECT_PATTERNS for each core language.
    Returns nested dict, e.g.
      {
        "Spanish": { "Andalusian": 2, "Caribbean": 1, "CentralAmerican": 0 },
        "Portuguese": { … },
        "Italian": { … }
      }
    """
    all_scores = {}
    for lang, dialects in DIALECT_PATTERNS.items():
        scores = {}
        for dialect, patterns in dialects.items():
            total = 0
            for pat in patterns:
                total += len(re.findall(pat, text, flags=re.IGNORECASE))
            scores[dialect] = total
        all_scores[lang] = scores
    return all_scores

# ─── EXAMPLE USAGE ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = "Apurao, ¿cómo está? Vos tenés razón, compa!"
    lang_scores = extract_language_clues(sample)
    print("Lang scores:", lang_scores)
    chosen = detect_language(lang_scores)
    print("Detected language:", chosen)
    dialect_scores = extract_dialect_clues(sample)
    print("Dialect scores:", dialect_scores.get(chosen, {}))
    print([m.group(0) for m in re.finditer(r"(?<=\w)[ptc](?=\w)", text)])

