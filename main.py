#!/usr/bin/env python3
"""
Main script to transcribe audio and detect its Romance language.
"""

import argparse
import whisper
import re
from detectors.romance_lang_detector import (
    extract_language_clues,
    detect_language,
    extract_dialect_clues,
    DIALECT_PATTERNS
)

def transcribe_audio(model, audio_path):
    """
    Uses Whisper to transcribe the audio file.
    Returns:
      text          (str): full transcription
      detected_lang (str): Whisper's own language code (e.g. 'pt', 'es', 'it')
    """
    result = model.transcribe(audio_path)
    return result["text"], result.get("language")

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a WAV file and detect its Romance language"
    )
    parser.add_argument(
        "audio_file",
        help="Path to the .wav audio file to process"
    )
    args = parser.parse_args()

    # 1. Load Whisper (default “base” model)
    model = whisper.load_model("base")
    print("Loaded Whisper model:", model)

    # 2. Transcribe
    text, whisper_lang = transcribe_audio(model, args.audio_file)
    print("Transcript:", text)
    if whisper_lang:
        print("Whisper language code:", whisper_lang)

    # 3. Regex clue extraction
    lang_scores = extract_language_clues(text)
    print("Language match scores:", lang_scores)

    # 4. Core language prediction
    predicted_lang = detect_language(lang_scores)
    print("Predicted language:", predicted_lang)

    # 5. Dialect-level clues (only for the predicted language)
    dialect_scores = extract_dialect_clues(text).get(predicted_lang, {})
    if dialect_scores:
        print(f"Dialect clue scores for {predicted_lang}:", dialect_scores)
    
        # — Audit: print the actual matches for each Central‐Italian pattern —
    if predicted_lang == "Italian":
        print("Matches per Central-Italian pattern:")
        for pat in DIALECT_PATTERNS["Italian"]["Central"]:
            found = re.findall(pat, text, flags=re.IGNORECASE)
            print(f"  {pat!r}: {found}")


if __name__ == "__main__":
    main()
