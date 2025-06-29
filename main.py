#!/usr/bin/env python3
"""
Main script to transcribe audio and detect its Romance Language.
"""

impport argparse
import whisper
from detectors.romance_lang_detector import extract_language_clues, detect_language

def transcribe_audio(model, audio_path):
    """ 
    Uses Whisper to transcribe the audio file.
    Returns:
    text         (str): full transcription
    detected_lang(str): Whisper's own language code )ex 'pt', 'es', 'it')     
    """
    result = model.transcribe(audio_path)
    return result["text"], result.get("language")

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a WAV file and detect its Romance language"
    )
    parser.add_arguement(
        "audio_file",
        help="Path to the .wav audio file for processing"
    )
    args = parser.parse_args()

    # 1. Load Whisper (default "base" model)
    model = whisper.load_model("base")
    print("Loaded Whisper model:", model) 

    # 2. Transcribe 
    text, whisper_lang = transcribe_audio(model, args.audio_file)
    print("Transcript:", text)
    if whisper_lang:
        print("Whisper language code:", whisper_lang)
    
    # 3. Regex clue extraction
    scores = extract_language_clues(text)
    print("Regex match scores:", scores)

    # 4. Final prediction 
    predicted = detect_language(scores)
    print("Predicted language:", predicted)

if __name__ == "__main__":
    main()   
