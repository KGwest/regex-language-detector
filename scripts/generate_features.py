#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

import glob
import whisper
import pandas as pd

from detectors.romance_lang_detector import extract_language_clues

def infer_label_from_filename(fname: str) -> str:
    """
    Extract your ground-truth label from the filename convention,
    e.g. 'mariapombo_castsp_sp_2.wav' → 'Spanish'
    Adjust this logic to match your naming scheme.
    """
    base = os.path.basename(fname).lower()
    if "_pt_" in base:
        return "Portuguese"
    if "_sp_" in base:
        return "Spanish"
    if "_it_" in base:
        return "Italian"
    return "Unknown"

def main():
    # 1. Load Whisper
    model = whisper.load_model("base")

    records = []
    for wav_path in glob.glob("data/samples/*.wav"):
        # 2a. Transcribe
        result = model.transcribe(wav_path)
        text = result["text"]

        # 2b. Extract regex clue counts
        scores = extract_language_clues(text)

        # 2c. Build a record dict
        rec = {
            "filename": os.path.basename(wav_path),
            "transcript": text.strip(),
            **scores,  # merges in Spanish, Portuguese, Italian counts
            "label": infer_label_from_filename(wav_path)
        }
        records.append(rec)

        print(f"Processed {rec['filename']}: {rec['label']} → {scores}")

    # 3. Create DataFrame and save CSV
    df = pd.DataFrame(records)
    out_path = "data/features.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved features to {out_path}")

if __name__ == "__main__":
    main()
