# Regex-Powered Romance Language Detector

**Author:** Kezia Grace West  
**Repo:** `regex-language-detector`

---

## Overview

A lightweight, explainable pipeline for detecting Romance languages (and dialects) in spoken audio.  
- **ASR**: Uses OpenAI Whisper to transcribe `.wav` files.  
- **Rule-Based Features**: Counts regex-based orthographic/morphological clues (`ñ`, `ción`, `che`, `zione`, `ção`, etc.).  
- **Prototype ML**: (Planned) Train a Multinomial Naive Bayes model on these counts for probabilistic classification.

---

## What’s implemented

1. **Project scaffold**  
   - `main.py`: CLI entrypoint (transcribe → extract → predict).  
   - `detectors/romance_lang_detector.py`: `extract_language_clues()` + rule-based `detect_language()`.  
   - `train_naive_bayes.py`: stub for training on regex-count features.

2. **Sample data pipeline**  
   - `scripts/generate_features.py`:  
     - Loads all `data/samples/*.wav`  
     - Runs Whisper transcription  
     - Extracts regex counts  
     - Builds and writes `data/features.csv`

---

## Quickstart

```bash
# 1. Clone & enter project
git clone https://github.com/KGwest/regex-language-detector.git
cd regex-language-detector

# 2. Create & activate venv
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows (Git Bash)
source venv/Scripts/activate

# 3. Install deps
pip install -r requirements.txt

# 4. (If needed) install FFmpeg on your PATH

# 5. Transcribe & detect a single file
python main.py data/samples/sofigioffreda_northit_it_1.wav

# 6. Generate feature CSV for ML
python scripts/generate_features.py

# 7. Train & evaluate Naive Bayes
python train_naive_bayes.py data/features.csv

```
## Dependencies
  -Python 3.8+
  -openai-whisper — Whisper ASR (PyPI)
  -scikit-learn — MultinomialNB classifier & train/test utilities (PyPI)
  -pandas — DataFrame creation & CSV I/O (PyPI)
  -ffmpeg — System‐level audio pre‐processing (ensure ffmpeg is on your PATH)
  -Python’s built-in re — Regex matching for all language & dialect clues
  -All other package versions are pinned in requirements.txt. Install everything with:

```bash

pip install -r requirements.txt
