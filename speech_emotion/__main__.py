"""
speech_emotion.__main__
=======================
CLI entry point for speech emotion preprocessing and feature extraction.

Run with:
    python -m speech_emotion [OPTIONS]
"""

import sys
from speech_emotion.preprocess_audio import main

if __name__ == "__main__":
    sys.exit(main())
