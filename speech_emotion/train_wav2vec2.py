"""CLI entry point for Wav2Vec2 MELD training."""

import argparse
from pathlib import Path

from speech_emotion.wav2vec2_model import evaluate_saved_wav2vec2, train_wav2vec2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_root", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.evaluate_only:
        if args.model_dir is None:
            evaluate_saved_wav2vec2(args.audio_root)
        else:
            evaluate_saved_wav2vec2(args.audio_root, model_dir=args.model_dir)
    else:
        train_wav2vec2(args.audio_root)
