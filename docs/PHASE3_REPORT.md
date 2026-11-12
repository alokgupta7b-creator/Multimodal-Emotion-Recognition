# Phase 3 DistilBERT Text Emotion Recognition

## Scope

This phase adds a separate DistilBERT implementation for the seven MELD
emotions. It does not replace the TF-IDF + Logistic Regression baseline and
does not add speech, vision, fusion or API functionality.

## Architecture

- Checkpoint: `distilbert-base-uncased`.
- Dynamic padding with a maximum sequence length of 128.
- Seven-way sequence classification using the canonical alphabetical MELD
  mapping: anger, disgust, fear, joy, neutral, sadness, surprise.
- AdamW optimization, gradient accumulation, optional CUDA FP16 and optional
  gradient checkpointing.
- Dev macro F1 controls best-checkpoint selection and early stopping.
- The test split is loaded only for the final evaluation after training.
- Artifacts are stored below `models/text_emotion/distilbert/best`.

## Hardware and execution status

The machine reports an NVIDIA GeForce RTX 4050 Laptop GPU with 6,141 MiB and
driver 592.82. The installed PyTorch package is 2.6.0+cu124, but importing it
initially failed with Windows Application Control error `WinError 4551` while
loading `torch\lib\shm.dll`; disabling Hugging Face Xet allowed the subsequent
training process to complete on CUDA. A separate initial probe also reported
duplicate OpenMP runtime initialization.

Actual training completed on CUDA using the RTX 4050. The best checkpoint was
selected at epoch 3 after four configured epochs, with 275.73 seconds of
training time. The saved artifact is approximately 268.6 MB, and measured
five-run GPU inference latency averaged 54.89 ms per sample after model load.

The implementation also provides CPU fallback. A sample GPU inference returned
`joy` with confidence 0.92184; all seven probabilities were finite,
non-negative and summed to 1.0.

## Results

On the held-out MELD test split (2,610 samples):

| Metric | TF-IDF baseline | DistilBERT |
|---|---:|---:|
| Accuracy | 0.5088 | 0.6310 |
| Macro precision | 0.3591 | 0.4977 |
| Macro recall | 0.3846 | 0.3998 |
| Macro F1 | 0.3683 | 0.4155 |
| Weighted F1 | 0.5216 | 0.6077 |

These are actual executions from the same MELD test annotations. DistilBERT
improved accuracy, macro F1 and weighted F1 in this run; minority-class
performance remains uneven, especially for fear and disgust.

## Validation

The preparation tests verify all three MELD split sizes and canonical labels:
9,989 train, 1,109 dev and 2,610 test records. Full training results,
latency, model size and comparison metrics are recorded above and in
`training_metadata.json`.

## Limitations

The current environment prevents downloading/loading the pretrained checkpoint
through PyTorch. The repository does not fabricate a checkpoint or evaluation
metrics. The pipeline uses a custom PyTorch loop rather than Hugging Face
Trainer to keep memory usage explicit for a six-gigabyte GPU.
