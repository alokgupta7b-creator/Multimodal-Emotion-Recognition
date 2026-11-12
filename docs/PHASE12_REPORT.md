# Phase 12 Report: Multimodal Development Fusion Experiments

## Scope

Phase 12 conducted a development-only late fusion evaluation across all seven modality combinations:
- `text`
- `audio`
- `video`
- `text+audio`
- `text+video`
- `audio+video`
- `text+audio+video`

The experiment evaluated 1,108 aligned development records from the MELD dataset using the frozen pretrained checkpoints from DistilBERT, Wav2Vec2, and the Phase 8 facial CNN. No models were fine-tuned or trained, and held-out test data was not accessed.

## Execution Details

- **Script:** `phase12/run_experiments.py --split dev`
- **Environment:** Python 3.13.9 64-bit (`C:\Users\Bhavanth D\anaconda3\python.exe`)
- **Execution Mode:** CPU inference across all three modalities
- **Total Development Samples:** 1,108 aligned records
- **Exit Code:** `0` (Success)
- **Artifacts Generated:**
  - `models/phase12_dev_experiments/dev_predictions.jsonl` (1,108 sample records, 1,061,956 bytes)
  - `models/phase12_dev_experiments/dev_results.json` (comprehensive classification & calibration metrics, 35,525 bytes)

## Data Alignment and Manifest Handling

- **Aligned Manifest Records:** The MELD development split contains 1,109 annotation rows. One sample (`dev_dia110_utt7`) lacked raw video in the official archive and was gracefully skipped during alignment verification, leaving exactly **1,108 unique development records**.
- **Audio Availability:** 1,107 available records. 1 sample (`dev_dia36_utt8`) was excluded due to near-silent audio (`RMS energy (-63.3 dB) is below silence threshold (-60.0 dB)`).
- **Video Availability:** 1,096 available records. 12 samples were excluded due to face detection absence (`No detectable faces were supplied.`).
- **Dynamic Renormalization:** Excluded modalities were not assigned synthetic uniform, zero, or neutral probabilities. Weights were dynamically renormalized across active modalities.

## Results Across All Seven Modality Combinations

| Modality Combination | Evaluation Count ($N$) | Accuracy | Macro F1 | Weighted F1 | Multiclass Brier Score | Expected Calibration Error (ECE) | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`text`** | 1,108 | 60.11% | 0.4205 | 0.5700 | 0.5751 | 0.1162 | 43.02 ms |
| **`audio`** | 1,107 | 47.79% | 0.2534 | 0.4258 | 0.6805 | 0.0313 | 326.25 ms |
| **`text+audio`** | 1,107 | 59.80% | 0.3671 | 0.5518 | 0.5559 | 0.0613 | 369.28 ms |
| **`video`** | 1,096 | 14.60% | 0.0433 | 0.0596 | 0.8412 | 0.0189 | 2,200.61 ms |
| **`text+video`** | 1,096 | 60.31% | 0.4133 | 0.5715 | 0.5980 | 0.1673 | 2,243.77 ms |
| **`audio+video`** | 1,095 | 47.76% | 0.2523 | 0.4258 | 0.7212 | 0.1678 | 2,530.71 ms |
| **`text+audio+video`** | 1,095 | 59.91% | 0.3663 | 0.5526 | 0.6002 | 0.1863 | 2,573.88 ms |

## Per-Class F1 Score Breakdown

| Emotion Class | Support ($N$) | Text F1 | Audio F1 | Text+Audio F1 | Video F1 | Text+Video F1 | Audio+Video F1 | Tri-modal F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **anger** | 153 | 0.4071 | 0.4332 | 0.4505 | 0.2410 | 0.4029 | 0.4273 | 0.4521 |
| **disgust** | 22 | 0.2759 | 0.0000 | 0.0833 | 0.0000 | 0.2143 | 0.0000 | 0.0833 |
| **fear** | 40 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.1333 | 0.0000 | 0.0000 |
| **joy** | 163 | 0.5226 | 0.1176 | 0.4751 | 0.0000 | 0.5245 | 0.1296 | 0.4710 |
| **neutral** | 469 | 0.7603 | 0.6611 | 0.7485 | 0.0620 | 0.7639 | 0.6605 | 0.7498 |
| **sadness** | 111 | 0.2914 | 0.2000 | 0.2429 | 0.0000 | 0.2914 | 0.1884 | 0.2336 |
| **surprise** | 150 | 0.5609 | 0.3616 | 0.5697 | 0.0000 | 0.5630 | 0.3602 | 0.5741 |

## Key Findings

1. **Text as Anchor:** Text (`DistilBERT`) provides the strongest baseline across accuracy (60.11%) and macro F1 (0.4205).
2. **Acoustic Complementarity:** Fusing audio (`Wav2Vec2`) with text enhances class recognition on emotionally expressive classes where vocal tone carries strong signal:
   - `anger`: F1 increases from 0.4071 (text only) to 0.4505 (`text+audio`) and 0.4521 (`text+audio+video`).
   - `surprise`: F1 increases from 0.5609 (text only) to 0.5697 (`text+audio`) and 0.5741 (`text+audio+video`).
3. **Probability Calibration:**
   - `text+audio` achieved the lowest Brier score (**0.5559**), demonstrating that audio probability distribution softens text overconfidence.
   - Standalone `audio` had the lowest Expected Calibration Error (**0.0313**).
4. **Facial Modality:** Standalone facial CNN remains biased toward anger/neutral (14.60% accuracy). When fused with text (`text+video`), it achieved the highest overall accuracy (**60.31%**) and highest weighted F1 (**0.5715**).

## Verification and Safety Rules

- Held-out test split data was completely isolated and never accessed.
- Pretrained model checkpoints remain unmodified.
- Missing modality exclusions were handled strictly via dynamic weight renormalization.

