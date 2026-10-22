# MELD Speech Emotion Recognition (SER) — Official Test Set Evaluation

## Summary Comparison on Test Split (2,610 Samples)

| Model | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Latency / Sample | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Selected Model (`C=0.01`, balanced)** | **23.18%** | **23.23%** | **19.17%** | **26.76%** | 21.98% | 23.23% | 0.004 ms | 5.0 KB |
| **Baseline Model (`C=1.0`, balanced)** | 22.45% | 22.45% | 18.67% | 26.00% | 21.70% | 22.45% | 0.004 ms | 5.0 KB |

---

## Selected Model Detailed Test Breakdown

### SER Evaluation Report: ser_best_model (logreg_balanced_C0.01) on `test`

#### Summary Metrics
| Metric | Value |
| :--- | :---: |
| **Samples** | 2,610 |
| **Accuracy** | 23.18% |
| **Balanced Accuracy** | 23.23% |
| **Macro F1** | 19.17% |
| **Weighted F1** | 26.76% |
| **Macro Precision** | 21.98% |
| **Macro Recall** | 23.23% |
| **Inference Latency** | 0.004 ms / sample |
| **Model Size** | 5.0 KB |

#### Per-Class Performance
| Emotion Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **anger** | 25.69% | 29.86% | 27.61% | 345 |
| **disgust** | 3.04% | 19.12% | 5.25% | 68 |
| **fear** | 4.04% | 22.00% | 6.83% | 50 |
| **joy** | 24.14% | 20.90% | 22.40% | 402 |
| **neutral** | 61.94% | 21.89% | 32.35% | 1,256 |
| **sadness** | 12.29% | 25.00% | 16.48% | 208 |
| **surprise** | 22.71% | 23.84% | 23.26% | 281 |

#### Confusion Matrix (Rows: True, Columns: Predicted)
| True \ Pred | anger | disgust | fear | joy | neutral | sadness | surprise |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **anger** | 103 | 54 | 36 | 37 | 30 | 48 | 37 |
| **disgust** | 20 | 13 | 9 | 4 | 5 | 11 | 6 |
| **fear** | 10 | 7 | 11 | 4 | 6 | 8 | 4 |
| **joy** | 72 | 67 | 26 | 84 | 59 | 59 | 35 |
| **neutral** | 127 | 222 | 134 | 159 | 275 | 211 | 128 |
| **sadness** | 33 | 20 | 24 | 25 | 36 | 52 | 18 |
| **surprise** | 36 | 44 | 32 | 35 | 33 | 34 | 67 |

---

## Baseline Model Detailed Test Breakdown

### SER Evaluation Report: ser_baseline_logreg (logreg_balanced_C1.0) on `test`

#### Summary Metrics
| Metric | Value |
| :--- | :---: |
| **Samples** | 2,610 |
| **Accuracy** | 22.45% |
| **Balanced Accuracy** | 22.45% |
| **Macro F1** | 18.67% |
| **Weighted F1** | 26.00% |
| **Macro Precision** | 21.70% |
| **Macro Recall** | 22.45% |
| **Inference Latency** | 0.004 ms / sample |
| **Model Size** | 5.0 KB |

#### Per-Class Performance
| Emotion Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **anger** | 25.77% | 28.99% | 27.29% | 345 |
| **disgust** | 2.84% | 17.65% | 4.90% | 68 |
| **fear** | 3.32% | 20.00% | 5.70% | 50 |
| **joy** | 24.32% | 22.14% | 23.18% | 402 |
| **neutral** | 61.81% | 20.62% | 30.93% | 1,256 |
| **sadness** | 12.59% | 25.00% | 16.75% | 208 |
| **surprise** | 21.26% | 22.78% | 21.99% | 281 |

#### Confusion Matrix (Rows: True, Columns: Predicted)
| True \ Pred | anger | disgust | fear | joy | neutral | sadness | surprise |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **anger** | 100 | 53 | 40 | 38 | 29 | 49 | 36 |
| **disgust** | 20 | 12 | 10 | 4 | 6 | 10 | 6 |
| **fear** | 9 | 9 | 10 | 2 | 5 | 10 | 5 |
| **joy** | 67 | 65 | 36 | 89 | 56 | 56 | 33 |
| **neutral** | 124 | 221 | 143 | 170 | 259 | 203 | 136 |
| **sadness** | 31 | 20 | 24 | 26 | 34 | 52 | 21 |
| **surprise** | 37 | 42 | 38 | 37 | 30 | 33 | 64 |