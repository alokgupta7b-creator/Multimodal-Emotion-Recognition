# CMU-MOSEI 7-Class Harmonized Speech Emotion Recognition Benchmark

## Development Split Model Comparison
| Model Configuration | Class Weight | C | Dev Accuracy | Dev Balanced Acc | Dev Macro F1 | Dev Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mosei_logreg_balanced_C1.0`** | balanced | 1.0 | 22.36% | 22.61% | 17.72% | 26.66% |
| **`mosei_logreg_balanced_C0.1`** | balanced | 0.1 | 22.52% | 22.35% | 17.73% | 26.97% |
| **`mosei_logreg_balanced_C0.01`** **(Selected)** | balanced | 0.01 | 23.22% | 22.43% | 17.76% | 27.73% |
| **`mosei_logreg_unweighted_C1.0`** | None | 1.0 | 46.77% | 16.36% | 13.67% | 34.02% |

## Final Held-Out Test Evaluation

**Selected Model**: `mosei_logreg_balanced_C0.01`

### SER Evaluation Report: mosei_logreg_balanced_C0.01 (Test) on `test`

#### Summary Metrics
| Metric | Value |
| :--- | :---: |
| **Samples** | 4,616 |
| **Accuracy** | 24.91% |
| **Balanced Accuracy** | 21.19% |
| **Macro F1** | 18.31% |
| **Weighted F1** | 29.68% |
| **Macro Precision** | 21.21% |
| **Macro Recall** | 21.19% |
| **Inference Latency** | 0.002 ms / sample |
| **Model Size** | 7.7 KB |

#### Per-Class Performance
| Emotion Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **anger** | 19.76% | 16.55% | 18.01% | 592 |
| **disgust** | 12.08% | 28.28% | 16.93% | 244 |
| **fear** | 2.09% | 18.18% | 3.74% | 99 |
| **joy** | 63.53% | 30.70% | 41.40% | 2,202 |
| **neutral** | 23.49% | 18.44% | 20.66% | 694 |
| **sadness** | 24.79% | 21.20% | 22.86% | 698 |
| **surprise** | 2.71% | 14.94% | 4.59% | 87 |

#### Confusion Matrix (Rows: True, Columns: Predicted)
| True \ Pred | anger | disgust | fear | joy | neutral | sadness | surprise |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **anger** | 98 | 91 | 133 | 92 | 54 | 74 | 50 |
| **disgust** | 39 | 69 | 29 | 33 | 15 | 41 | 18 |
| **fear** | 9 | 11 | 18 | 21 | 15 | 11 | 14 |
| **joy** | 205 | 207 | 394 | 676 | 241 | 218 | 261 |
| **neutral** | 66 | 80 | 118 | 131 | 128 | 101 | 70 |
| **sadness** | 68 | 100 | 160 | 89 | 79 | 148 | 54 |
| **surprise** | 11 | 13 | 11 | 22 | 13 | 4 | 13 |