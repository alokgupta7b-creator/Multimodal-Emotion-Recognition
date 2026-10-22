# MELD SER Evaluation Report — Split: TEST

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