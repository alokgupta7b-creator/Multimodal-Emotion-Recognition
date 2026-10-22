# MELD SER Evaluation Report — Split: TEST

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