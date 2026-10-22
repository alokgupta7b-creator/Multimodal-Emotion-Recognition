# MELD Speech Emotion Recognition (SER) — Development Benchmark

**Selected Top Model**: `logreg_balanced_C0.01` (selected by `macro_f1`)

## Model Comparison on Development Split

| Model Configuration | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Training Time | Latency / Sample | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`logreg_balanced_C1.0`** | 23.29% | 24.09% | 20.44% | 25.34% | 0.3 s | 0.003 ms | 5.0 KB |
| **`logreg_unweighted_C1.0`** | 45.31% | 18.88% | 16.02% | 33.41% | 0.4 s | 0.002 ms | 5.0 KB |
| **`logreg_balanced_C0.1`** | 23.38% | 24.12% | 20.50% | 25.47% | 0.3 s | 0.003 ms | 5.0 KB |
| **`logreg_balanced_C0.01`** **(Selected)** | 23.65% | 24.25% | 20.62% | 25.75% | 0.2 s | 0.002 ms | 5.0 KB |
| **`mlp_relu_balanced`** | 44.95% | 20.21% | 18.04% | 35.56% | 2.4 s | 0.005 ms | 126.2 KB |
| **`svm_rbf_balanced_C1.0`** | 25.99% | 21.43% | 19.90% | 27.49% | 172.8 s | 2.191 ms | 4083.0 KB |

## Top Model Detailed Breakdown

### SER Evaluation Report: logreg_balanced_C0.01 on `dev`

#### Summary Metrics
| Metric | Value |
| :--- | :---: |
| **Samples** | 1,108 |
| **Accuracy** | 23.65% |
| **Balanced Accuracy** | 24.25% |
| **Macro F1** | 20.62% |
| **Weighted F1** | 25.75% |
| **Macro Precision** | 23.72% |
| **Macro Recall** | 24.25% |
| **Training Duration** | 0.21 s |
| **Inference Latency** | 0.002 ms / sample |
| **Model Size** | 5.0 KB |

#### Per-Class Performance
| Emotion Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **anger** | 33.02% | 46.41% | 38.59% | 153 |
| **disgust** | 1.65% | 13.64% | 2.94% | 22 |
| **fear** | 7.50% | 22.50% | 11.25% | 40 |
| **joy** | 19.69% | 15.34% | 17.24% | 163 |
| **neutral** | 61.70% | 18.55% | 28.52% | 469 |
| **sadness** | 18.88% | 33.33% | 24.10% | 111 |
| **surprise** | 23.62% | 20.00% | 21.66% | 150 |

#### Confusion Matrix (Rows: True, Columns: Predicted)
| True \ Pred | anger | disgust | fear | joy | neutral | sadness | surprise |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **anger** | 71 | 24 | 18 | 10 | 10 | 13 | 7 |
| **disgust** | 4 | 3 | 3 | 1 | 1 | 5 | 5 |
| **fear** | 10 | 5 | 9 | 3 | 5 | 2 | 6 |
| **joy** | 37 | 30 | 13 | 25 | 17 | 21 | 20 |
| **neutral** | 38 | 87 | 47 | 66 | 87 | 96 | 48 |
| **sadness** | 16 | 18 | 10 | 9 | 10 | 37 | 11 |
| **surprise** | 39 | 15 | 20 | 13 | 11 | 22 | 30 |