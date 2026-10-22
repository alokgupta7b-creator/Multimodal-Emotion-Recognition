# CMU-MOSEI 3-Class Speech Sentiment Recognition Benchmark

## Development Split Model Comparison
| Model Configuration | Class Weight | C | Dev Accuracy | Dev Balanced Acc | Dev Macro F1 | Dev Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mosei_sentiment_balanced_C1.0`** **(Selected)** | balanced | 1.0 | 42.19% | 39.71% | 39.40% | 42.84% |
| **`mosei_sentiment_balanced_C0.1`** | balanced | 0.1 | 41.76% | 39.20% | 38.93% | 42.41% |
| **`mosei_sentiment_unweighted_C1.0`** | None | 1.0 | 50.54% | 38.68% | 34.27% | 42.49% |

## Final Held-Out Test Evaluation

**Selected Model**: `mosei_sentiment_balanced_C1.0`

### SER Evaluation Report: mosei_sentiment_balanced_C1.0 (Test) on `test`

#### Summary Metrics
| Metric | Value |
| :--- | :---: |
| **Samples** | 4,616 |
| **Accuracy** | 43.26% |
| **Balanced Accuracy** | 40.64% |
| **Macro F1** | 40.44% |
| **Weighted F1** | 44.09% |
| **Macro Precision** | 40.70% |
| **Macro Recall** | 40.64% |
| **Inference Latency** | 0.002 ms / sample |
| **Model Size** | 5.2 KB |

#### Per-Class Performance
| Emotion Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **negative** | 38.45% | 40.19% | 39.30% | 1,341 |
| **neutral** | 24.96% | 31.23% | 27.75% | 1,015 |
| **positive** | 58.69% | 50.49% | 54.28% | 2,260 |

#### Confusion Matrix (Rows: True, Columns: Predicted)
| True \ Pred | negative | neutral | positive |
| :--- | :---: | :---: | :---: |
| **negative** | 539 | 367 | 435 |
| **neutral** | 330 | 317 | 368 |
| **positive** | 533 | 586 | 1,141 |