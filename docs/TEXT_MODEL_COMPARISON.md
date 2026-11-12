# Text Model Comparison

| Model | Test accuracy | Macro F1 | Weighted F1 | Status |
|---|---:|---:|---:|---|
| TF-IDF + Logistic Regression | 0.5088 | 0.3683 | 0.5216 | Existing baseline, same MELD test split |
| DistilBERT | 0.6310 | 0.4155 | 0.6077 | CUDA RTX 4050, best epoch 3 |

DistilBERT macro precision was 0.4977 and macro recall was 0.3998. The
baseline macro precision was 0.3591 and macro recall was 0.3846. These metrics
were calculated from the same 2,610 MELD test records; no test records were
used for training, early stopping or model selection.
