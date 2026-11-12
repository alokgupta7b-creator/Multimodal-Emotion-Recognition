# Facial Emotion Recognition Improvement — Diagnostic Phase Report

**Author:** Deep Learning Research & Engineering Agent  
**Date:** November 11, 2026  
**Status:** Completed Diagnostic Audit — Model Frozen (No Models Trained)  
**Deliverable Document:** `docs/FACIAL_MODEL_DIAGNOSTIC_REPORT.md`  
**Artifact Directory:** `docs/facial_diagnostics/`  

---

## 1. Executive Summary & Root Cause Findings

An exhaustive investigation was conducted to determine why the current Facial Emotion Classifier produces weak classification performance (Test Accuracy: **38.46%**, Macro F1: **11.63%**; Dev Accuracy: **14.26%**, Macro F1: **0.0366**) and severely uncertain predictions on real-world conversational video (**15.2% confidence**, predicting Neutral over 52 detected faces across 17 sampled frames).

### Key Diagnostic Discoveries

1. **The Inference Normalization Inversion Bug (Primary Cause of 15.2% Confidence):**
   * During training, face crops were normalized to $[-1.0, +1.0]$ via `(tensor - 0.5) / 0.5`.
   * In `FacialEmotionPredictor.predict_faces` (`facial_emotion/model.py`, line 87), the code checks `self.normalize_input = bool(checkpoint.get("input_normalization", False))`.
   * Because `facial_cnn_best.pt` does not contain this metadata key, `self.normalize_input` evaluated to `False`, executing `tensor = tensor * 0.5 + 0.5`.
   * This shifted all inputs during video inference into $[0.0, 1.0]$. For a CNN with weights trained on zero-centered $[-1, 1]$ inputs without batch normalization, this domain shift saturates all activations, compressing output logits into near-zero uniform noise ($\sim 0.152$ across all 7 classes, where uniform random guessing is $1/7 = 0.1428$).
2. **Model Training Non-Convergence (Vanished Learning from Scratch):**
   * Training loss across all 5 epochs remained flat:
     $$\text{Epoch 1: } 1.9465 \longrightarrow \text{Epoch 5: } 1.9444$$
   * Theoretical loss for uniform random guessing across 7 classes is $\ln(7) = 1.9459$.
   * The training loss was within **0.0015** of theoretical random guessing at every single epoch. The 94k-parameter scratch CNN completely failed to extract separable emotion representations.
3. **Severe Class Imbalance & Single-Class Collapse:**
   * Neutral accounts for **47.1%** of training samples and **42.4%** of development samples.
   * Disgust (**2.7%**) and Fear (**2.6%**) have only ~260 samples each.
   * The frozen checkpoint collapsed to predicting Anger or Neutral, with **0.0000 Precision and Recall** across Disgust, Fear, Joy, Sadness, and Surprise on the development split.
4. **OpenCV Haar Cascade False Positives & Multi-Speaker Pollution:**
   * In the audited test video (`dia4_utt1.mp4`, 1.79s, 43 frames), Haar cascade detected **52 bounding boxes** across 15 sampled frames (averaging 3.5 faces/frame).
   * Inspection revealed detections on background objects (e.g. coffee tables at $y=642$, tiny 26x26 background wall noise), and detected both speakers simultaneously without tracking the target speaker.
5. **The Face Tracking Center Flaw in Video Preprocessing:**
   * In `facial_emotion/train.py` (line 70), tracking attempts to match previous face centers:
     `face = min(faces, key=lambda crop: norm([crop.shape[1]/2, crop.shape[0]/2] - center))`
   * Because `crop` has already been resized to $96 \times 96$, its center is identically $(48.0, 48.0)$ for every face candidate. The distance was identically zero, causing the pipeline to pick the first detected face arbitrarily.

---

## 2. Dataset Audit

### 2.1 Split Coverage & Video Manifest Verification

The dataset manifests (`video_manifest.csv`) for MELD were audited across `train` and `dev` splits:

| Split | Manifest Videos | Status OK | Cached Face Crops | Excluded (No Face Detected) | Cache Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 9,988 | 9,988 | 9,756 | 232 | 97.68% |
| **Dev** | 1,108 | 1,108 | 1,080 | 28 | 97.47% |
| **Total** | 11,096 | 11,096 | 10,836 | 260 | 97.66% |

All excluded videos (232 train, 28 dev) failed due to OpenCV Haar cascade finding zero faces in extreme head turns, dark lighting, or motion blur.

### 2.2 Seven-Class Emotion Distribution & Extreme Imbalance

The class distribution across raw manifests and cached face crops reveals an extreme majority bias:

| Emotion Class | Train Manifest Count | Train % | Train Cached | Dev Cached Count | Dev % | Imbalance vs Neutral |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Neutral** | 4,709 | 47.15% | 4,593 | 458 | 42.41% | **1.0x (Reference)** |
| **Joy** | 1,743 | 17.45% | 1,710 | 155 | 14.35% | 2.7x smaller |
| **Surprise** | 1,205 | 12.06% | 1,177 | 147 | 13.61% | 3.9x smaller |
| **Anger** | 1,109 | 11.10% | 1,082 | 152 | 14.07% | 4.2x smaller |
| **Sadness** | 683 | 6.84% | 670 | 108 | 10.00% | 6.9x smaller |
| **Disgust** | 271 | 2.71% | 267 | 22 | 2.04% | **17.2x smaller** |
| **Fear** | 268 | 2.68% | 257 | 38 | 3.52% | **17.9x smaller** |
| **Total** | 9,988 | 100.0% | 9,756 | 1,080 | 100.0% | — |

* Minority classes (Disgust and Fear) combined make up only **5.37%** of the entire training dataset.
* Without explicit cost-sensitive weighting or focal loss, gradient updates are completely swamped by Neutral and Joy samples.

### 2.3 Image Quality & Split Integrity Audit

A full pixel-level pass across all 9,756 training crops and 1,080 development crops was executed:

* **Image Dimensions:** Verified $100\%$ consistent at $96 \times 96 \times 3$ uint8 across all samples.
* **Mean Pixel Intensity (Train):** $70.64 \pm 25.25$ (typical indoor sitcom lighting).
* **Mean Image Contrast (Std):** $38.26 \pm 9.79$.
* **Black / Corrupt Images (Max Pixel $< 10$):** **0** across all splits.
* **Low-Contrast Crops (Std $< 15$):** **75** samples in train (mostly blurry background detections).
* **Exact Duplicate Detection (MD5 Hashes):**
  * **Intra-Train Duplicates:** 162 identical crops (arising from repeated frames in short sitcom shots).
  * **Train-to-Dev Duplicate Leakage:** **0 identical crops** (zero cross-split data leakage).
  * **Dialogue ID Overlap:** Zero true overlap (MELD dialogue indices are split-local `train_diaX` vs `dev_diaX`).

---

## 3. Model Architecture & Convergence Audit

### 3.1 Model Architecture Dissection

The current facial model (`facial_emotion/model.py`) is an un-normalized 3-layer convolutional network:

```text
Input: (Batch, 3, 96, 96)
  │
  ├── [Layer 0] Conv2d(3, 32, kernel=3, padding=1)     -> 896 params
  ├── [Layer 1] ReLU()
  ├── [Layer 2] MaxPool2d(2, stride=2)                -> (Batch, 32, 48, 48)
  │
  ├── [Layer 3] Conv2d(32, 64, kernel=3, padding=1)    -> 18,496 params
  ├── [Layer 4] ReLU()
  ├── [Layer 5] MaxPool2d(2, stride=2)                -> (Batch, 64, 24, 24)
  │
  ├── [Layer 6] Conv2d(64, 128, kernel=3, padding=1)   -> 73,856 params
  ├── [Layer 7] ReLU()
  ├── [Layer 8] AdaptiveAvgPool2d((1, 1))              -> (Batch, 128, 1, 1)
  │
  ├── [Layer 9] Flatten()                             -> (Batch, 128)
  └── [Layer 10] Linear(128, 7)                       -> 903 params
Total Parameters: 94,151 (0.094M) | Trainable: 94,151
```

#### Deficiencies of this Architecture:
1. **No Normalization Layers:** Absence of `BatchNorm2d` or `LayerNorm` causes internal covariate shift and vanishing gradients during random initialization.
2. **Extreme Underparameterization:** 94,151 parameters is inadequate to capture complex facial geometry, fine facial action units (micro-expressions), head pose rotations, and lighting variation.
3. **No Residual Connections:** Features cannot bypass layers, leading to degraded representation capacity.
4. **Scratch Training:** Attempting to train vision features on 9,756 noisy in-the-wild crops without pretraining on large-scale face or image datasets (e.g. ImageNet, VGGFace2, FER2013).

### 3.2 Loss Trajectory & Non-Convergence Proof

Inspection of `models/facial_emotion/training_metadata.json` shows the training progression over 5 epochs:

| Epoch | Training Loss | Development Loss | Dev Accuracy | Dev Macro F1 | Theoretical Uniform Loss $\ln(7)$ | Delta to Uniform Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.9465 | 1.9170 | 42.41% | 0.0851 | 1.9459 | +0.0006 |
| **2** | 1.9447 | 1.9011 | 42.41% | 0.0851 | 1.9459 | -0.0012 |
| **3** | 1.9460 | 1.9193 | 30.93% | 0.0997 | 1.9459 | +0.0001 |
| **4** | 1.9450 | 1.9244 | 14.44% | 0.0425 | 1.9459 | -0.0009 |
| **5** | 1.9444 | 1.9070 | 42.22% | 0.0867 | 1.9459 | **-0.0015** |

**Mathematical Proof of Non-Convergence:**  
For a 7-class classification task with cross-entropy loss, a model outputting uniform logits $z_i = c$ produces:
$$\mathcal{L}_{\text{uniform}} = -\ln\left(\frac{1}{7}\right) = \ln(7) \approx 1.94591$$
The actual training loss began at $1.9465$ and concluded at $1.9444$. At no point during training did the model depart by more than $0.0015$ from theoretical random guessing. The model never converged.

### 3.3 Development Set Evaluation of Frozen Checkpoint

Evaluating the frozen `facial_cnn_best.pt` on all 1,080 development split face crops yields:

| Emotion Class | Precision | Recall | F1-Score | Support | True Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Anger** | 0.1410 | 1.0000 | 0.2472 | 152 | Predicting virtually all samples as Anger |
| **Disgust** | 0.0000 | 0.0000 | 0.0000 | 22 | Total Class Collapse (0 detections) |
| **Fear** | 0.0000 | 0.0000 | 0.0000 | 38 | Total Class Collapse (0 detections) |
| **Joy** | 0.0000 | 0.0000 | 0.0000 | 155 | Total Class Collapse (0 detections) |
| **Neutral** | 1.0000 | 0.0044 | 0.0087 | 458 | Total Class Collapse (2 true positives) |
| **Sadness** | 0.0000 | 0.0000 | 0.0000 | 108 | Total Class Collapse (0 detections) |
| **Surprise** | 0.0000 | 0.0000 | 0.0000 | 147 | Total Class Collapse (0 detections) |
| **Macro Average** | **0.1630** | **0.1435** | **0.0366** | **1,080** | **Severe 5-class total blackout** |
| **Overall Accuracy** | — | — | **14.26%** | **1,080** | **Matches uniform random rate ($1/7 = 14.28\%$)** |

![Development Set Facial Confusion Matrix](docs/facial_diagnostics/dev_confusion_matrix.png)

```text
Confusion Matrix (Rows: Ground Truth, Columns: Predictions):
             anger  disgust  fear   joy  neutral  sadness  surprise
anger     [   152,       0,    0,    0,       0,       0,        0 ]
disgust   [    22,       0,    0,    0,       0,       0,        0 ]
fear      [    38,       0,    0,    0,       0,       0,        0 ]
joy       [   155,       0,    0,    0,       0,       0,        0 ]
neutral   [   456,       0,    0,    0,       2,       0,        0 ]
sadness   [   108,       0,    0,    0,       0,       0,        0 ]
surprise  [   147,       0,    0,    0,       0,       0,        0 ]
```

**Interpretation:** 1,078 out of 1,080 samples were predicted as Anger, with 2 predicted as Neutral. 5 out of 7 classes suffered a complete collapse to zero true positives.

---

## 4. Video Inference Audit: The Real-World Video Case Study

The audited test video is `dev_dia4_utt1.mp4` (Duration: 1.79s, 43 frames, $1280 \times 720$, H.264, Ground Truth: `neutral`).

### 4.1 Sampled Frame & Face Detection Breakdown

Sampling every 3 frames yields 15 actual video frames. However, OpenCV Haar cascade generated **52 detected faces**:

| Frame Index | Video Timestamp (s) | Detections | Bounding Boxes $(x, y, w, h)$ | Visual Diagnosis |
| :---: | :---: | :---: | :--- | :--- |
| **0** | 0.00s | 4 | `(314,72,27,27)`, `(371,131,65,65)`, `(455,185,73,73)`, `(314,53,27,27)` | 2 true faces (Chandler & Monica) + 2 background false positives |
| **3** | 0.13s | 3 | `(363,117,95,95)`, `(456,184,75,75)`, `(316,54,27,27)` | 2 true faces + 1 wall texture false positive |
| **6** | 0.25s | 3 | `(368,128,69,69)`, `(455,187,71,71)`, `(339,55,26,26)` | 2 true faces + 1 background false positive |
| **9** | 0.38s | 2 | `(446,180,79,79)`, `(365,126,74,74)` | Both actors detected cleanly |
| **12** | 0.50s | 4 | `(370,124,77,77)`, `(454,182,74,74)`, `(317,54,26,26)`, `(340,55,27,27)` | 2 true faces + 2 false positives |
| **15** | 0.63s | 4 | `(368,130,67,67)`, `(455,179,79,79)`, `(317,54,27,27)`, `(341,56,26,26)` | 2 true faces + 2 false positives |
| **18** | 0.75s | 4 | `(457,179,80,80)`, `(340,127,85,85)`, `(316,55,27,27)`, `(340,56,26,26)` | 2 true faces + 2 false positives |
| **21** | 0.88s | 4 | `(307,133,79,79)`, `(334,127,89,89)`, `(317,55,26,26)`, `(340,56,26,26)` | Chandler moving + 2 false positives |
| **24** | 1.00s | 4 | `(349,128,84,84)`, `(525,216,26,26)`, `(317,56,27,27)`, `(362,57,26,26)` | 1 true face + 3 false positives |
| **27** | 1.13s | 4 | `(475,183,76,76)`, `(876,164,66,66)`, `(318,57,26,26)`, `(341,57,26,26)` | Monica + background extra + 2 false positives |
| **30** | 1.25s | 2 | `(461,184,71,71)`, `(317,57,26,26)` | Monica + 1 background false positive |
| **33** | 1.38s | 4 | `(456,181,69,69)`, `(316,76,28,28)`, `(317,57,26,26)`, `(340,58,25,25)` | Monica + 3 background false positives |
| **36** | 1.50s | 3 | `(458,179,76,76)`, `(318,58,26,26)`, `(340,59,26,26)` | Monica + 2 background false positives |
| **39** | 1.63s | 2 | `(473,183,75,75)`, `(318,58,26,26)` | Monica + 1 background false positive |
| **42** | 1.75s | 5 | `(486,185,75,75)`, `(914,642,71,71)`, `(317,76,28,28)`, `(318,56,27,27)`, `(341,58,26,26)` | Monica + coffee table/floor $(y=642)$ + 3 false positives |

#### Key Inference Artifacts:
* **The "17 Sampled Frames" Calculation Myth:**  
  In `FacialEmotionPredictor.predict_video` (line 110):
  `num_sampled_frames = max(1, int(round(len(faces) / max(1, sample_every))))`  
  Because $\text{len}(faces) = 52$ and $sample\_every = 3$:
  $$\text{num\_sampled\_frames} = \text{round}\left(\frac{52}{3}\right) = \text{round}(17.33) = \mathbf{17}$$
  The API never counted sampled video frames; it divided detected face crops by 3.
* **Persistent False Positives:** Bounding boxes like `(317, 54, 26, 26)` occurred repeatedly in every frame. Haar cascade triggered on a wall light fixture.
* **Floor Artifact:** Bounding box `(914, 642, 71, 71)` at $y=642$ in a 720p frame represents a patch on the coffee table.

### 4.2 Mathematical Mechanism of the 15.2% Confidence Collapse

When face crops are normalized properly in $[-1.0, +1.0]$ vs when shifted into $[0.0, 1.0]$ by the un-normalized flag:

| Emotion Class | Normalized Input $[-1.0, +1.0]$ | Shifted Input $[0.0, 1.0]$ (Live Pipeline Bug) |
| :--- | :---: | :---: |
| **Anger** | **67.16%** | **16.58%** |
| **Neutral** | 12.87% | **15.75%** |
| **Joy** | 10.53% | **15.35%** |
| **Sadness** | 5.57% | **14.37%** |
| **Surprise** | 2.60% | **14.25%** |
| **Disgust** | 1.01% | **12.55%** |
| **Fear** | 0.26% | **11.16%** |
| **Top Predicted Emotion** | **Anger** | **Neutral / Anger (Tie)** |
| **Prediction Confidence** | **67.16%** | **15.75% — 16.58%** |
| **Prediction Entropy** | 1.144 | **1.940 (99.7% of Max Entropy)** |

The input normalization shift compressed logits by a factor of 10x, directly reducing confidence from $67.2\%$ down to $15.7\%-16.5\%$. Combined with temporal averaging over 52 crops, the live video output produced **15.2% confidence**.

---

## 5. Improvement Proposals & Controlled Experiments

To elevate the Facial Emotion Recognition module from a non-converged toy CNN to a high-precision, production-grade vision system, we propose a 4-dimensional ablation matrix for controlled development-only experimentation.

### Dimension 1: Backbone Architectures

| Model Architecture | Parameters | Pretraining Source | Key Advantages | Intended Role |
| :--- | :---: | :---: | :--- | :--- |
| **1. Baseline CNN (Current)** | 0.094M | None (Scratch) | Fast execution (~2 ms) | Reference baseline |
| **2. ResNet-18** | 11.2M | ImageNet-1K (`ResNet18_Weights.DEFAULT`) | Residual connections prevent vanishing gradients; proven transfer capability on FER | High-accuracy robust standard |
| **3. EfficientNet-B0** | 4.0M | ImageNet-1K (`EfficientNet_B0_Weights.DEFAULT`) | Depthwise separable convolutions; compound scaling; 65% fewer parameters than ResNet18 | Edge/real-time balanced standard |

### Dimension 2: Loss Formulations & Class Balancing

To counteract the 17.5x class imbalance without causing minority-class overcompensation:

1. **Standard Cross-Entropy (Baseline):**  
   $\mathcal{L}_{\text{CE}} = -\log(p_{y})$
2. **Class-Weighted Cross-Entropy:**  
   $$\mathcal{L}_{\text{WCE}} = -w_{y} \log(p_{y}), \quad w_c = \left(\frac{N}{7 \cdot N_c}\right)^\alpha \quad (\alpha \in \{0.5, 1.0\})$$
3. **Focal Loss (Lin et al.):**  
   $$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t), \quad \gamma = 2.0$$  
   Downweights easy Neutral/Joy examples and concentrates gradients on ambiguous or minority frames.
4. **Balanced Weighted Random Sampler:**  
   Samples training batches uniformly across the 7 classes, evaluating whether data-level oversampling outperforms loss-level reweighting.

### Dimension 3: Modern Face Detection & Speaker Tracking

Replace OpenCV Haar cascades with modern landmark-aware face detectors:

| Component | Current Approach | Proposed Modern Approach | Benefit |
| :--- | :--- | :--- | :--- |
| **Face Detector** | Haar Cascade (`haarcascade_frontalface_default.xml`) | **MediaPipe Face Mesh** or **OpenCV DNN Face (ResNet10-SSD)** | Eliminates false-positive furniture/lighting detections; provides detection confidence |
| **Facial Alignment** | Unaligned box crop resized to $96 \times 96$ | **Affine 5-point landmark alignment** (eyes horizontal, normalized inter-pupil distance) | Removes pose tilt; presents canonically aligned faces to CNN |
| **Speaker Tracking** | Arbitrary index or broken center check | **IoU-based spatial tracking on original video frame bounding boxes** | Tracks the primary speaking character; discards background listener faces |

### Dimension 4: Quality-Aware Temporal Aggregation

Replace uniform arithmetic logit averaging across all detected crops:

1. **Confidence-Weighted Aggregation:**  
   $$\bar{z} = \frac{\sum_{t=1}^T c_t \cdot z_t}{\sum_{t=1}^T c_t}$$  
   Weights each frame's logits by the face detector's confidence score $c_t$ and face bounding-box area.
2. **Top-$K$ Emotion Logit Pooling:**  
   Takes the mean of the top-$K$ highest-activation frames for each emotion, preventing neutral dialogue frames from washing out brief, high-intensity affective expressions.
3. **Temporal Attention Module:**  
   A lightweight 1-layer temporal attention head learning frame importance weights dynamically:
   $$\alpha_t = \text{softmax}(w^T \tanh(W h_t))$$

---

## 6. Proposed Experimental Matrix (Controlled Protocol)

All candidate models will be evaluated strictly on the **Development split ($N = 1,080$)** under reproducible seeds:

| Exp # | Model Backbone | Face Detector | Loss / Sampling | Normalization Fix | Expected Outcome |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **Exp 1** | Baseline CNN (94k) | Haar Cascade | Standard CE | Fixed to $[-1, +1]$ | Establishes true corrected baseline performance |
| **Exp 2** | Baseline CNN (94k) | Haar Cascade | Class-Weighted CE | Fixed to $[-1, +1]$ | Measures class reweighting benefit on toy CNN |
| **Exp 3** | **ResNet-18 (Pretrained)** | Haar Cascade | Standard CE | ImageNet Norm | Tests ImageNet transfer on raw crops |
| **Exp 4** | **ResNet-18 (Pretrained)** | MediaPipe / DNN | Class-Weighted CE | ImageNet Norm | Landmark alignment + class weighting |
| **Exp 5** | **ResNet-18 (Pretrained)** | MediaPipe / DNN | Focal Loss ($\gamma=2.0$) | ImageNet Norm | Addresses hard negative and minority classes |
| **Exp 6** | **EfficientNet-B0** | MediaPipe / DNN | Focal Loss ($\gamma=2.0$) | ImageNet Norm | High-efficiency alternative benchmark |
| **Exp 7** | Best Model from 4-6 | MediaPipe + Tracking | Best Loss | Best Norm | **Full Pipeline with Speaker Tracking & Top-K Temporal Pooling** |

---

## 7. Strict Restrictions Compliance Confirmation

In accordance with strict system guidelines:
* **DistilBERT:** Untouched and frozen.
* **Wav2Vec2:** Untouched and frozen.
* **Candidate 4 Fusion Engine:** Untouched and frozen ($\tau_a = 0.20, \lambda_{\max} = 0.40, \kappa_a = 1.50, \lambda_v = 0.00$).
* **Model Checkpoints:** No checkpoint files modified.
* **MELD Test Split:** Untouched — zero evaluation or tuning on test split during this diagnostic phase.
* **No Models Trained:** Strictly diagnostic and audit phase; no training was initiated.

---

## 8. Summary of Saved Artifacts

All supporting diagnostic artifacts have been generated and archived in `docs/facial_diagnostics/`:
* [`docs/facial_diagnostics/dataset_audit.json`](file:///C:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/docs/facial_diagnostics/dataset_audit.json): Pixel statistics, duplicate counts, split distributions.
* [`docs/facial_diagnostics/model_audit.json`](file:///C:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/docs/facial_diagnostics/model_audit.json): Architecture layers, parameter counts, dev metrics, probability distributions.
* [`docs/facial_diagnostics/dev_confusion_matrix.png`](file:///C:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/docs/facial_diagnostics/dev_confusion_matrix.png): Development set confusion matrix showing single-class collapse.
* [`docs/facial_diagnostics/video_audit.json`](file:///C:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/docs/facial_diagnostics/video_audit.json): Video metadata, frame detection counts, temporal segment distributions.
* [`docs/facial_diagnostics/frame_level_face_predictions.csv`](file:///C:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/docs/facial_diagnostics/frame_level_face_predictions.csv): Frame-by-frame bounding boxes and 7-class probability distributions for all 52 detected faces.

