# Facial Emotion Detection Module

## 1. Overview

This module implements the facial emotion recognition component of the Multimodal Emotion Recognition project.

It extracts facial landmarks from MELD video clips using OpenCV and MediaPipe Face Mesh, generates numerical facial features, and classifies facial expressions into seven emotion categories.

The module is designed to provide visual emotion probability outputs for integration with the text and speech emotion recognition branches.

## 2. Technologies Used

- Python 3.11
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Joblib

## 3. Emotion Categories

The model recognizes seven MELD emotion classes:

1. Anger
2. Disgust
3. Fear
4. Joy
5. Neutral
6. Sadness
7. Surprise

## 4. Methodology

### Step 1: Video Input

MELD video clips are read using OpenCV. Videos are sampled at multiple evenly distributed frame positions.

### Step 2: Face Detection

MediaPipe Face Detection identifies faces in the sampled frames.

The largest detected face is selected for the baseline feature extraction process.

### Step 3: Facial Landmark Extraction

MediaPipe Face Mesh extracts 478 facial landmarks for each successfully processed face.

Each landmark contributes normalized X and Y coordinates.

### Step 4: Feature Extraction

For each video, the extracted landmark features are aggregated using:

- Mean of facial landmark coordinates
- Standard deviation of facial landmark coordinates

The resulting feature vector contains 1,912 numerical features.

### Step 5: Emotion Classification

A Logistic Regression classifier is trained using:

- StandardScaler for feature normalization
- Class balancing to address class imbalance
- Multiclass emotion classification

The classifier generates probability distributions for all seven emotion categories.

## 5. Dataset Preparation

The module uses the MELD dataset with three splits:

| Split | Successful feature samples |
|---|---:|
| Train | 9,791 |
| Development | 1,078 |
| Test | 2,548 |
| **Total** | **13,417** |

Some videos do not produce usable facial landmarks because of missing faces, partial visibility, or unsuccessful face detection.

These samples are excluded from facial classifier training and prediction.

## 6. Model Performance

The current Logistic Regression baseline achieved:

| Metric | Development | Test |
|---|---:|---:|
| Accuracy | 21.80% | 20.02% |
| Macro F1 | 19.41% | 17.72% |
| Weighted F1 | 22.97% | 21.40% |

The model comparison experiment evaluated Logistic Regression, Random Forest, Extra Trees, and RBF SVM using development-set performance.

Logistic Regression achieved the highest development Macro F1 among these experiments.

## 7. Output Files

The trained model is saved at:

`D:/MELD_Facial_Output/models/facial_emotion_logistic_regression.joblib`

Fusion probability files:

- `visual_dev_probabilities.csv`
- `visual_test_probabilities.csv`

These files contain:

- `sample_id`
- `anger`
- `disgust`
- `fear`
- `joy`
- `neutral`
- `sadness`
- `surprise`

Each row contains the predicted probability distribution for one sample.

## 8. Running the Module

Activate the configured Python environment and navigate to the repository root.

### Extract facial features

```bash
python -m facial_emotion.extract_dataset
```

### Retry previously failed samples

```bash
python -m facial_emotion.retry_failed_samples
```

### Validate extracted features

```bash
python -m facial_emotion.validate_features
```

### Train the facial classifier

```bash
python -m facial_emotion.train_classifier
```

### Compare classifiers

```bash
python -m facial_emotion.compare_classifiers
```

## 9. Important Limitations

- Facial detection can fail when faces are partially visible or obscured.
- The largest detected face is not necessarily the actual speaker.
- Mean and standard deviation of landmarks may not capture all facial expression characteristics.
- The current classifier has relatively low Macro F1 performance.
- Missing visual predictions must be handled appropriately during multimodal fusion.
- The current landmark features should not be interpreted as Facial Action Coding System measurements.

## 10. Integration with Multimodal Fusion

The facial emotion module provides seven-class probability outputs.

These outputs can be combined with the text and speech emotion probability distributions using `sample_id` as the common identifier.

Missing modality predictions should be handled explicitly during fusion rather than replaced with fabricated probabilities.

## 11. Module Status

The following components have been implemented:

- MELD video processing
- MediaPipe face detection
- Facial landmark extraction
- Numerical feature generation
- Feature validation
- Logistic Regression classification
- Classifier comparison
- Development and test probability generation


## Running on Another Machine

The paths in `facial_emotion/config.py` are currently configured for the original development machine. Before running the pipeline on another computer, update the following paths in `config.py`:

- `video_root`: Set this to the location of the extracted MELD video dataset.
- `manifest_root`: Set this to the location of the MELD manifest CSV files in the repository.
- `output_dir`: Set this to a writable output directory on the new machine.

Install the required Python dependencies using the project's `requirements.txt` and ensure the compatible Python and MediaPipe versions are installed. Then run the feature extraction, validation, and training scripts in the documented order.

The generated feature batches and trained model are stored in the configured output directory and must be regenerated on the new machine.

The module is ready for further experimentation and integration with the other emotion recognition branches.