# Dataset Card for CMU-MOSEI (Multimodal Opinion Sentiment and Emotion Intensity)

## 1. Dataset Summary

The **CMU-MOSEI (Carnegie Mellon University Multimodal Opinion Sentiment and Emotion Intensity)** dataset is an authoritative benchmark for in-the-wild multimodal sentiment analysis and emotion recognition. Developed by the Carnegie Mellon MultiComp Lab, it consists of **23,453 annotated monologue video segments** extracted from **3,294 distinct YouTube videos** spoken by over **1,000 distinct speakers** covering **250 distinct topics**.

Each segment represents a single sentence or utterance, precisely bounded by temporal start and end timestamps (`[start_time, end_time]`), and aligned across three primary sensory modalities:
1. **Text**: Transcribed verbal monologue and corresponding linguistic embeddings (e.g., GloVe, BERT).
2. **Speech (Audio)**: Acoustic waveforms and extracted prosodic/spectral feature descriptors (COVAREP, openSMILE, MFCCs).
3. **Facial (Visual)**: Monologue video clips capturing facial landmarks, head pose, gaze vectors, and Facial Action Units (FAUs) extracted via Facet 4.2 and OpenFace 2.0.

### Intended Research Use
This dataset is integrated into this repository to support:
- In-the-wild multimodal sentiment and emotion intensity prediction.
- Dynamic reliability and consistency estimation across text, speech, and facial modalities prior to fusion.
- Evaluating model robustness against sensory noise, missing modalities, and out-of-distribution acoustic/visual artifacts.
- Cross-dataset generalization analysis when evaluated alongside conversational benchmarks like MELD.

---

## 2. Modalities & Label Space

### Modalities Supported
- **Text Modality**: Sentence-level transcripts and word-aligned embeddings capturing semantic meaning and lexical sentiment.
- **Audio Modality**: Vocal acoustic streams characterized by pitch, intensity, formant frequencies, harmonicity, and prosody.
- **Visual Modality**: Face-centered video streams capturing affective micro-expressions, facial muscle contractions (Action Units), and head movements.

### Annotation Schema: Continuous Sentiment & Emotion Intensity
Unlike datasets that restrict annotations to discrete, mutually exclusive categorical labels, CMU-MOSEI provides fine-grained, continuous real-valued annotations produced by crowd annotators:

1. **Continuous Sentiment**:
   - Continuous real value on a 7-point Likert-style continuum: `[-3.0, +3.0]`.
   - Scale interpretation:
     - `[-3.0, -2.0]`: Highly negative
     - `(-2.0, -1.0]`: Moderately negative
     - `(-1.0, 0.0)`: Weakly negative
     - `0.0`: Completely neutral
     - `(0.0, +1.0)`: Weakly positive
     - `[+1.0, +2.0)`: Moderately positive
     - `[+2.0, +3.0]`: Highly positive

2. **Emotion Intensity (6 Ekman Dimensions)**:
   - Each segment is evaluated across 6 primary emotion dimensions on an intensity scale of `[0.0, 3.0]`:
     - **Happiness**: Degree of positive affect, amusement, or joy.
     - **Sadness**: Degree of sorrow, melancholy, or grief.
     - **Anger**: Degree of frustration, hostility, or rage.
     - **Surprise**: Degree of astonishment, shock, or unexpected discovery.
     - **Disgust**: Degree of revulsion, distaste, or moral aversion.
     - **Fear**: Degree of apprehension, worry, or dread.
   - An intensity of `0.0` denotes complete absence of that emotion. Values `> 0.0` quantify the perceived intensity.

### Comparison with MELD Label Space

| Attribute | MELD | CMU-MOSEI |
| :--- | :--- | :--- |
| **Communication Setting** | Multi-party conversational dialogue (*Friends* sitcom) | Single-speaker in-the-wild video monologues (YouTube) |
| **Emotion Targets** | 7 Discrete categorical classes (`neutral`, `joy`, `sadness`, `anger`, `surprise`, `disgust`, `fear`) | 6 Continuous emotion intensities in `[0.0, 3.0]` (`happy`, `sad`, `anger`, `surprise`, `disgust`, `fear`) |
| **Sentiment Targets** | 3 Discrete classes (`positive`, `negative`, `neutral`) | Continuous sentiment score in `[-3.0, +3.0]` |
| **Neutrality Representation** | Explicit categorical class (`neutral`) | Absence of emotion intensity (all 6 dimensions = 0) or sentiment score = 0.0 |
| **Joy / Happiness** | Formally labeled `joy` | Formally labeled `happy` |
| **Target Formulation** | Single-label multiclass classification | Multi-target continuous regression or thresholded multi-label classification |

> [!IMPORTANT]
> The preparation pipeline preserves the original continuous annotations. Continuous targets are never silently collapsed into categorical labels. For downstream discrete classification tasks, configurable decision boundaries (`--sentiment-threshold`, `--emotion-threshold`) are supported while leaving the raw ground-truth values intact.

---

## 3. Split Conventions & Speaker Isolation

To prevent **speaker identity leakage** (where the same speaker appears in both train and test partitions), CMU-MOSEI defines standard splits strictly by **video identifier (`video_id`)**:
- All segment clips originating from YouTube video $V$ are allocated to exactly one partition.
- The pipeline's automated split isolation audit verifies that the intersection of video IDs across partitions is strictly empty:
  $$\text{Train}_{\text{videos}} \cap \text{Dev}_{\text{videos}} = \emptyset, \quad \text{Train}_{\text{videos}} \cap \text{Test}_{\text{videos}} = \emptyset, \quad \text{Dev}_{\text{videos}} \cap \text{Test}_{\text{videos}} = \emptyset$$

### Published Benchmark Split Distribution (Estimates from Authoritative Literature)

The following partition counts reflect published benchmark statistics from the CMU-MOSEI ACL 2018 paper and CMU-MultimodalSDK:

| Split | Unique Videos (Estimated) | Segment Count (Published Benchmark) | Primary Research Function |
| :--- | :--- | :--- | :--- |
| **Train** | ~2,500 | ~16,326 | Model representation learning and unimodal feature encoder optimization |
| **Development (Dev)** | ~300 | ~1,871 | Validation, threshold tuning, early stopping, and consistency parameter estimation |
| **Test** | ~700 | ~4,659 | Held-out evaluation of dynamic reliability estimation and multimodal fusion |
| **Total** | **~3,500** | **~22,856 – 23,453** | Complete benchmark collection (aligned vs. unaligned variants) |

---

## 4. Known Data Limitations & Considerations

1. **In-The-Wild Video Variability**:
   - Recorded under unconstrained web conditions: diverse webcam resolutions, inconsistent lighting, varied compression codecs, and dynamic camera movements.
2. **Acoustic Noise & Background Tracks**:
   - Many YouTube videos include background ambient sound, room reverberation, or non-speech background music. Acoustic branches require robust prosodic representations (e.g., COVAREP, eGeMAPS) or noise-filtering preprocessing.
3. **Continuous Annotation Variance**:
   - Crowd annotations exhibit annotator variance. Models targeting emotion regression frequently optimize Mean Absolute Error (MAE) or Pearson Correlation ($r$) alongside binary presence classification.
4. **Missing Modality Intervals**:
   - In certain monologue segments, facial tracking may drop frames due to extreme head turning or occlusion. Downstream fusion architectures must account for samples where `has_visual` is false or visual confidence is low.
5. **Estimated Local Storage Requirements**:
   - **Raw Video Files**: ~35 GB (estimated if downloading full YouTube MP4 clips).
   - **Precomputed Multimodal Sequences (`.csd` via SDK)**: ~8–12 GB (estimated across COVAREP, Facet, and GloVe features).
   - **Metadata & Annotation Files**: ~15 MB (estimated).

---

## 5. Provenance, Citations & Licensing

- **Dataset Creators**: AmirAli Bagher Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, Louis-Philippe Morency (Carnegie Mellon MultiComp Lab).
- **Official Portal**: [https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/](https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/)
- **Official SDK**: [https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)
- **Redistribution Terms & Governance**:
  - CMU-MOSEI data is distributed for academic research purposes.
  - Raw YouTube video files, large precomputed feature arrays (`.csd`, `.h5`), and copyright-restricted transcripts are **not hosted in this Git repository**.
  - Researchers must obtain data via the CMU-MultimodalSDK (`mmdatasdk`) or official download portals into local directories.

### Academic Citation
```bibtex
@inproceedings{zadeh2018multimodal,
  title={Multimodal Language Analysis in the Wild: {CMU-MOSEI} Dataset and Interpretable Dynamic Fusion Approaches},
  author={Zadeh, AmirAli Bagher and Liang, Paul Pu and Poria, Soujanya and Cambria, Erik and Morency, Louis-Philippe},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={2236--2246},
  year={2018},
  publisher={Association for Computational Linguistics},
  doi={10.18653/v1/P18-1208}
}
```
