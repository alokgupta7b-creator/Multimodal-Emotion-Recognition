# Multimodal Emotion Recognition — System Architecture

**Schema Version:** Phase 19 (Canonical 7-Class MELD Order)  
**Labels:** `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`

---

## 1. End-to-End Request & Inference Architecture

```mermaid
flowchart LR
    U["User / Browser<br/>(React + Vite + TypeScript)"]
    LC["LiveCapture.tsx<br/>(Mic -> 16kHz PCM WAV,<br/>Cam -> WebM/MP4 + Consent Gate)"]
    API["FastAPI Application<br/>(api/app.py)<br/>Size Guard · Rate Limit · Sec Headers"]
    VAL["Upload Validation<br/>(api/validation.py)<br/>Magic Bytes + Temp Auto-Cleanup"]
    REG["ModelRegistry<br/>(api/model_registry.py)"]

    subgraph Branch["Modality Predictors (Zero Test-Split Exposure)"]
        TM["Text Branch<br/>DistilBERT (6-layer)<br/>torch / CUDA or CPU"]
        SM["Speech Branch<br/>Wav2Vec2<br/>16 kHz Mono WAV"]
        VM["Video Branch<br/>OpenCV Frame Sampler + MTCNN<br/>+ ResNet18 (exp5_ce_weighted)<br/>Fallback: exp5_focal_g3_wt"]
    end

    subgraph Fusion["Reliability-Aware Adaptive Fusion"]
        C4["Candidate 4 Engine<br/>(api/fusion.py · PRESERVED)<br/>Text-Anchored + Acoustic Synergy"]
        EXT["Phase 18D Multimodal Extension<br/>(api/multimodal_extension.py)<br/>Calibrated 8% Facial Blend"]
    end

    U --> LC
    LC -->|JSON / multipart form-data| API
    U -->|JSON / multipart form-data| API
    API --> VAL
    VAL --> REG
    REG --> TM
    REG --> SM
    REG --> VM
    TM --> C4
    SM --> C4
    C4 --> EXT
    VM --> EXT
    EXT -->|Probabilities, Rule Applied,<br/>Weights, Quality Metrics| U
```

---

## 2. Core Design Invariants

1. **Candidate 4 Preservation (`api/fusion.py`):** Text + Speech inputs preserve exact Candidate 4 logic (`text_anchor_default`, `synergistic_anger_surprise_override`, protected class lockouts for `joy`/`sadness`/`disgust`/`fear`, silent/unusable audio exclusion).
2. **SHA-256 Checkpoint Integrity (`api/config.py`, `facial_emotion/resnet_predictor.py`):** Approved ResNet18 facial primary (`8825d865...`) and focal fallback (`4830ac84...`) weights are verified before `torch.load(weights_only=True)`.
3. **Ephemeral Media Lifecycle (`api/validation.py`, `api/app.py`):** Magic-byte headers are verified on streamed chunks (`<= 50 MB`), written with randomized `emotion_*` filenames strictly inside OS temp space, deleted inside `finally` blocks immediately after inference, and never written alongside user input filenames or logs.
4. **Sequential Grab Video Decode Optimization (`facial_emotion/mtcnn_detector.py`):** Controlled by `EMOTION_API_OPTIMIZED_VIDEO_PIPELINE=1`, short MP4 clips (`<= 240` total frames) decode via single forward `grab()` + `retrieve()` without random keyframe re-seeking and without double container opening.

