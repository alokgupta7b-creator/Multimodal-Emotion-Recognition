# Phase 10 Implementation Plan: FastAPI Backend Integration

## Status

Inspection is complete. This document is the implementation plan requested
before backend changes. No FastAPI application, backend package, or frontend
package was found in the repository. No model training or artifact changes are
planned.

## Existing inference entry points

| Modality | Existing entry point | Approved artifact |
|---|---|---|
| Text | `text_emotion.distilbert_model.DistilBertTextPredictor.predict` | `models/text_emotion/distilbert/best` |
| Audio | `speech_emotion.wav2vec2_model.Wav2Vec2Predictor.predict_file` | `models/speech_emotion/wav2vec2/best` |
| Video | `facial_emotion.model.FacialEmotionPredictor.predict_video` | `models/facial_emotion/facial_cnn_best.pt` |

The shared canonical class order is:

`anger, disgust, fear, joy, neutral, sadness, surprise`

The API will preserve this order in every probability object. Existing model
architectures and preprocessing functions will be reused rather than
reimplemented.

## Planned backend structure

Create a small `api/` package:

- `api/app.py`: FastAPI application, lifespan startup, routes, and CORS.
- `api/config.py`: repository-relative artifact paths, upload limits, duration
  limits, and allowed frontend origins from environment variables.
- `api/schemas.py`: request/response models and safe error schemas.
- `api/model_registry.py`: startup model loading, status, device selection,
  and inference locking/resource policy.
- `api/validation.py`: text/file validation, media type sniffing, upload size
  enforcement, safe temporary files, and cleanup.

Add backend tests under `api/tests/`.

## API contract

- `GET /health`: process health and readiness.
- `GET /models/status`: loaded/unavailable status, model identifiers, device,
  and artifact paths represented safely.
- `POST /predict/text`: JSON text input.
- `POST /predict/audio`: multipart upload.
- `POST /predict/video`: multipart upload.

Prediction responses will contain:

- `predicted_emotion`
- `probabilities` with all seven classes
- `confidence`
- `inference_duration_ms`
- `modality`
- `modality_status`
- modality quality fields when available

Video responses will include sampled-frame count, detected-face count,
coverage, entropy, and missing-face status when the existing predictor returns
those values.

## Startup and resource policy

Models will load once during FastAPI lifespan startup. The registry will select
CUDA only when available and configured; otherwise it will use CPU. Because
the RTX 4050 has 6 GB VRAM, the registry will expose a conservative device
policy and serialize heavyweight inference calls if needed rather than loading
duplicate model instances. Startup failures will be explicit and reflected in
`/health` and `/models/status`; routes will not return success-shaped fallback
responses.

## Input and security controls

- Validate non-empty text through the existing DistilBERT predictor.
- Inspect file signatures/content through the media loaders, not only suffixes.
- Enforce configurable maximum upload bytes and video duration.
- Use `NamedTemporaryFile`/temporary directories with generated names.
- Pass only resolved temporary paths to predictors.
- Remove temporary files in `finally` blocks.
- Reject path-like client input; clients provide file bytes only.
- Return generic safe errors while logging no file contents or secrets.
- Configure CORS from an explicit allowlist; default to no cross-origin
  origins unless configured.
- Reject corrupt, silent, empty, unsupported, or no-face media with clear
  client errors.

## Testing plan

Add tests for:

- application lifespan and model registry startup;
- model status and health responses;
- text, audio, and video prediction routes;
- canonical seven-class mapping and probability normalization;
- CPU registry/inference;
- CUDA path when available;
- invalid signatures, oversized uploads, empty text, silent audio, corrupt
  media, and missing faces;
- temporary-file cleanup;
- startup/model-loading failures and safe HTTP errors.

Tests will use existing artifacts and small repository fixtures where possible.
No tests will fabricate model predictions or modify datasets.

## Validation commands

After implementation:

```text
python -m pytest api -q
python -m pytest -q
python -m py_compile api\*.py api\tests\*.py
```

The final report will record actual model-loading and endpoint results,
dependency changes, warnings, and remaining limitations. Phase 10 will not
implement multimodal fusion or retrain any model.
