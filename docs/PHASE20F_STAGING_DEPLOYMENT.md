# Phase 20F — Staging Deployment & End-to-End Validation

**Status:** Local staging-mode verification completed (**`deployed_to_production = False`**; external staging infrastructure remains **`AWAITING USER APPROVAL`**)

---

## 1. Staging Simulation Architecture

Because provisioning an external cloud VPS, domain, or DNS record requires user approval and credentials, Phase 20F executed a full local staging-mode verification via [`scripts/run_phase20f_staging_e2e.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/scripts/run_phase20f_staging_e2e.py) and [`scripts/e2e_phase20f_staging.cjs`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/scripts/e2e_phase20f_staging.cjs):

- **Backend**: Single-worker Uvicorn (`--workers 1`) running on `127.0.0.1:8018` with `EMOTION_API_ENV=staging`, `EMOTION_API_ENABLE_DOCS=0`, `EMOTION_API_FAIL_ON_MISSING_MODELS=1`, `EMOTION_API_DEVICE=cpu`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `EMOTION_API_ALLOWED_ORIGINS=http://localhost:5174`.
- **Frontend**: Production Vite bundle built with `VITE_API_BASE_URL=http://127.0.0.1:8018` and served statically via `vite preview --host localhost --port 5174` (and restored to default build config in `finally`).
- **Fixtures**: Strict development fixtures only (`dia0_utt0.wav` and `dia0_utt0.mp4`); held-out MELD `test` data was never accessed.

---

## 2. Verification Results (`phase20f_staging_validation.json` & `phase20f_uat_validation.json`)

1. **Staging Security & Readiness Probes**:
   - `GET /ready` returned HTTP **`200`** (`{"status": "ok", "ready": true, "classes": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]}`).
   - `GET /docs`, `GET /openapi.json`, and `GET /redoc` all returned HTTP **`404`**.
   - Security headers (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, `Cross-Origin-Resource-Policy`) verified on `/ready`; `Cache-Control: no-store` verified on `/predict/*`.

2. **Browser E2E & UI Validation (`task-4383`)**:
   - **Accessibility & Validation**: Skip link, main landmark, 4 ARIA tabs (1 active), non-diagnostic disclaimer present before and after inference, empty-text alert check, `503` Retry recovery flow on Words tab, and no-face MP4 warning handling on Vision tab all passed.
   - **All 7 Multimodal Combinations Passed (`200 OK`, weight sum `1.0`)**:
     - `1_text_only` (`anger`, `0.7677`), `2_audio_only` (`surprise`, `0.2976`), `3_video_only` (`neutral`, `0.3302`), `4_text_audio` (`anger`, `0.7677`), `5_text_video` (`surprise`, `0.7426`), `6_audio_video` (`surprise`, `0.2863`), and `7_text_audio_video` (`joy`, `0.8827`).
   - **Responsive Viewports (`0px` Horizontal Overflow Across All 5)**: `1920x1080`, `1366x768`, `768x1024`, `390x844`, and `360x800` all passed with `horizontalOverflowPx: 0` and `submitWidthOk: true`.
   - **Console & Page Errors**: `unexpected_console_errors: []`, `page_errors: []`.
   - **Visual Verification Screenshots**: Captured desktop (`1366x768`) and mobile (`390x844`) views in scratch artifacts and visually inspected cleanly.

3. **Reliability, Failure Modes & Restart Recovery**:
   - **Sequential & 4-Worker Concurrency**: Live HTTP `text`, `audio`, `video`, and `multimodal` requests all returned HTTP **`200`** sequentially and under 4 concurrent worker threads.
   - **Oversized & Invalid Upload Rejections**: `> 50 MiB` upload returned HTTP **`413`**; corrupt `.wav` upload returned HTTP **`415`**; `leaked_temp_media_files: []`.
   - **Process Restart Recovery**: Killed Uvicorn gracefully, restarted in staging mode on port `8018`, polled `/ready` (`200`), and re-verified `/predict/multimodal` (`200`, `joy`).
   - **Fallback Recovery Staging Instance (`port 8019`)**: Simulated corrupt primary ResNet18 SHA-256 (`ffff...`); server activated focal-loss fallback (`active_role: "fallback"`, SHA-256 `4830ac84...`), passed `/ready` (`200`), and served `/predict/video` (`200`).

---

## 3. External Hosting Gate (`AWAITING USER APPROVAL`)

No external server, cloud bucket, domain, DNS record, or TLS certificate was provisioned in this phase. External deployment actions await explicit user selection of a target host and domain (`docker compose -f docker-compose.staging.yml --env-file .env.staging --profile tls up -d`).

