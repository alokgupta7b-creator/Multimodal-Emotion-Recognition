# Phase 19C — Application Security Audit & Hardening Report

**Date:** 2026-11-11  
**Audit Scope:** FastAPI backend (`api/`), upload & temp-file lifecycle (`api/validation.py`), React frontend (`frontend/src/`), model loading integrity (`api/model_registry.py`, `facial_emotion/resnet_predictor.py`), dependencies, and runtime secrets (`deployed_to_production = False`).  
**Verification Status:** `PASS` (for local deployment scope; see Medium/Low advisories below before any public exposure).

---

## 1. Executive Summary & Verification Evidence

- **Phase 19C Security Test Suite (`api/tests/test_phase19c_security.py`):** **`8 passed` in `50.19s`** (`python -m pytest`).
- **Live Capture Consent Gate & Upload Cleanup (`scripts/validate_phase19a_hardware.py`):** **`automated_passed: true`**, `temp_files_leaked: []`, Chrome verified consent required before `getUserMedia()`.
- **Frontend Dependency Audit (`npm.cmd --prefix frontend audit --json`):** **`0 vulnerabilities`** (`info: 0`, `low: 0`, `moderate: 0`, `high: 0`, `critical: 0` across `212` dependencies).
- **Python Dependency Vulnerability Scanner:** `pip-audit` / `safety` are **not installed** in the local environment (`NOT RUN` for automated PyPI CVE matching); exact installed library versions were extracted and inspected manually below.
- **Secret / Credential Scan (`rg` across source/scripts/docs):** **0 matches** (`AKIA...`, private keys, tokens, hardcoded credentials). No `.env` file with live credentials exists in the workspace.

---

## 2. Severity-Classified Findings & Remediations

### Critical / High Findings
None identified in local workspace execution scope. Model checkpoints (`exp5_ce_weighted_best.pt`, `exp5_focal_g3_wt_best.pt`, `distilbert`, `wav2vec2`) are pinned to local directories and SHA-256 verified before `torch.load` (`weights_only=True`).

### Medium Severity Findings

| ID | Component | Risk / Issue Observed | Remediation Applied / Status |
|---|---|---|---|
| **SEC-M1** | `frontend/src/LiveCapture.tsx` | Live mic/camera could activate `getUserMedia()` directly on button click without explicit local-processing consent disclosure. | **Fixed (`PASS`):** Added `MediaConsentNotice` checkbox + ephemeral-processing disclosure gating `Start recording` and `Preview camera` in both `LiveAudioCapture` and `LiveVideoCapture`. Verified in Chrome E2E (`validate_phase19a_hardware.py`). |
| **SEC-M2** | `api/app.py` | Inference routes (`/predict/text`, `/predict/audio`, `/predict/video`, `/predict/multimodal`) lacked rate limiting, enabling repeated CPU/GPU exhaustion via rapid local or LAN floods. | **Fixed (`PASS`):** Added thread-safe `SlidingWindowRateLimiter` (`EMOTION_API_RATE_LIMIT_REQUESTS=120` per `EMOTION_API_RATE_LIMIT_WINDOW_SECONDS=60` per client IP) emitting HTTP `429` + `Retry-After`. Verified via `test_sliding_window_rate_limiter_enforces_429_and_recovers`. |
| **SEC-M3** | `api/app.py` | CORS configuration paired `allow_credentials=bool(settings.allowed_origins)` with `allow_headers=["*"]`, and responses lacked baseline hardening headers or `Cache-Control: no-store`. | **Fixed (`PASS`):** Restricted `allow_headers=["Content-Type", "Accept"]` and added HTTP security headers middleware (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Permissions-Policy`, `Cross-Origin-Resource-Policy: same-site`, `Cache-Control: no-store` on `/predict/*`). |
| **SEC-M4** | `api/app.py` (`/predict/multimodal`) | `RuntimeError` exceptions from audio/video model execution were formatted directly into HTTP `exclusion_reasons` (`f"Model unavailable: {exc}"`), risking internal filesystem path disclosure. | **Fixed (`PASS`):** Sanitized `RuntimeError` exclusion strings to `"Audio model unavailable."` and `"Video model unavailable."`. Verified via `test_runtime_error_details_are_sanitized_in_multimodal_exclusion`. |

### Low Severity Findings

| ID | Component | Risk / Issue Observed | Remediation Applied / Status |
|---|---|---|---|
| **SEC-L1** | `api/config.py`, `api/app.py` | Swagger/OpenAPI endpoints (`/docs`, `/redoc`, `/openapi.json`) were unconditionally enabled regardless of deployment environment. | **Fixed (`PASS`):** Added `EMOTION_API_ENABLE_DOCS` (`enable_docs` setting) to disable interactive API docs (`docs_url=None`, `redoc_url=None`, `openapi_url=None`) when set to `0`/`false`/`no`. Verified in `test_docs_toggle_disables_openapi_when_configured`. |
| **SEC-L2** | `api/validation.py` | Upload storage already ignored client filenames via `tempfile.NamedTemporaryFile(prefix="emotion_", suffix=...)`, but lacked a dedicated filename sanitizer helper if labels are ever displayed/logged. | **Fixed (`PASS`):** Added `sanitize_upload_label()` and verified path-traversal upload filenames (`../../evil_payload.wav`) stay strictly inside OS tempdir (`test_magic_byte_spoofing_and_path_traversal_filename_safe`, `test_temp_files_strictly_inside_tempdir_and_cleaned`). |

### Informational / Pre-Public Deployment Advisories

| ID | Area | Observation & Required Action Before Public Hosting |
|---|---|---|
| **SEC-I1** | Python Dependencies (`starlette==0.38.6`, `fastapi==0.115.0`) | `pip list` shows `starlette==0.38.6`, `fastapi==0.115.0`, `python-multipart==0.0.27`, `pillow==12.3.0`, `torch==2.6.0+cu124`, `transformers==5.2.0`, `cryptography==46.0.3`, `uvicorn==0.32.0`, `pydantic==2.10.6`. While outer streaming middleware (`limit_upload_size_middleware`) caps payload size to `50 MB` before multipart parsing, `starlette==0.38.6` predates newer multipart advisories (`>=0.40.0`/`0.41.2`). Install `pip-audit` in CI and upgrade FastAPI/Starlette prior to public release. |
| **SEC-I2** | Transport Security & Third-Party CSS Fonts | Localhost dev runs over HTTP (`http://localhost:5173` / `http://127.0.0.1:8000`). Public deployment requires TLS/HTTPS (also required by browsers for `getUserMedia` off localhost), proxy-level distributed rate limiting, and self-hosting the Inter/Plus Jakarta Sans fonts currently referenced from `fonts.googleapis.com` in `frontend/src/index.css`. |
| **SEC-I3** | Logging Hygiene | Code inspection (`rg`) confirms neither `api/` nor `facial_emotion/` logs raw user utterances, uploaded filenames, or raw media buffers. Note that Uvicorn HTTP access logs are active by default and should avoid request body logging behind reverse proxies. |

---

## 3. OWASP API / Web Control Checklist (Verified Local Posture)

| Control | Mechanism | Verified By |
|---|---|---|
| Upload size & streamed bomb guard | `limit_upload_size_middleware` (`50 MB`) + chunked `store_upload()` | `validate_phase19a_hardware.py`, `api/tests/test_api.py` |
| Magic-byte validation | `RIFF...WAVE` (WAV), `ftyp` (MP4), EBML `webm` (WebM) | `test_phase19c_security.py` (PE/MZ `.exe` spoof -> `415`) |
| Path traversal resistance | `tempfile.NamedTemporaryFile` ignores `upload.filename` | `test_temp_files_strictly_inside_tempdir_and_cleaned` |
| Temp file cleanup | `cleanup_file()` inside `finally` blocks across all upload endpoints | `validate_phase19a_hardware.py` (`temp_files_leaked: []`) |
| Security response headers & no-cache | `nosniff`, `DENY`, `no-referrer`, `same-site`, `no-store` | `test_security_headers_and_no_store_on_predict` |
| Brute-force / inference DoS throttle | Per-IP `SlidingWindowRateLimiter` (`429` + `Retry-After`) | `test_sliding_window_rate_limiter_enforces_429_and_recovers` |
| Model checkpoint supply-chain integrity | Locked SHA-256 verification before `torch.load(weights_only=True)` | `api/tests/test_api.py`, `facial_emotion/resnet_predictor.py` |

