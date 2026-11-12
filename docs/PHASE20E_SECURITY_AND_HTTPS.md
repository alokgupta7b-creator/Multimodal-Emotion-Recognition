# Phase 20E — Security, HTTPS & Environment Configuration

**Status:** Completed (local verification passed; live Let's Encrypt issuance pending external DNS staging host)

---

## 1. Environment & Secrets Posture

1. **Template Only in Repository**: [`.env.staging.example`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/.env.staging.example) provides clean non-secret defaults (`EMOTION_API_ENV=staging`, `EMOTION_API_ENABLE_DOCS=0`, `EMOTION_API_FAIL_ON_MISSING_MODELS=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, placeholder `*.example.invalid` domains).
2. **Gitignore Protection**: [`.gitignore`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/.gitignore#L152-L158) excludes `.env`, `.env.*`, `backups/`, and `staging_models_bundle/` while keeping `.env.example` and `.env.staging.example` trackable.
3. **Secret Scan (`task-4335`)**:
   - Pattern scan across `api/`, `frontend/src/`, `deploy/`, `scripts/`, `.env.example`, `.env.staging.example`, `docker-compose.staging.yml`, and Dockerfiles returned **`SECRET_PATTERN_MATCHES=0`**.
   - Only `.env.example` (`1,192` B) and `.env.staging.example` (`1,711` B) exist in the workspace root (no unignored live `.env` or `.env.staging`).

---

## 2. TLS, Proxy & Header Hardening

1. **HTTPS & HSTS ([`deploy/caddy/Caddyfile`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/deploy/caddy/Caddyfile))**:
   - Automatic Let's Encrypt TLS termination (`--profile tls`), `Strict-Transport-Security "max-age=31536000; includeSubDomains"`, `X-Content-Type-Options "nosniff"`, `X-Frame-Options "DENY"`, `Referrer-Policy "no-referrer"`, `Permissions-Policy "camera=(self), microphone=(self), geolocation=()"`, server banner removal (`-Server`), and `max_size 52MB` request caps.
   - Live ACME certificate issuance cannot run without a DNS-pointed public staging domain and remains **`AWAITING USER APPROVAL`**.
2. **Frontend Nginx Security Headers & CSP ([`deploy/nginx/security_headers.inc`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/deploy/nginx/security_headers.inc), [`deploy/nginx/default.conf`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/deploy/nginx/default.conf), [`Dockerfile.frontend`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/Dockerfile.frontend))**:
   - Extracted shared headers into `/etc/nginx/security_headers.inc` and included them inside `server`, `location = /healthz`, `location /assets/`, and `location /` so location-level `Cache-Control` `add_header` directives do not suppress Nginx `Content-Security-Policy` or security headers on SPA shells or static bundles.
   - CSP enforces `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; font-src 'self' data:; connect-src 'self' https:; frame-ancestors 'none'; base-uri 'self'; form-action 'self';` (external Google Font requests were removed in Phase 20C).
3. **Backend API Guardrails ([`api/app.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/app.py#L585-L607))**:
   - Wildcard `"*"` in `EMOTION_API_ALLOWED_ORIGINS` is rejected with `ValueError`.
   - Swagger/ReDoc/OpenAPI (`/docs`, `/redoc`, `/openapi.json`) disabled when `EMOTION_API_ENABLE_DOCS=0`.
   - Rate-limiting and `Cache-Control: no-store` evaluation uses `request.scope.get("path", request.url.path)` (exact ASGI routing path) rather than `Host`-dependent `request.url.path`, mitigating `CVE-2026-48710` URL-reconstruction bypass vectors.

---

## 3. Dependency Vulnerability & Test Audit

- **Frontend (`npm audit --json`)**: **`0` vulnerabilities** across 212 packages (verified prior to Phase 20A).
- **Backend (`pip-audit`)**: **NOT RUN** — `pip-audit` is not installed in the offline Python 3.13 environment.
- **Manual PyPI Advisory Review (`requirements.txt` pins vs. upstream CVEs)**:
  - `python-multipart==0.0.27`: includes the fix for `CVE-2026-42561` (`>= 0.0.27`).
  - `starlette==0.38.6` / `fastapi==0.115.0`: `CVE-2026-48710` (`BadHost`, `< 1.0.1`) middleware-path bypass is mitigated in application middleware via `request.scope["path"]`; `CVE-2025-62727` (`FileResponse` range DoS) affects `>= 0.39.0` (`< 0.49.1`) and our API serves no `FileResponse`/`StaticFiles`. Plan future FastAPI/Starlette minor upgrade alongside full regression testing.
  - `transformers==5.2.0`: `CVE-2026-4372` / `CVE-2026-5241` (`config.json` injection RCE on untrusted checkpoints) is operationally mitigated by forcing `local_files_only=True`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and verified local checkpoint directories only; recommend upgrading to `>= 5.3.0` in a dedicated dependency maintenance window.
  - `torch==2.6.0`: defaults to safer `weights_only=True`; upstream component advisories (`CVE-2025-2148` JIT profiler callback tuple handler, `CVE-2025-3730` `ctc_loss` crash) are unrelated to inference paths in this application.
- **Security Test Verification (`task-4346`)**:
  - Command: `pytest api/tests/test_phase20e_security.py api/tests/test_phase19c_security.py -q`
  - Result: **10 passed, 0 failed** in **35.19s** (validates `.env.staging.example` readiness, malformed `Host` header rate-limit/header immunity, upload caps, path traversal/magic-byte rejection, rate limits, CORS, and temp cleanup).

