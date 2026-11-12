# Phase 20G — Monitoring, Backup, Recovery & Rollback

**Status:** Completed  
**Scope:** Non-PII Runtime Telemetry (`/metrics`), Prometheus/Grafana Provisioning, Snapshot Backup/Restore Tooling & Rollback Runbook

---

## 1. Runtime Telemetry & Monitoring Stack

1. **Zero-PII Prometheus Endpoint ([`api/telemetry.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/telemetry.py), `GET /metrics` in [`api/app.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/app.py#L184-L197))**:
   - Exposes `text/plain; version=0.0.4; charset=utf-8` metrics without recording raw text, filenames, prediction classes, query strings, or client IPs (`route` labels are strictly mapped to `KNOWN_ROUTES` or `"other"`).
   - Tracks:
     - `emotion_api_models_ready` (gauge `0`/`1`) and `emotion_api_model_load_duration_ms` (gauge).
     - `emotion_api_process_rss_bytes` and `emotion_api_cuda_allocated_bytes` (gauges sampled dynamically per scrape).
     - `emotion_api_requests_total{route,method,status_class}` and `emotion_api_request_errors_total{route}` (counters).
     - `emotion_api_request_duration_seconds_bucket` / `_sum` / `_count` (histogram with buckets `0.05s` to `60s` + `+Inf`).
     - `emotion_api_upload_rejections_total{reason}` (`size_limit`, `media_validation`) and `emotion_api_cleanup_failures_total`.

2. **Staging Observability Stack ([`deploy/monitoring/`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/deploy/monitoring/prometheus.yml), [`docker-compose.staging.yml`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/docker-compose.staging.yml#L109-L139))**:
   - Optional `monitoring` Docker Compose profile runs `prom/prometheus:v2.55.1` (scraping `backend:8000/metrics` every `15s`) and `grafana/grafana:11.3.1` with provisioned dashboard `Multimodal Emotion Studio — Staging Telemetry` bound strictly to `127.0.0.1`.
   - Validated via `docker compose -f docker-compose.staging.yml --env-file .env.staging.example --profile monitoring --profile tls config --quiet` (exit code `0`).

---

## 2. Backup, Verification & Non-Destructive Restore Drill

- Implemented [`scripts/backup_staging_state.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/scripts/backup_staging_state.py) (`backup`, `verify`, `restore`) covering all **24 critical staging assets** (11 runtime model files, `api/config.py`, `api/fusion.py`, Compose/Dockerfiles, Nginx/Caddy/Prometheus configs, and `.env*.example`).
- **Executed Real 24-File Drill**:
  - `backup` verified `24` files (`mismatches: []`, `ok: true`).
  - `restore` copied and re-verified SHA-256 digests on all `24` files into an isolated drill directory (`restored_files: 24`, `ok: true`), followed by clean temporary drill cleanup.
- **Automated Tests (`task-4422`)**: [`api/tests/test_phase20g_monitoring_and_backup.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/api/tests/test_phase20g_monitoring_and_backup.py) passed **3 / 3** in `8.72s` (Prometheus telemetry + PII exclusion, temp-cleanup failure tracking, and corruption rejection during verify/restore).

---

## 3. Staging Rollback & Recovery Procedures

1. **Automatic ResNet18 Fallback (Primary Corrupt/Unavailable)**:
   - Built into `ModelRegistry`: fails over automatically to weighted focal-loss checkpoint `exp5_focal_g3_wt_best.pt` (`4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0`) whenever primary SHA-256 fails.
2. **Instant Rollback to Legacy CNN Video Predictor**:
   - Set `EMOTION_API_FACIAL_BACKEND=legacy_cnn` in `.env.staging` and restart backend (`docker compose -f docker-compose.staging.yml --env-file .env.staging up -d --force-recreate backend`).
3. **Configuration / Snapshot Restore**:
   - Run `python scripts/backup_staging_state.py verify --snapshot-dir <snapshot>` followed by `restore` to recover exact verified model artifacts and Compose/proxy configuration.
4. **Full Stack Container Rollback**:
   - Tag validated staging images prior to upgrade (`docker tag emotion-studio-backend:staging emotion-studio-backend:staging-prev`) and revert with `docker compose --env-file .env.staging up -d --force-recreate`.

