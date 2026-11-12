# Phase 19A — Physical Device & Live Capture Validation

- **Date:** 2026-11-11
- **Subphase Status:** **`AWAITING USER VERIFICATION`** *(All automatable UI error-recovery, format, cleanup, and resource checks **PASSED**; physical-device hardware checklist below requires hands-on user execution.)*
- **Artifacts:**
  - [`scripts/validate_phase19a_hardware.py`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/scripts/validate_phase19a_hardware.py) -> [`models/facial_emotion/resnet18/phase19a_hardware_validation.json`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/facial_emotion/resnet18/phase19a_hardware_validation.json)
  - [`scripts/e2e_phase19a_live_capture.cjs`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/scripts/e2e_phase19a_live_capture.cjs) -> [`models/facial_emotion/resnet18/phase19a_browser_validation.json`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/models/facial_emotion/resnet18/phase19a_browser_validation.json)

---

## 1. Code Audit Findings & Fixes Applied

Inspecting [`frontend/src/LiveCapture.tsx`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/frontend/src/LiveCapture.tsx), [`frontend/src/media.ts`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/frontend/src/media.ts), and [`frontend/src/App.tsx`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/frontend/src/App.tsx) revealed five gaps addressed safely before testing:

1. **Missing Camera & Microphone Selection Dropdowns:** Added `useMediaDevices()` (`enumerateDevices()` + live `"devicechange"` listener + refresh upon stream grant) and `DeviceSelect` controls (`aria-label="Select microphone device"` and `aria-label="Select camera device"`) so users can switch between built-in and virtual/external cameras or microphones directly in the UI, with automatic preview switch when changing active cameras and `OverconstrainedError` fallback.
2. **Missing Mid-Capture Disconnect Detection:** Added `track.addEventListener("ended", ...)` on active microphone and camera tracks so unplugging a USB device or terminating a virtual camera mid-recording stops the recorder safely, surfaces a clear error message, and never crashes or hangs the interface.
3. **UI Upload Limit Mismatch (`100 MB` vs Backend `50 MB`):** Updated [`App.tsx`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/frontend/src/App.tsx) multimodal video hint text from `max 100 MB` to `max 50 MB` (matching `ApiSettings.max_upload_bytes`) and added `validateUploadFileSize()` (`50 MB`) in [`media.ts`](file:///c:/Users/Bhavanth%20D/OneDrive/Pictures/Desktop/Multimodal-Emotion-Recognition-main/Multimodal-Emotion-Recognition-main/frontend/src/media.ts) across both uploads and finalized live recordings.
4. **Non-Fatal Live Audio Meter Initialization:** Isolated `AudioContext` RMS meter setup in its own `try/catch` inside `LiveAudioCapture` so environments or audio drivers that restrict `createMediaStreamSource()` never block microphone capture.
5. **Unmount & Expanded Error Guards:** Added `mountedRef` teardown guards against post-unmount state updates or leaked object URLs, plus explicit handling for `OverconstrainedError`, `AbortError`, and `SecurityError`.

---

## 2. Automated Verification Results (Clearly Separated from Physical Hardware Tests)

### 2.1 Live API Upload, Format & Stress Verification (`MELD dev` + Synthetic Inputs)

| Scenario | Endpoint | HTTP Status | Latency (ms) | Notes / Output |
|---|---|---|---|---|
| Real speech WAV (`dev_dia0_utt0`) | `/predict/audio` | `200` | `366.82` | `surprise` |
| Unsupported `.ogg` file | `/predict/audio` | `415` | — | Clean unsupported-signature rejection |
| Silent 1s WAV | `/predict/audio` | `422` | — | Explicit silence rejection |
| Multi-face video (`dev_dia0_utt0`) | `/predict/video` | `200` | `1056.04` | `21` detections, `8` frames, `coverage = 1.0`, `neutral` |
| Short video (`dev_dia99_utt3`) | `/predict/video` | `200` | `275.61` | `2` detections, `2` frames, `neutral` |
| Long video (`dev_dia74_utt19`) | `/predict/video` | `200` | `730.18` | `9` detections, `8` sampled frames, `neutral` |
| Low-light video (`dev_dia10_utt3`) | `/predict/video` | `200` | `527.87` | `24` detections, `8` sampled frames, `neutral` |
| VP9 WebM clip (FFmpeg live-style) | `/predict/video` | `200` | `583.05` | `neutral` |
| Blank no-face video | `/predict/video` | `422` | — | Clean rejection, no fabricated `neutral` |
| Corrupt MP4 payload | `/predict/video` | `422` | — | Safe decode rejection |
| Multimodal Audio + Video + Text | `/predict/multimodal` | `200` | `628.95` | `candidate4_plus_validated_resnet18_video_v1` |
| Temporary file cleanup (`%TEMP%`) | OS temp dir | — | — | `temp_files_leaked = []` (0 leaked files) |

- **Observed Process & GPU Telemetry:** `process_rss_mb = 1521.2 MB`, `cuda_available = true` (RTX 4050), `cuda_peak_allocated_mb = 1231.29 MB`, `cuda_reserved_mb = 1370.0 MB`.

### 2.2 Chrome Browser Enumeration & Automated UI State Tests (`uiFaultInjectionChecks`)

- **Real Browser Pre-Permission Enumeration (No fake Chrome flags):** `enumerateDevices()` supported; detected `1` `audioinput`, `1` `videoinput`, `1` `audiooutput` (`entriesCount: 3`).
- **UI State & Error-Handler Injection Checks** *(simulated browser API faults to test React UI handlers — NOT counted as physical hardware verification)*:
  - Microphone & camera selector dropdowns rendered and switched (`2` devices each): **PASS**
  - Denied permissions (`NotAllowedError`) on mic & camera: **PASS** (friendly error displayed, zero crash)
  - Missing microphone (`NotFoundError`) & busy camera (`NotReadableError`): **PASS**
  - Mid-recording `MediaStreamTrack` `"ended"` disconnect on mic & camera: **PASS**
  - Client-side `> 50 MB` upload size guard: **PASS**
  - Browser `consoleErrors = []`, `pageErrors = []`.

---

## 3. Manual Physical Device Testing Checklist (`AWAITING USER VERIFICATION`)

Please execute these tests interactively in a local browser (`http://localhost:5173`) with physical hardware connected and record actual results before public deployment:

| # | Scenario | Action | Expected Outcome | Status | Actual Notes |
|---|---|---|---|---|---|
| 1 | Built-in laptop webcam preview & capture | Switch Face or Multimodal Video dropdown to **Live camera**, select built-in camera, click **Preview camera**, then **Start recording** for 3–5s, click **Stop capture**, play preview, click **Analyze** | Live mirrored preview appears; recorded MP4/WebM plays smoothly; API returns 200 emotion distribution + face coverage | **AWAITING USER VERIFICATION** | |
| 2 | External / virtual camera switching | Attach USB/virtual camera, verify camera selector shows multiple devices, switch camera in dropdown while preview is active | Preview switches to newly selected camera without page reload or error | **AWAITING USER VERIFICATION** | |
| 3 | Built-in & external microphone capture | Switch Speech or Multimodal Audio dropdown to **Live microphone**, pick mic in selector, record 3–5s speech, click **Stop capture**, play WAV audio, click **Analyze** | Live RMS level bar moves with voice; 16 kHz WAV file plays cleanly; `/predict/audio` returns 200 | **AWAITING USER VERIFICATION** | |
| 4 | Camera permission denied | Click browser site lock icon -> set Camera to **Block**, click **Preview camera** | Friendly banner: *"Camera access was denied. Allow browser permission and retry."* No page crash | **AWAITING USER VERIFICATION** | |
| 5 | Microphone permission denied | Block Microphone in browser settings, click **Start recording** | Friendly banner: *"Microphone access was denied. Allow browser permission and retry."* No page crash | **AWAITING USER VERIFICATION** | |
| 6 | Device disconnected mid-recording | Start recording with external USB camera/mic, unplug device during recording | Capture stops gracefully and displays disconnection message without browser freeze | **AWAITING USER VERIFICATION** | |
| 7 | Low-light camera input | Record 3–5s in dimly lit room and click **Analyze video** | Either predicts emotion when MTCNN finds face or rejects cleanly (`422` no detectable face) | **AWAITING USER VERIFICATION** | |
| 8 | No-face camera video | Point camera at blank wall or cover lens, record 3s, click **Analyze video** | Backend returns explicit 422 message (*No detectable faces*), zero temp files left | **AWAITING USER VERIFICATION** | |
| 9 | Multiple-face camera video | Have 2+ people in camera frame, record 3–5s, click **Analyze video** | Completes with `200`, `num_faces >= 2` in Signal Quality card, finite 7-class probabilities | **AWAITING USER VERIFICATION** | |
| 10 | Short (<1.5s) & near-limit (~24s) clips | Test very short clip (`< 0.5s` audio or `1s` video) and long clip up to auto-stop limit (`25s` video / `45s` audio) | Very short empty clips prompt to record longer; max duration auto-stops cleanly without exceeding 50 MB | **AWAITING USER VERIFICATION** | |

