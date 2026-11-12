# Phase 19C — Privacy, Media Lifecycle & Data Protection Review

**Date:** 2026-11-11  
**Technical Control Status:** `PASS`  
**Regulatory / Legal Sign-Off Status:** `PENDING QUALIFIED LEGAL COUNSEL REVIEW`  
**Scope:** Local development workstation (`http://localhost:8000` + `http://localhost:5173`, `deployed_to_production = False`)

---

## 1. End-to-End Media & Text Lifecycle Inventory

| Stage | Text Utterance | Uploaded / Live Audio (`.wav`) | Uploaded / Live Video (`.mp4` / `.webm`) |
|---|---|---|---|
| **1. Collection & Consent** | User enters text manually or submits multimodal form (`App.tsx`). | File upload OR Live Mic (`LiveAudioCapture` in `LiveCapture.tsx`) gated behind explicit affirmative consent checkbox before `getUserMedia()`. | File upload OR Live Camera (`LiveVideoCapture` in `LiveCapture.tsx`) gated behind explicit affirmative consent checkbox before `getUserMedia()`. |
| **2. Browser Processing** | Held strictly in React component state (`text`); never saved to `localStorage`, `sessionStorage`, or `IndexedDB`. | Live Mic converts PCM samples into 16 kHz mono WAV in browser memory (`blobToWavFile()` in `media.ts`); object URL preview revoked on clear/re-record/unmount. | Live Camera records WebM/MP4 via `MediaRecorder` in browser memory; object URL preview revoked on clear/re-record/unmount. `MediaStream` tracks stopped on component unmount or mode switch. |
| **3. Transport** | HTTPS/HTTP localhost (`POST /predict/text` or `/predict/multimodal`), `Cache-Control: no-store`. | `multipart/form-data` (`POST /predict/audio` or `/predict/multimodal`), max `50 MB`, `Cache-Control: no-store`. | `multipart/form-data` (`POST /predict/video` or `/predict/multimodal`), max `50 MB`, max `60s`, `Cache-Control: no-store`. |
| **4. Backend Handling** | Validated in memory (`TextRequest` max 20,000 chars); passed directly to DistilBERT tokenizer. | Magic-byte validated (`RIFF...WAVE`) and streamed via `store_upload()` (`api/validation.py`) to OS temp dir (`emotion_*.wav`) ignoring user filename. | Magic-byte validated (`ftyp` MP4 or EBML `webm`) and streamed via `store_upload()` to OS temp dir (`emotion_*.mp4/.webm`) ignoring user filename. |
| **5. Retention & Deletion** | Not written to disk, database, or logs. | `cleanup_file(path)` (`unlink(missing_ok=True)`) invoked in `finally` blocks in `_predict_file()` and `predict_multimodal()` immediately after inference. | `cleanup_file(path)` (`unlink(missing_ok=True)`) invoked in `finally` blocks in `_predict_file()` and `predict_multimodal()` immediately after inference; decoded OpenCV frames live in volatile RAM only. |
| **6. Third-Party Transfers** | None. Inference executes 100% locally. | None. Inference executes 100% locally. | None. Inference executes 100% locally (note: `index.css` loads Google Fonts stylesheets unless self-hosted prior to air-gapped production). |

---

## 2. Biometric, Acoustic & Facial Processing Protections

1. **Purpose Limitation & Ephemeral Processing:**
   - The system computes transient 7-class emotion probabilities (`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`) without face recognition, speaker identification, persistent face embeddings/templates, or longitudinal user profiling.
   - Pre-capture consent disclosure (`MediaConsentNotice` in `frontend/src/LiveCapture.tsx`) requires affirmative checkbox opt-in before activating microphone or camera hardware (`getUserMedia`).
2. **Non-Diagnostic Disclosure & Ethical AI Safeguards:**
   - Every prediction card (`UncertaintyNotice` in `frontend/src/App.tsx`) and footer clearly states that predictions are probabilistic AI estimates and not medical or psychological diagnoses.
   - Low-confidence predictions (`< 45%`) highlight an explicit low-confidence caveat.
3. **Zero Silent Recording & Hardware Release:**
   - Recording indicators (`Live mic level`, `Recording in progress...`, countdown timers, active preview stop/clear buttons) remain visible throughout capture, and `useEffect` unmount hooks stop `AudioContext`, cancel animation frames, and stop all `MediaStreamTrack`s on navigation or close.

---

## 3. Jurisdiction & Governance Review Notes (DPDP Act 2023 / GDPR Readiness)

> [!IMPORTANT]
> The items below are technical engineering observations, **not** legal advice. Before exposing Emotion Studio outside local development or collecting real end-user voice/video data, obtain formal review from qualified privacy/legal counsel.

1. **Indian Digital Personal Data Protection (DPDP) Act, 2023 Considerations:**
   - **Notice & Affirmative Consent (Sec. 6):** Notice is surfaced pre-capture via `MediaConsentNotice` (`LiveCapture.tsx`). If deployed publicly, pair this with a formal versioned Privacy Notice page specifying Data Fiduciary contact details, grievance officer contact, and withdrawal workflow.
   - **Purpose Limitation & Data Minimization (Sec. 4 & 8):** No permanent media storage or databases exist (`temp_files_leaked: []`). Audit access logs on reverse proxies before any public hosting to ensure request bodies and filenames are excluded.
   - **Children's Data (Sec. 9):** Public deployment requires age-gating or verifiable parental consent mechanisms if accessible to minors.
   - **Cross-Border / External Font Note:** `frontend/src/index.css` imports fonts from `fonts.googleapis.com`; self-host font assets locally before offline/strict-privacy deployment to avoid leaking client IP headers to third-party CDNs.
2. **GDPR / International Biometric Data Context (if serving EU/UK users):**
   - While facial crops and voice buffers are deleted immediately and not used for unique identification (template matching), affect computing over live audio/video triggers heightened transparency obligations (DPIA, Art. 9/13 disclosures, and strict avoidance of prohibited workplace/educational emotion-inference contexts under EU AI Act Art. 5(1)(f)).

