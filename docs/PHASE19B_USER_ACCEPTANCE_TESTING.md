# Phase 19B — User Acceptance Testing & Usability Validation

**Date:** 2026-11-11  
**Automated Browser & E2E UAT Status:** `PASS`  
**Human Participant UAT Status:** `AWAITING USER VERIFICATION`  
**MELD Test Split Access:** `None` (validation used MELD development clip `dev_dia0_utt0` plus synthetic empty/silent inputs)

---

## 1. UX, Accessibility & Transparency Improvements Applied

Implemented targeted reliability, clarity, and accessibility fixes before UAT execution:

1. **Error recovery & request timeout (`frontend/src/api.ts`, `frontend/src/App.tsx`):**
   - Added a 60-second `AbortController` request timeout (`REQUEST_TIMEOUT_MS = 60_000`) and clear connection/timeout error wording in `frontend/src/api.ts`.
   - Added a **Retry** button inside the `role="alert"` prediction error banner (`retryAction` callback in `App.tsx`) so transient outages or timeouts recover in one click without losing input text or attached files.
2. **Probabilistic uncertainty & non-diagnostic notice (`frontend/src/App.tsx`):**
   - Added the shared `UncertaintyNotice` component to both `MultimodalResultCard` and `ResultCard` (highlighting low-confidence estimates `< 45%` and explicitly cautioning that predictions are statistical, non-diagnostic AI estimates).
   - Added footer non-diagnostic disclaimer (`Phase 19 · Adaptive Multimodal Studio · Non-diagnostic probabilistic estimates`).
3. **Accessibility & WCAG-AA dark contrast (`frontend/src/App.tsx`, `frontend/src/index.css`):**
   - Added keyboard focus-visible outline styling (`:focus-visible { outline: 2px solid #4de1d2; outline-offset: 2px; }`).
   - Added a `"Skip to main content"` landmark link (`#main-content`), `role="tablist"` / `role="tab"` / `aria-selected` semantics on the 4 modality tabs, `aria-live="polite"` on API status and result cards, and upgraded muted/footer/empty-state dark slate contrast (`text-slate-600` / `text-slate-700` → `text-slate-400` / `text-slate-300`).

---

## 2. Automated Browser E2E Verification (`scripts/run_phase19b_uat.py` / `scripts/e2e_phase19b_uat.cjs`)

Evidence artifact: `models/facial_emotion/resnet18/phase19b_uat_validation.json`  
Result: **`"passed": true`**, **`unexpectedConsoleErrors: []`**, **`pageErrors: []`** (note: `allConsoleErrorsCount: 2` reflects the two deliberate negative-test browser console logs for injected `503` and `422 no-face`).

### A. Accessibility & Error Recovery Checks

| Check | Observed Outcome | Status |
|---|---|---|
| Skip link (`a[href="#main-content"]`) & `<main id="main-content">` | Present (`firstFocusedTag: "A"` on `Tab`) | `PASS` |
| Tablist & 4 tabs (`role="tablist"`, `aria-selected`) | `tabCount: 4`, `selectedTabCount: 1` | `PASS` |
| Empty text submission guard | `[role="alert"]` displayed `"Enter some text before analyzing."` | `PASS` |
| Simulated `503` backend outage → **Retry** button click | Alert rendered `Retry`, click recovered cleanly with HTTP `200` (`joy`, `95.08%` confidence) | `PASS` |
| No-face video error surfacing (`face_tab`) | `[role="alert"]` rendered `"No detectable faces found across 8 sampled frames (filtered_fp=0)."` | `PASS` |

### B. All 7 Modality Combinations (Live UI + FastAPI Parity)

Executed inside live headless Chrome (`http://localhost:5173` → `http://127.0.0.1:8000`) using MELD development sample `dev_dia0_utt0` for audio/video inputs:

| Combination | HTTP | Predicted Emotion | Confidence | Fusion Method | Rule Applied | Weights (T / A / V) | Weight Sum | Uncertainty Notice Rendered |
|---|---|---|---|---|---|---|---|---|
| 1. Text only | 200 | `anger` | 0.7677 | `text_anchored_class_conditional_bounded_fusion_v1` | `text_anchor_default` | `1.00 / 0.00 / 0.00` | `1.0` | Yes |
| 2. Audio only | 200 | `surprise` | 0.2977 | `single_modality_audio_wav2vec2_v1` | `audio_only_fallback` | `0.00 / 1.00 / 0.00` | `1.0` | Yes |
| 3. Video only | 200 | `neutral` | 0.3301 | `single_modality_video_resnet18_v1` | `video_only_resnet18` | `0.00 / 0.00 / 1.00` | `1.0` | Yes |
| 4. Text + Audio | 200 | `anger` | 0.7677 | `text_anchored_class_conditional_bounded_fusion_v1` | `text_anchor_default` | `1.00 / 0.00 / 0.00` | `1.0` | Yes |
| 5. Text + Video | 200 | `surprise` | 0.7426 | `candidate4_plus_validated_resnet18_video_v1` | `text_anchor_default+validated_video_calibrated` | `0.92 / 0.00 / 0.08` | `1.0` | Yes |
| 6. Audio + Video | 200 | `surprise` | 0.2864 | `audio_primary_plus_validated_resnet18_video_v1` | `audio_video_calibrated_blend` | `0.00 / 0.92 / 0.08` | `1.0` | Yes |
| 7. Text + Audio + Video | 200 | `joy` | 0.8826 | `candidate4_plus_validated_resnet18_video_v1` | `protected_class_lockout_joy+validated_video_calibrated` | `0.92 / 0.00 / 0.08` | `1.0` | Yes |

### C. Responsive Layout Across All 5 Required Viewports

| Viewport | `innerWidth` (px) | `scrollWidth` (px) | Horizontal Overflow (px) | Submit Button Inside Viewport | Status |
|---|---|---|---|---|---|
| Desktop Wide | `1920` | `1920` | `0` | Yes | `PASS` |
| Laptop Standard | `1366` | `1366` | `0` | Yes | `PASS` |
| Tablet Portrait | `768` | `768` | `0` | Yes | `PASS` |
| Mobile Modern | `390` | `390` | `0` | Yes | `PASS` |
| Mobile Compact | `360` | `360` | `0` | Yes | `PASS` |

---

## 3. Human Participant UAT & Feedback Log

**Status:** `AWAITING USER VERIFICATION`

Human usability testing with live participants (both technical and non-technical reviewers) has **not** yet been executed by human testers. No human UAT ratings or participant quotes have been fabricated.

Use [`docs/PHASE19B_USER_TESTING_TEMPLATE.md`](PHASE19B_USER_TESTING_TEMPLATE.md) during manual walkthrough sessions covering scenarios `UAT-01` through `UAT-12`, and append verified findings below once complete.

