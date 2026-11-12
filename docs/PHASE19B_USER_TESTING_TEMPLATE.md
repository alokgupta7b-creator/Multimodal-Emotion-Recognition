# Phase 19B — Human Participant UAT Protocol & Feedback Template

**Status:** `AWAITING USER VERIFICATION`  
**Important:** Do not record fabricated responses. Each participant should test the live application independently against local API `http://localhost:8000` and frontend `http://localhost:5173`.

---

## Participant Metadata

| Field | Value |
|---|---|
| Participant ID / Alias (non-PII) | |
| Role / Familiarity (Technical / Non-Technical) | |
| Operating System & Browser | |
| Viewport Tested (Desktop / Laptop / Tablet / Mobile) | |
| Date / Session Duration | |

---

## Structured UAT Test Matrix

Please record exact observed outcomes verbatim:

| ID | Scenario | User Action | Expected Outcome | Actual Outcome | Pass / Fail | Participant Feedback |
|---|---|---|---|---|---|---|
| UAT-01 | Text-only quick check | Open **Text** tab, enter a clear sentence (e.g., `"I cannot believe you pulled this off!"`), click **Analyze text**. | Clear predicted emotion, confidence percentage, 7 probability bars, and probabilistic estimate notice. | | | |
| UAT-02 | Empty / invalid input handling | Clear Text box and click **Analyze text**; upload corrupt or 0-face clip on **Face** tab. | Clear plain-language error alert with a working **Retry** option; app does not freeze or crash. | | | |
| UAT-03 | Uploaded WAV speech check | Open **Speech** tab, upload a short `.wav` clip, click **Analyze audio**. | Audio emotion, confidence, and SNR/clipping reliability metrics appear cleanly. | | | |
| UAT-04 | Uploaded MP4/WebM face check | Open **Face** tab, upload a short face video, click **Analyze video**. | Facial emotion, confidence, and face coverage metrics appear cleanly. | | | |
| UAT-05 | Live microphone speech capture | Switch Speech source dropdown to **Live microphone**, acknowledge local-processing disclosure, record 3–5s, play back, run inference. | Mic selector dropdown works, waveform/timer updates, WAV converts cleanly, and prediction completes. | | | |
| UAT-06 | Live camera video capture | Switch Face source dropdown to **Live camera**, acknowledge disclosure, preview camera, record 3–5s, play back, run inference. | Live camera preview matches framing, playback works, and facial prediction completes cleanly. | | | |
| UAT-07 | All 7 multimodal combinations | In **Multimodal Studio**, test: (1) Text, (2) Audio, (3) Video, (4) Text+Audio, (5) Text+Video, (6) Audio+Video, (7) Text+Audio+Video. | Active/Excluded/Advisory labels, modality weights summing to `1.00`, and individual model breakdown remain understandable. | | | |
| UAT-08 | Conflicting signals interpretation | Submit joyful text paired with an angry/furious tone or expression in Multimodal Studio. | Rule banner and individual modality cards explain why final prediction chose the fused outcome. | | | |
| UAT-09 | Error recovery & retry flow | Stop backend briefly, trigger prediction, restart backend, click **Retry** button on the alert banner. | Informative connection error with **Retry** button; retrying succeeds without losing input context. | | | |
| UAT-10 | Mobile responsive layout | Open UI at phone width (~360–390 px), switch tabs, submit an analysis, inspect probability bars. | Zero horizontal scrolling, readable text, accessible buttons, clean stacking. | | | |
| UAT-11 | Keyboard-only navigation | Navigate tabs (`Multimodal`, `Text`, `Speech`, `Face`), Skip link, and Submit using `Tab` / `Enter`. | Visible cyan focus outline on all controls and clean focus order. | | | |
| UAT-12 | Uncertainty & non-diagnostic clarity | Review the result banner notice when confidence is below 45% vs above 75%. | Participant clearly understands output is statistical and non-diagnostic. | | | |

---

## Qualitative Summary Questions

1. **Clarity of labels & fusion weights:** Did you understand why a modality showed *Active*, *Advisory*, or *Excluded*?  
   - Rating (1–5): ___  
   - Comments: 

2. **Live capture comfort & trust:** Were permission prompts, recording indicator (`REC`), playback review, and local-processing notices clear?  
   - Rating (1–5): ___  
   - Comments: 

3. **Confusion points or defects observed:**  
   - Bugs / UX friction list: 

