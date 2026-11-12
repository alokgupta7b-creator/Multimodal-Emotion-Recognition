import assert from "node:assert/strict";
import test from "node:test";
import {
  classifyInputError,
  formatFusionMethod,
  predictFile,
  predictMultimodal,
  predictText,
  request,
  resolveApiBaseUrl,
} from "../src/api.ts";
import {
  analyzeWaveformLevel,
  encodeWavPcm16,
  formatProbabilityPercent,
  measureWavRmsDbFromBuffer,
} from "../src/media.ts";
import { createRequestScope, createRetryableExecution } from "../src/retry.ts";

test("resolveApiBaseUrl strips trailing slashes and preserves relative or staging origins", () => {
  assert.equal(resolveApiBaseUrl(undefined), "http://localhost:8000");
  assert.equal(resolveApiBaseUrl("https://api-staging.example.com///"), "https://api-staging.example.com");
  assert.equal(resolveApiBaseUrl("/api/"), "/api");
  assert.equal(resolveApiBaseUrl(""), "");
});

test("1. request resolves JSON payload cleanly on normal 200 response", async () => {
  const originalFetch = globalThis.fetch;
  let requestedUrl = "";
  globalThis.fetch = async (url, init) => {
    requestedUrl = String(url);
    assert.ok(init?.signal instanceof AbortSignal);
    return new Response(
      JSON.stringify({
        predicted_emotion: "joy",
        probabilities: { anger: 0, disgust: 0, fear: 0, joy: 1, neutral: 0, sadness: 0, surprise: 0 },
        confidence: 1.0,
        inference_duration_ms: 12.5,
        modality: "text",
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  };
  try {
    const result = await predictText("I am thrilled!", { baseUrl: "https://api-staging.example.com/" });
    assert.equal(requestedUrl, "https://api-staging.example.com/predict/text");
    assert.equal(result.predicted_emotion, "joy");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("2. timeout cancellation raises clear timeout message and cleans up timer", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (_url, init) =>
    new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => {
        reject(new DOMException("Aborted", "AbortError"));
      });
    });
  try {
    await assert.rejects(
      () => request("/predict/text", { method: "POST" }, 25, "http://localhost:8000"),
      /Request timed out after 0s\. Please try again\./,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("3. caller cancellation propagates AbortError instead of false timeout message", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (_url, init) =>
    new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => {
        reject(new DOMException("User aborted", "AbortError"));
      });
    });

  const controller = new AbortController();
  const promise = predictText("hello", { signal: controller.signal, timeoutMs: 5_000 });
  controller.abort();
  try {
    await assert.rejects(promise, (err) => {
      assert.equal(err.name, "AbortError");
      assert.doesNotMatch(err.message, /timed out/i);
      return true;
    });

    // Already-aborted caller signal short-circuits before calling fetch
    let fetchCalled = false;
    globalThis.fetch = async () => {
      fetchCalled = true;
      return new Response("{}");
    };
    const preAborted = new AbortController();
    preAborted.abort();
    await assert.rejects(() => predictText("preaborted", { signal: preAborted.signal }), { name: "AbortError" });
    assert.equal(fetchCalled, false);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("4. both caller signal and timeout enforced with zero listener leaks on completion", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (_url, init) => {
    assert.ok(init?.signal);
    return new Response(JSON.stringify({ status: "ready", models: {} }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  let addCalls = 0;
  let removeCalls = 0;
  const controller = new AbortController();
  const signal = controller.signal;
  const origAdd = signal.addEventListener.bind(signal);
  const origRemove = signal.removeEventListener.bind(signal);
  signal.addEventListener = (...args) => {
    addCalls += 1;
    return origAdd(...args);
  };
  signal.removeEventListener = (...args) => {
    removeCalls += 1;
    return origRemove(...args);
  };

  try {
    await request("/models/status", { signal }, 2_000, "http://localhost:8000");
    assert.equal(addCalls, 1);
    assert.equal(removeCalls, 1);

    // Timeout still triggers even when caller passed an active, non-aborted signal
    globalThis.fetch = async (_url, init) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("Timed out", "AbortError"));
        });
      });

    await assert.rejects(
      () => request("/predict/multimodal", { signal, method: "POST" }, 20, "http://localhost:8000"),
      /Request timed out/i,
    );
    assert.equal(addCalls, 2);
    assert.equal(removeCalls, 2);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("5. network failure translates TypeError to actionable API unreachable error", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new TypeError("Failed to fetch");
  };
  try {
    await assert.rejects(
      () => predictMultimodal("test text", null, null, { timeoutMs: 500 }),
      /Unable to reach the Emotion Studio API/,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("6. retry behavior clears previous failure and succeeds on subsequent retry", async () => {
  const originalFetch = globalThis.fetch;
  let attempt = 0;
  globalThis.fetch = async () => {
    attempt += 1;
    if (attempt === 1) {
      return new Response(JSON.stringify({ detail: "Temporary upstream failure" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response(
      JSON.stringify({
        predicted_emotion: "neutral",
        probabilities: { anger: 0, disgust: 0, fear: 0, joy: 0, neutral: 1, sadness: 0, surprise: 0 },
        confidence: 0.92,
        inference_duration_ms: 15,
        modality: "audio",
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  };

  let loadingStates = [];
  let errorMessage = "";
  let prediction = null;
  const scope = createRequestScope();
  const sampleFile = new File([new Uint8Array([82, 73, 70, 70])], "sample.wav", { type: "audio/wav" });

  const execute = createRetryableExecution(
    (signal) => predictFile("audio", sampleFile, { signal, timeoutMs: 1_000 }),
    {
      onStart: () => {
        loadingStates.push(true);
        errorMessage = "";
      },
      onSuccess: (val) => {
        prediction = val;
      },
      onError: (msg) => {
        errorMessage = msg;
      },
      onFinally: () => {
        loadingStates.push(false);
      },
      fallbackError: "Prediction failed",
    },
    scope,
  );

  try {
    const firstOk = await execute();
    assert.equal(firstOk, false);
    assert.equal(errorMessage, "Temporary upstream failure");
    assert.equal(prediction, null);

    const retryOk = await execute();
    assert.equal(retryOk, true);
    assert.equal(errorMessage, "");
    assert.equal(prediction?.predicted_emotion, "neutral");
    assert.deepEqual(loadingStates, [true, false, true, false]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("7. formatFusionMethod removes internal candidate identifiers from UI labels", () => {
  const cases = [
    null,
    undefined,
    "",
    "text_anchored_class_conditional_bounded_fusion_v1",
    "candidate4_plus_validated_resnet18_video_v1",
    "single_modality_video_resnet18_v1",
    "single_modality_audio_wav2vec2_v1",
    "audio_primary_plus_validated_resnet18_video_v1",
    "Candidate_9_custom_branch",
  ];
  for (const input of cases) {
    const label = formatFusionMethod(input);
    assert.ok(label.length > 0);
    assert.doesNotMatch(label, /candidate/i);
  }
  assert.equal(formatFusionMethod("candidate4_plus_validated_resnet18_video_v1"), "Adaptive Multimodal Fusion + ResNet18");
  assert.equal(formatFusionMethod("text_anchored_class_conditional_bounded_fusion_v1"), "Text-Anchored Bounded Fusion");
});

test("8. formatProbabilityPercent matches 1-decimal confidence badge formatting", () => {
  assert.equal(formatProbabilityPercent(0.3092), "30.9%");
  assert.equal(formatProbabilityPercent(0.2561), "25.6%");
});

test("9. measureWavRmsDbFromBuffer accurately flags silent vs normal audio levels", () => {
  const silent = new Float32Array(1600);
  const silentWav = encodeWavPcm16(silent, 16000);
  const silentReport = measureWavRmsDbFromBuffer(silentWav);
  assert.ok(silentReport);
  assert.equal(silentReport.isNearSilent, true);
  assert.equal(silentReport.isLowVolume, true);

  const quiet = new Float32Array(1600).fill(0.004);
  const quietReport = analyzeWaveformLevel(quiet);
  assert.equal(quietReport.isLowVolume, true);
  assert.equal(quietReport.isNearSilent, false);

  const normal = new Float32Array(1600).fill(0.1);
  const normalReport = measureWavRmsDbFromBuffer(encodeWavPcm16(normal, 16000));
  assert.ok(normalReport);
  assert.equal(normalReport.isLowVolume, false);
  assert.equal(normalReport.isNearSilent, false);
});

test("10. classifyInputError separates validation rejection from server/network errors", () => {
  const val = classifyInputError("Invalid audio: silent_audio (RMS energy (-70.0 dB) is below silence threshold (-60.0 dB))");
  assert.equal(val.isValidationError, true);
  assert.match(val.title, /Input rejected/i);
  assert.ok(val.guidance);

  const srv = classifyInputError("Unable to reach the Emotion Studio API. Check that the backend is running and retry.");
  assert.equal(srv.isValidationError, false);
  assert.equal(srv.guidance, null);
});
