export const CLASS_NAMES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"] as const;
export type Emotion = (typeof CLASS_NAMES)[number];
export type Prediction = { predicted_emotion: Emotion; probabilities: Record<Emotion, number>; confidence: number; inference_duration_ms: number; modality: string; modality_status: string; quality: Record<string, number> };
export type ModelStatus = {
  loaded: boolean;
  model_identifier: string | null;
  device: string | null;
  error: string | null;
  architecture?: string | null;
  checkpoint_sha256?: string | null;
  active_role?: string | null;
  preprocessing?: string | null;
  pooling_method?: string | null;
};
export type ModelsStatus = { status: string; models: Record<"text" | "audio" | "video", ModelStatus> };
export function resolveApiBaseUrl(
  rawUrl: string | undefined = typeof import.meta !== "undefined" ? import.meta.env?.VITE_API_BASE_URL : undefined,
): string {
  const value = rawUrl ?? "http://localhost:8000";
  return value.replace(/\/+$/, "");
}

export const API_BASE = resolveApiBaseUrl();

export const REQUEST_TIMEOUT_MS = 60_000;

export type RequestOptions = {
  signal?: AbortSignal;
  timeoutMs?: number;
  baseUrl?: string;
};

function makeAbortError(message = "Request was cancelled."): Error {
  if (typeof DOMException !== "undefined") {
    return new DOMException(message, "AbortError");
  }
  const err = new Error(message);
  err.name = "AbortError";
  return err;
}

export async function request<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = REQUEST_TIMEOUT_MS,
  baseUrl = API_BASE,
): Promise<T> {
  const controller = new AbortController();
  const callerSignal = init?.signal;
  let timedOut = false;

  if (callerSignal?.aborted) {
    throw makeAbortError();
  }

  const onCallerAbort = () => {
    controller.abort();
  };

  if (callerSignal) {
    callerSignal.addEventListener("abort", onCallerAbort, { once: true });
  }

  const timer = globalThis.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    const normalizedBase = baseUrl.replace(/\/+$/, "");
    const response = await fetch(`${normalizedBase}${path}`, { ...init, signal: controller.signal });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`);
    }
    return body as T;
  } catch (cause) {
    if (timedOut) {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s. Please try again.`);
    }
    if (callerSignal?.aborted || (cause instanceof Error && cause.name === "AbortError")) {
      throw cause instanceof Error && cause.name === "AbortError" ? cause : makeAbortError();
    }
    if (cause instanceof TypeError) {
      throw new Error("Unable to reach the Emotion Studio API. Check that the backend is running and retry.");
    }
    throw cause;
  } finally {
    globalThis.clearTimeout(timer);
    callerSignal?.removeEventListener("abort", onCallerAbort);
  }
}

export const getModelStatus = (options?: RequestOptions) =>
  request<ModelsStatus>(
    "/models/status",
    options?.signal ? { signal: options.signal } : undefined,
    options?.timeoutMs,
    options?.baseUrl,
  );

export const predictText = (text: string, options?: RequestOptions) =>
  request<Prediction>(
    "/predict/text",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      ...(options?.signal ? { signal: options.signal } : {}),
    },
    options?.timeoutMs,
    options?.baseUrl,
  );

export const predictFile = (modality: "audio" | "video", file: File, options?: RequestOptions) => {
  const form = new FormData();
  form.append("file", file);
  return request<Prediction>(
    `/predict/${modality}`,
    { method: "POST", body: form, ...(options?.signal ? { signal: options.signal } : {}) },
    options?.timeoutMs,
    options?.baseUrl,
  );
};

export type MultimodalPrediction = {
  predicted_emotion: Emotion;
  probabilities: Record<Emotion, number>;
  confidence: number;
  fusion_method: string;
  modalities_available: { text: boolean; audio: boolean; video: boolean };
  exclusion_reasons: { text?: string | null; audio: string | null; video: string | null };
  modality_weights: { text: number; audio: number; video: number };
  individual_predictions: {
    text: Prediction | null;
    audio: Prediction | null;
    video: Prediction | null;
  };
  inference_duration_ms: number;
  rule_applied?: string | null;
  quality?: Record<string, number | null>;
};

export const FUSION_METHOD_LABELS: Record<string, string> = {
  text_anchored_class_conditional_bounded_fusion_v1: "Text-Anchored Bounded Fusion",
  candidate4_plus_validated_resnet18_video_v1: "Adaptive Multimodal Fusion + ResNet18",
  single_modality_video_resnet18_v1: "Single-Modality ResNet18 Face",
  single_modality_audio_wav2vec2_v1: "Single-Modality Wav2Vec2 Speech",
  audio_primary_plus_validated_resnet18_video_v1: "Audio Primary + ResNet18 Video",
};

export function formatFusionMethod(method?: string | null): string {
  if (!method) return "Adaptive Multimodal Fusion";
  const mapped = FUSION_METHOD_LABELS[method];
  if (mapped) return mapped;
  return method
    .replace(/candidate\s*\d*_?/gi, "adaptive_")
    .replace(/candidate/gi, "adaptive")
    .replaceAll("_", " ")
    .replace(/\s+/g, " ")
    .trim();
}

export const predictMultimodal = (
  text: string,
  audioFile?: File | null,
  videoFile?: File | null,
  options?: RequestOptions,
): Promise<MultimodalPrediction> => {
  if (audioFile || videoFile) {
    const form = new FormData();
    if (text.trim()) {
      form.append("text", text);
    }
    if (audioFile) {
      form.append("file", audioFile);
    }
    if (videoFile) {
      form.append("video", videoFile);
    }
    return request<MultimodalPrediction>(
      "/predict/multimodal",
      { method: "POST", body: form, ...(options?.signal ? { signal: options.signal } : {}) },
      options?.timeoutMs,
      options?.baseUrl,
    );
  }
  return request<MultimodalPrediction>(
    "/predict/multimodal",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      ...(options?.signal ? { signal: options.signal } : {}),
    },
    options?.timeoutMs,
    options?.baseUrl,
  );
};

export type InputErrorClassification = {
  isValidationError: boolean;
  title: string;
  guidance: string | null;
};

export function classifyInputError(message: string): InputErrorClassification {
  const lower = message.toLowerCase();
  if (
    lower.includes("silent_audio") ||
    lower.includes("silence threshold") ||
    lower.includes("duration") ||
    lower.includes("invalid audio") ||
    lower.includes("unreadable") ||
    lower.includes("no detectable faces") ||
    lower.includes("upload rejected") ||
    lower.includes("unsupported") ||
    lower.includes("empty") ||
    lower.includes("no usable modality")
  ) {
    return {
      isValidationError: true,
      title: "Input rejected (invalid or silent signal)",
      guidance:
        "Empty, silent (< -60 dB), unreadable, or face-free media is rejected instead of returning an emotion score. Low-confidence speech or video still returns a normal prediction with a <45% notice.",
    };
  }
  return {
    isValidationError: false,
    title: "Request error",
    guidance: null,
  };
}
