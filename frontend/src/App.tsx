import { memo, useEffect, useMemo, useRef, useState } from "react";
import { createRequestScope, createRetryableExecution } from "./retry";
import {
  Activity,
  AudioLines,
  Check,
  ChevronDown,
  CircleAlert,
  Film,
  Gauge,
  Mic2,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Type,
  Upload,
  Video,
  Volume2,
  X,
} from "lucide-react";
import {
  classifyInputError,
  formatFusionMethod,
  getModelStatus,
  predictFile,
  predictMultimodal,
  predictText,
  type Emotion,
  type ModelsStatus,
  type MultimodalPrediction,
  type Prediction,
} from "./api";
import { LiveAudioCapture, LiveVideoCapture } from "./LiveCapture";
import { formatProbabilityPercent, validateUploadFileSize } from "./media";

type Tab = "multimodal" | "text" | "audio" | "video";
type InputMode = "upload" | "live";

const tabs: { id: Tab; label: string; icon: typeof Sparkles; hint: string; category: string }[] = [
  { id: "multimodal", label: "Multimodal Studio", icon: Sparkles, hint: "Adaptive fusion + ResNet18 facial branch", category: "Adaptive Fusion" },
  { id: "text", label: "Text", icon: Type, hint: "DistilBERT semantic emotion inference", category: "Words" },
  { id: "audio", label: "Speech", icon: AudioLines, hint: "Wav2Vec2 acoustic prosody inference", category: "Voice" },
  { id: "video", label: "Face", icon: Film, hint: "ResNet18 + MTCNN facial emotion inference", category: "Vision" },
];

function SourceSelect({
  ariaLabel,
  mode,
  onChange,
  uploadLabel,
  liveLabel,
  compact = false,
}: {
  ariaLabel: string;
  mode: InputMode;
  onChange: (next: InputMode) => void;
  uploadLabel: string;
  liveLabel: string;
  compact?: boolean;
}) {
  return (
    <div className="relative inline-flex items-center">
      <select
        aria-label={ariaLabel}
        value={mode}
        onChange={(e) => onChange(e.target.value as InputMode)}
        className={`cursor-pointer appearance-none rounded-xl border border-white/10 bg-ink/80 text-white outline-none transition hover:border-white/20 focus:border-cyan/60 ${
          compact ? "py-1.5 pl-3 pr-8 text-xs font-medium" : "w-full py-2.5 pl-3.5 pr-9 text-xs font-semibold sm:w-64"
        }`}
      >
        <option value="upload" className="bg-[#0b1526] text-white">
          {uploadLabel}
        </option>
        <option value="live" className="bg-[#0b1526] text-white">
          {liveLabel}
        </option>
      </select>
      <ChevronDown
        size={14}
        className="pointer-events-none absolute right-2.5 text-slate-400"
      />
    </div>
  );
}

const barColors: Record<Emotion, string> = {
  anger: "bg-coral",
  disgust: "bg-lavender",
  fear: "bg-amber-300",
  joy: "bg-cyan",
  neutral: "bg-slate-400",
  sadness: "bg-blue-400",
  surprise: "bg-pink-400",
};

function formatRule(rule?: string | null): { label: string; tone: "cyan" | "coral" | "lavender" | "slate" } {
  if (!rule) return { label: "Standard Prediction", tone: "slate" };
  if (rule === "video_only_resnet18") return { label: "ResNet18 Facial Single-Modality", tone: "cyan" };
  if (rule === "audio_only_fallback") return { label: "Wav2Vec2 Speech Single-Modality", tone: "coral" };
  if (rule.startsWith("audio_video_")) return { label: "Audio Primary + Calibrated ResNet18 Video", tone: "cyan" };
  const videoCal = rule.includes("+validated_video_calibrated") ? " + Calibrated Video (0.08)" : rule.includes("+video_advisory_only") ? " + Video Advisory" : "";
  const baseRule = rule.split("+")[0];
  if (baseRule.startsWith("protected_class_lockout")) {
    const cls = baseRule.replace("protected_class_lockout_", "");
    return { label: `Protected Class Lockout (${cls})${videoCal}`, tone: "lavender" };
  }
  if (baseRule.startsWith("acoustic_synergy")) {
    const cls = baseRule.replace("acoustic_synergy_", "");
    return { label: `Acoustic Synergy Active (${cls})${videoCal}`, tone: "coral" };
  }
  if (baseRule === "text_anchor_default") {
    return { label: `Text-Anchor Default${videoCal}`, tone: "cyan" };
  }
  return { label: rule, tone: "slate" };
}

function UncertaintyNotice({
  confidence,
  signalNote,
}: {
  confidence: number;
  signalNote?: string | null;
}) {
  const lowConfidence = confidence < 0.45;
  return (
    <div
      className={`rounded-xl border px-3.5 py-2.5 text-xs leading-relaxed ${
        lowConfidence
          ? "border-amber-400/25 bg-amber-400/10 text-amber-200"
          : "border-white/10 bg-white/5 text-slate-300"
      }`}
    >
      <p className="font-semibold">
        {lowConfidence ? "Low-confidence estimate (<45%)" : "Probabilistic AI estimate"}
      </p>
      {signalNote && <p className="mt-1 text-[11px] font-medium text-amber-200">{signalNote}</p>}
      <p className="mt-0.5 text-[11px] opacity-90">
        Emotion scores reflect statistical patterns from training data and may misclassify subtle,
        sarcastic, or accented expression. Intended for informational exploration only — not a medical
        or psychological diagnosis.
      </p>
    </div>
  );
}

function getSignalNote(quality?: Record<string, number | null> | null): string | null {
  if (!quality) return null;
  const lowVol = quality.low_volume_warning ?? quality.audio_low_volume_warning ?? 0;
  const cov = quality.face_detection_coverage ?? quality.video_face_detection_coverage ?? null;
  const notes: string[] = [];
  if (Number(lowVol) >= 0.5) {
    notes.push("Low speech volume detected (< -40 dB) — acoustic cues may be understated.");
  }
  if (cov !== null && Number(cov) < 0.5) {
    notes.push("Partial face visibility across sampled video frames reduced visual stability.");
  }
  return notes.length > 0 ? notes.join(" ") : null;
}

function isSignalRejectionReason(reason?: string | null): boolean {
  if (!reason) return false;
  const lower = reason.toLowerCase();
  return !lower.startsWith("no audio file provided") && !lower.startsWith("no video file provided") && !lower.startsWith("no text input provided");
}

function App() {
  const [tab, setTab] = useState<Tab>("multimodal");
  const [status, setStatus] = useState<ModelsStatus | null>(null);
  const [statusError, setStatusError] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [multimodalPrediction, setMultimodalPrediction] = useState<MultimodalPrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [retryAction, setRetryAction] = useState<(() => Promise<boolean>) | null>(null);
  const [requestScope] = useState(() => createRequestScope());
  const inFlightRef = useRef(false);

  const refreshStatus = async () => {
    try {
      setStatusError("");
      setStatus(await getModelStatus());
    } catch (cause) {
      setStatusError(cause instanceof Error ? cause.message : "Backend unavailable");
    }
  };

  useEffect(() => {
    void refreshStatus();
    return () => requestScope.abortActive();
  }, [requestScope]);

  const runGuarded = async (fn: () => Promise<boolean>) => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      await fn();
    } finally {
      inFlightRef.current = false;
    }
  };

  const runAnalysis = async <T,>(
    operation: (signal: AbortSignal) => Promise<T>,
    onSuccess: (value: T) => void,
    fallbackError: string,
  ) => {
    if (inFlightRef.current) return;
    const execute = createRetryableExecution(
      operation,
      {
        onStart: () => {
          setLoading(true);
          setError("");
          setPrediction(null);
          setMultimodalPrediction(null);
        },
        onSuccess,
        onError: setError,
        onFinally: () => setLoading(false),
        fallbackError,
      },
      requestScope,
    );
    setRetryAction(() => execute);
    await runGuarded(execute);
  };

  const runSingle = (operation: (signal: AbortSignal) => Promise<Prediction>) =>
    runAnalysis(operation, setPrediction, "The prediction failed.");

  const runMultimodal = (text: string, audioFile: File | null, videoFile: File | null) =>
    runAnalysis(
      (signal) => predictMultimodal(text, audioFile, videoFile, { signal }),
      setMultimodalPrediction,
      "Multimodal prediction failed.",
    );

  const active = tabs.find((item) => item.id === tab)!;

  return (
    <div className="grid-lines min-h-screen">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-cyan focus:px-4 focus:py-2 focus:text-sm focus:font-bold focus:text-ink"
      >
        Skip to main content
      </a>
      <header className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6 lg:px-10">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan text-ink">
            <Sparkles size={21} />
          </div>
          <div>
            <p className="font-display text-sm font-bold tracking-wide text-white">EMOTION / STUDIO</p>
            <p className="text-[10px] uppercase tracking-[.2em] text-slate-400">Multimodal intelligence</p>
          </div>
        </div>
        <div
          aria-live="polite"
          className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-300"
        >
          <span className={`h-2 w-2 rounded-full ${status?.status === "ready" ? "bg-cyan shadow-[0_0_10px_#4de1d2]" : "bg-coral"}`} />
          {status?.status === "ready" ? "API connected" : "Connecting to API"}
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-7xl px-6 pb-16 lg:px-10">
        <section className="grid gap-10 pb-12 pt-10 lg:grid-cols-[1.15fr_.85fr] lg:items-end">
          <div>
            <p className="eyebrow mb-5">Reliability-aware adaptive intelligence</p>
            <h1 className="max-w-3xl font-display text-5xl font-extrabold leading-[1.05] tracking-[-.04em] text-white sm:text-6xl">
              Read the feeling<br />
              <span className="text-cyan">behind the signal.</span>
            </h1>
            <p className="muted mt-6 max-w-xl text-base">
              Explore reliability-aware adaptive multimodal fusion and individual single-modality predictors.
              Text remains authoritative while speech prosody synergistically elevates anger and surprise.
            </p>
          </div>
          <div role="tablist" aria-label="Analysis modality" className="panel grid grid-cols-2 gap-3 p-4 sm:grid-cols-4">
            {tabs.map(({ id, label, icon: Icon, category }) => (
              <button
                key={id}
                role="tab"
                aria-selected={tab === id}
                onClick={() => {
                  requestScope.abortActive();
                  setTab(id);
                  setPrediction(null);
                  setMultimodalPrediction(null);
                  setError("");
                  setRetryAction(null);
                }}
                className={`rounded-xl p-4 text-left transition ${
                  tab === id ? "bg-white/10 text-white shadow-inner" : "text-slate-300 hover:bg-white/5 hover:text-white"
                }`}
              >
                <Icon size={19} className={tab === id ? "text-cyan" : ""} />
                <p className="mt-8 text-sm font-semibold">{label}</p>
                <p className="mt-1 text-[10px] uppercase tracking-wider text-slate-400">{category}</p>
              </button>
            ))}
          </div>
        </section>

        <div className="grid gap-6 lg:grid-cols-[1.1fr_.9fr]">
          <section className="panel min-h-[460px] p-6 sm:p-8">
            <div className="mb-8 flex items-start justify-between">
              <div>
                <p className="eyebrow">{active.label}</p>
                <h2 className="mt-2 font-display text-2xl font-bold text-white">{active.hint}</h2>
              </div>
              <div className="rounded-full bg-cyan/10 p-3 text-cyan">
                {tab === "multimodal" ? <Sparkles size={20} /> : tab === "text" ? <Type size={20} /> : tab === "audio" ? <Mic2 size={20} /> : <Video size={20} />}
              </div>
            </div>

            {tab === "multimodal" ? (
              <MultimodalInput onSubmit={runMultimodal} loading={loading} />
            ) : tab === "text" ? (
              <TextInput
                onSubmit={(value) => {
                  if (!value.trim()) return setError("Enter some text before analyzing.");
                  void runSingle((signal) => predictText(value, { signal }));
                }}
                loading={loading}
              />
            ) : (
              <FileInput key={tab} kind={tab} onSubmit={(file) => void runSingle((signal) => predictFile(tab, file, { signal }))} loading={loading} />
            )}

            {error && (() => {
              const errInfo = classifyInputError(error);
              return (
                <div
                  role="alert"
                  className="mt-5 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-coral/25 bg-coral/10 p-4 text-sm text-coral"
                >
                  <div className="flex items-start gap-3">
                    <CircleAlert size={18} className="mt-0.5 shrink-0" />
                    <div>
                      <p className="font-semibold text-white">{errInfo.title}</p>
                      <p className="mt-0.5">{error}</p>
                      {errInfo.guidance && (
                        <p className="mt-1 text-xs text-slate-300">{errInfo.guidance}</p>
                      )}
                    </div>
                  </div>
                  {retryAction && (
                    <button
                      type="button"
                      disabled={loading}
                      onClick={() => void runGuarded(retryAction)}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-coral/40 bg-coral/15 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-coral/25 disabled:opacity-40"
                    >
                      <RefreshCw size={12} />
                      Retry
                    </button>
                  )}
                </div>
              );
            })()}
          </section>

          {tab === "multimodal" ? (
            <MultimodalResultCard prediction={multimodalPrediction} loading={loading} />
          ) : (
            <ResultCard prediction={prediction} loading={loading} />
          )}
        </div>

        <section className="mt-6 grid gap-6 lg:grid-cols-[1fr_1.2fr]">
          <StatusCard status={status} error={statusError} onRefresh={() => void refreshStatus()} />
          <QualityCard prediction={prediction} multimodal={multimodalPrediction} />
        </section>
      </main>

      <footer className="mx-auto flex max-w-7xl flex-col gap-2 px-6 pb-8 text-xs text-slate-400 sm:flex-row sm:justify-between lg:px-10">
        <span>Phase 19 · Adaptive Multimodal Studio · Non-diagnostic probabilistic estimates</span>
        <span>Seven-class MELD emotion schema (DistilBERT + Wav2Vec2 + ResNet18)</span>
      </footer>
    </div>
  );
}

function MultimodalInput({
  onSubmit,
  loading,
}: {
  onSubmit: (text: string, audioFile: File | null, videoFile: File | null) => void;
  loading: boolean;
}) {
  const [text, setText] = useState("");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [speechMode, setSpeechMode] = useState<InputMode>("upload");
  const [videoMode, setVideoMode] = useState<InputMode>("upload");
  const [capturingAudio, setCapturingAudio] = useState(false);
  const [capturingVideo, setCapturingVideo] = useState(false);
  const [uploadError, setUploadError] = useState("");

  const capturing = capturingAudio || capturingVideo;
  const canSubmit = Boolean(text.trim() || audioFile || videoFile) && !capturing && !loading;

  const handleSelectAudio = (picked: File | null) => {
    if (!picked) {
      setAudioFile(null);
      return;
    }
    const sizeErr = validateUploadFileSize(picked);
    if (sizeErr) {
      setUploadError(sizeErr);
      setAudioFile(null);
      return;
    }
    setUploadError("");
    setAudioFile(picked);
  };

  const handleSelectVideo = (picked: File | null) => {
    if (!picked) {
      setVideoFile(null);
      return;
    }
    const sizeErr = validateUploadFileSize(picked);
    if (sizeErr) {
      setUploadError(sizeErr);
      setVideoFile(null);
      return;
    }
    setUploadError("");
    setVideoFile(picked);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    onSubmit(text, audioFile, videoFile);
  };

  const handleSpeechModeChange = (next: InputMode) => {
    setSpeechMode(next);
    setAudioFile(null);
    setCapturingAudio(false);
  };

  const handleVideoModeChange = (next: InputMode) => {
    setVideoMode(next);
    setVideoFile(null);
    setCapturingVideo(false);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="mb-2 block text-xs font-semibold uppercase tracking-wider text-slate-400">
          Utterance / Dialogue text{" "}
          <span className="text-slate-500">(Required unless speech or video attached)</span>
        </label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          maxLength={20000}
          placeholder="Enter utterance or dialogue turn (e.g. 'Get out of here right now!')..."
          className="h-28 w-full resize-none rounded-2xl border border-white/10 bg-ink/60 p-4 text-sm text-white outline-none transition placeholder:text-slate-600 focus:border-cyan/60"
        />
        <div className="mt-1 flex justify-end">
          <span className="text-[10px] text-slate-500">{text.length.toLocaleString()} / 20,000 characters</span>
        </div>
      </div>

      <div>
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
            Acoustic speech recording <span className="text-slate-500">(Optional WAV)</span>
          </label>
          <div className="flex items-center gap-3">
            {speechMode === "upload" && audioFile && (
              <button
                type="button"
                onClick={() => setAudioFile(null)}
                className="flex items-center gap-1 text-[11px] text-coral hover:underline"
              >
                <X size={13} /> Remove audio
              </button>
            )}
            <SourceSelect
              ariaLabel="Multimodal speech source"
              mode={speechMode}
              onChange={handleSpeechModeChange}
              uploadLabel="Upload WAV file"
              liveLabel="Live microphone"
              compact
            />
          </div>
        </div>

        {speechMode === "upload" ? (
          <label className="flex h-24 cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-white/15 bg-ink/40 p-3 text-center transition hover:border-cyan/60 hover:bg-cyan/5">
            <Volume2 size={20} className="mb-1 text-cyan" />
            <span className="text-xs font-medium text-white">
              {audioFile ? audioFile.name : "Attach a WAV audio file (optional for acoustic synergy)"}
            </span>
            <span className="text-[10px] text-slate-500">16 kHz mono WAV · max 50 MB</span>
            <input
              type="file"
              accept=".wav,audio/wav"
              className="hidden"
              onChange={(e) => {
                handleSelectAudio(e.target.files?.[0] ?? null);
                e.currentTarget.value = "";
              }}
            />
          </label>
        ) : (
          <LiveAudioCapture
            compact
            file={audioFile}
            onChange={setAudioFile}
            onCapturingChange={setCapturingAudio}
            disabled={loading}
          />
        )}
      </div>

      <div>
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400">
            Facial video stream <span className="text-slate-500">(Optional MP4 / WebM)</span>
          </label>
          <div className="flex items-center gap-3">
            {videoMode === "upload" && videoFile && (
              <button
                type="button"
                onClick={() => setVideoFile(null)}
                className="flex items-center gap-1 text-[11px] text-coral hover:underline"
              >
                <X size={13} /> Remove video
              </button>
            )}
            <SourceSelect
              ariaLabel="Multimodal video source"
              mode={videoMode}
              onChange={handleVideoModeChange}
              uploadLabel="Upload MP4 / WebM"
              liveLabel="Live camera"
              compact
            />
          </div>
        </div>

        {videoMode === "upload" ? (
          <label className="flex h-24 cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-white/15 bg-ink/40 p-3 text-center transition hover:border-cyan/60 hover:bg-cyan/5">
            <Film size={20} className="mb-1 text-cyan" />
            <span className="text-xs font-medium text-white">
              {videoFile ? videoFile.name : "Attach an MP4 / WebM video file (optional ResNet18 facial cue)"}
            </span>
            <span className="text-[10px] text-slate-500">MP4 / WebM · max 50 MB · MTCNN ResNet18</span>
            <input
              type="file"
              accept=".mp4,.webm,video/mp4,video/webm"
              className="hidden"
              onChange={(e) => {
                handleSelectVideo(e.target.files?.[0] ?? null);
                e.currentTarget.value = "";
              }}
            />
          </label>
        ) : (
          <LiveVideoCapture
            compact
            file={videoFile}
            onChange={setVideoFile}
            onCapturingChange={setCapturingVideo}
            disabled={loading}
          />
        )}
      </div>

      {uploadError && (
        <div className="flex items-start gap-2 rounded-xl border border-coral/25 bg-coral/10 p-3 text-xs text-coral">
          <CircleAlert size={14} className="mt-0.5 shrink-0" />
          <span>{uploadError}</span>
        </div>
      )}

      <div className="pt-2">
        <button
          type="submit"
          disabled={!canSubmit}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-cyan px-5 py-3 text-sm font-bold text-ink transition hover:bg-cyan/80 disabled:opacity-50"
        >
          {loading ? <RefreshCw className="animate-spin" size={16} /> : <Sparkles size={16} />}
          {videoFile
            ? "Run Multimodal Fusion (with ResNet18 Face)"
            : "Run Adaptive Multimodal Prediction"}
        </button>
      </div>
    </form>
  );
}

function TextInput({ onSubmit, loading }: { onSubmit: (value: string) => void; loading: boolean }) {
  const [value, setValue] = useState("");
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit(value);
      }}
    >
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        maxLength={20000}
        placeholder="Type an utterance, message, or dialogue turn..."
        className="h-52 w-full resize-none rounded-2xl border border-white/10 bg-ink/60 p-5 text-base text-white outline-none transition placeholder:text-slate-600 focus:border-cyan/60"
      />
      <div className="mt-4 flex items-center justify-between">
        <span className="text-xs text-slate-600">{value.length.toLocaleString()} / 20,000 characters</span>
        <button
          disabled={loading}
          className="flex items-center gap-2 rounded-xl bg-cyan px-5 py-3 text-sm font-bold text-ink transition hover:bg-cyan/80 disabled:opacity-50"
        >
          {loading ? <RefreshCw className="animate-spin" size={16} /> : <Activity size={16} />} Analyze text
        </button>
      </div>
    </form>
  );
}

function FileInput({
  kind,
  onSubmit,
  loading,
}: {
  kind: "audio" | "video";
  onSubmit: (file: File) => void;
  loading: boolean;
}) {
  const [mode, setMode] = useState<InputMode>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [uploadError, setUploadError] = useState("");

  const handleModeChange = (next: InputMode) => {
    setMode(next);
    setFile(null);
    setUploadError("");
  };

  const handleSelectFile = (picked: File | null) => {
    if (!picked) {
      setFile(null);
      return;
    }
    const sizeErr = validateUploadFileSize(picked);
    if (sizeErr) {
      setUploadError(sizeErr);
      setFile(null);
      return;
    }
    setUploadError("");
    setFile(picked);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          {kind === "audio" ? "Speech input source" : "Video input source"}
        </label>
        <SourceSelect
          ariaLabel={kind === "audio" ? "Speech input source" : "Video input source"}
          mode={mode}
          onChange={handleModeChange}
          uploadLabel={kind === "audio" ? "Upload WAV file" : "Upload MP4 / WebM video"}
          liveLabel={kind === "audio" ? "Live microphone (Capture & Listen)" : "Live camera (Capture & Listen)"}
        />
      </div>

      {mode === "upload" ? (
        <label className="flex h-52 cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-white/15 bg-ink/50 text-center transition hover:border-cyan/60 hover:bg-cyan/5">
          <Upload size={27} className="mb-4 text-cyan" />
          <span className="font-semibold text-white">
            {file ? file.name : `Choose a ${kind === "audio" ? "WAV audio" : "MP4 or WebM video"} file`}
          </span>
          <span className="mt-2 text-xs text-slate-500">Validated by the backend · max 50 MB</span>
          <input
            type="file"
            accept={kind === "audio" ? ".wav,audio/wav" : ".mp4,.webm,video/mp4,video/webm"}
            className="hidden"
            onChange={(event) => {
              handleSelectFile(event.target.files?.[0] ?? null);
              event.currentTarget.value = "";
            }}
          />
        </label>
      ) : kind === "audio" ? (
        <LiveAudioCapture
          file={file}
          onChange={(nextFile) => {
            setUploadError("");
            setFile(nextFile);
          }}
          onCapturingChange={setCapturing}
          disabled={loading}
        />
      ) : (
        <LiveVideoCapture
          file={file}
          onChange={(nextFile) => {
            setUploadError("");
            setFile(nextFile);
          }}
          onCapturingChange={setCapturing}
          disabled={loading}
        />
      )}

      {uploadError && (
        <div className="flex items-start gap-2 rounded-xl border border-coral/25 bg-coral/10 p-3 text-xs text-coral">
          <CircleAlert size={14} className="mt-0.5 shrink-0" />
          <span>{uploadError}</span>
        </div>
      )}

      <div className="flex justify-end">
        <button
          disabled={!file || capturing || loading}
          onClick={() => file && onSubmit(file)}
          className="flex items-center gap-2 rounded-xl bg-cyan px-5 py-3 text-sm font-bold text-ink transition hover:bg-cyan/80 disabled:opacity-40"
        >
          {loading ? <RefreshCw className="animate-spin" size={16} /> : <Gauge size={16} />} Analyze {kind}
        </button>
      </div>
    </div>
  );
}

const MultimodalResultCard = memo(function MultimodalResultCard({
  prediction,
  loading,
}: {
  prediction: MultimodalPrediction | null;
  loading: boolean;
}) {
  const videoActive = Boolean(
    prediction?.modalities_available.video && (prediction?.modality_weights.video ?? 0) > 0,
  );
  const videoAdvisory = Boolean(
    prediction?.modalities_available.video && (prediction?.modality_weights.video ?? 0) === 0,
  );

  return (
    <section aria-live="polite" className="panel relative overflow-hidden p-6 sm:p-8">
      <div className="absolute -right-20 -top-20 h-48 w-48 rounded-full bg-cyan/10 blur-3xl" />
      <div className="flex items-center justify-between gap-2">
        <p className="eyebrow">Multimodal Fused Result</p>
        {prediction && (
          <span
            className="max-w-[70%] truncate rounded-full bg-cyan/10 px-3 py-1 text-[11px] font-bold text-cyan"
            title={formatFusionMethod(prediction.fusion_method)}
          >
            {formatFusionMethod(prediction.fusion_method)}
          </span>
        )}
      </div>

      {loading ? (
        <div className="flex h-[360px] flex-col items-center justify-center text-slate-400">
          <RefreshCw className="mb-4 animate-spin text-cyan" size={28} />
          <p className="text-sm">Running adaptive multimodal fusion...</p>
        </div>
      ) : !prediction ? (
        <div className="flex h-[360px] flex-col items-center justify-center text-center">
          <div className="mb-4 rounded-2xl bg-white/5 p-4 text-slate-400">
            <Sparkles size={27} />
          </div>
          <p className="text-sm text-slate-300">Your multimodal prediction will appear here.</p>
          <p className="mt-2 text-xs text-slate-400">
            Enter text, optional WAV audio, and/or MP4/WebM face video to analyze
          </p>
        </div>
      ) : (
        <div className="relative mt-6 space-y-6">
          <div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[.2em] text-slate-400">Predicted Emotion</p>
                <h2 className="mt-1 font-display text-4xl font-extrabold capitalize text-white sm:text-5xl">
                  {prediction.predicted_emotion}
                </h2>
              </div>
              <div className="text-right">
                <span className="inline-block rounded-full bg-lavender/15 px-3 py-1 text-xs font-bold text-lavender">
                  {(prediction.confidence * 100).toFixed(1)}% confidence
                </span>
                <p className="mt-1 text-xs text-slate-400">{prediction.inference_duration_ms.toFixed(1)} ms total</p>
              </div>
            </div>

            {/* Fusion Rule Banner */}
            {prediction.rule_applied && (
              <div className="mt-4 flex flex-wrap items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs">
                <ShieldCheck size={15} className="shrink-0 text-cyan" />
                <span className="font-semibold text-slate-300">Rule Applied:</span>
                <span className="text-cyan">{formatRule(prediction.rule_applied).label}</span>
              </div>
            )}
          </div>

          {/* Probability Bars */}
          <ProbabilityBars probabilities={prediction.probabilities} />

          {/* Modality Status & Weights */}
          <div className="grid grid-cols-1 gap-2 border-t border-white/10 pt-4 text-xs sm:grid-cols-3">
            <div className="rounded-lg bg-ink/60 p-2.5">
              <span className="text-[10px] uppercase text-slate-500">Text Modality</span>
              <p className={`font-semibold ${prediction.modalities_available.text ? "text-white" : "text-slate-400"}`}>
                {prediction.modalities_available.text
                  ? `Active (w: ${prediction.modality_weights.text.toFixed(2)})`
                  : "Excluded"}
              </p>
              {prediction.exclusion_reasons.text && (
                <p className="truncate text-[10px] text-slate-400" title={prediction.exclusion_reasons.text}>
                  {prediction.exclusion_reasons.text}
                </p>
              )}
            </div>
            <div className="rounded-lg bg-ink/60 p-2.5">
              <span className="text-[10px] uppercase text-slate-500">Audio Modality</span>
              {(() => {
                const audioRejected =
                  !prediction.modalities_available.audio &&
                  isSignalRejectionReason(prediction.exclusion_reasons.audio);
                return (
                  <>
                    <p
                      className={`font-semibold ${
                        prediction.modalities_available.audio
                          ? "text-white"
                          : audioRejected
                          ? "text-coral"
                          : "text-slate-400"
                      }`}
                    >
                      {prediction.modalities_available.audio
                        ? `Active (w: ${prediction.modality_weights.audio.toFixed(2)})`
                        : audioRejected
                        ? "Excluded (invalid input)"
                        : "Excluded"}
                    </p>
                    {prediction.exclusion_reasons.audio && (
                      <p
                        className={`truncate text-[10px] ${audioRejected ? "text-coral/90" : "text-slate-400"}`}
                        title={prediction.exclusion_reasons.audio}
                      >
                        {prediction.exclusion_reasons.audio}
                      </p>
                    )}
                  </>
                );
              })()}
            </div>
            <div className="rounded-lg bg-ink/60 p-2.5">
              <span className="text-[10px] uppercase text-slate-500">Video Modality</span>
              {(() => {
                const videoRejected =
                  !prediction.modalities_available.video &&
                  isSignalRejectionReason(prediction.exclusion_reasons.video);
                return (
                  <>
                    <p
                      className={`font-semibold ${
                        videoActive
                          ? "text-white"
                          : videoAdvisory
                          ? "text-amber-300"
                          : videoRejected
                          ? "text-coral"
                          : "text-slate-400"
                      }`}
                    >
                      {videoActive
                        ? `Active (w: ${prediction.modality_weights.video.toFixed(2)})`
                        : videoAdvisory
                        ? "Advisory (w: 0.00)"
                        : videoRejected
                        ? "Excluded (invalid input)"
                        : "Excluded"}
                    </p>
                    {prediction.exclusion_reasons.video && (
                      <p
                        className={`truncate text-[10px] ${videoRejected ? "text-coral/90" : "text-slate-400"}`}
                        title={prediction.exclusion_reasons.video}
                      >
                        {prediction.exclusion_reasons.video}
                      </p>
                    )}
                  </>
                );
              })()}
            </div>
          </div>

          {/* Individual Predictions Breakdown */}
          <div className="border-t border-white/10 pt-4">
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
              Individual Modality Breakdown
            </p>
            <div className="grid grid-cols-1 gap-3 text-xs sm:grid-cols-3">
              <div className="rounded-xl border border-white/5 bg-ink/40 p-3">
                <span className="text-[10px] uppercase tracking-wider text-cyan">DistilBERT Text</span>
                <p className="mt-1 font-bold capitalize text-white">
                  {prediction.individual_predictions.text?.predicted_emotion ?? "Not active"}
                </p>
                <p className="text-[10px] text-slate-400">
                  {prediction.individual_predictions.text
                    ? `${(prediction.individual_predictions.text.confidence * 100).toFixed(1)}% conf`
                    : prediction.exclusion_reasons.text ?? "Not provided"}
                </p>
              </div>

              <div className="rounded-xl border border-white/5 bg-ink/40 p-3">
                <span className="text-[10px] uppercase tracking-wider text-coral">Wav2Vec2 Speech</span>
                <p className="mt-1 font-bold capitalize text-white">
                  {prediction.individual_predictions.audio?.predicted_emotion ?? "Not active"}
                </p>
                <p className="text-[10px] text-slate-400">
                  {prediction.individual_predictions.audio
                    ? `${(prediction.individual_predictions.audio.confidence * 100).toFixed(1)}% conf`
                    : prediction.exclusion_reasons.audio ?? "Not provided"}
                </p>
              </div>

              <div className="rounded-xl border border-white/5 bg-ink/40 p-3">
                <span className="text-[10px] uppercase tracking-wider text-lavender">ResNet18 Face</span>
                <p className="mt-1 font-bold capitalize text-white">
                  {prediction.individual_predictions.video?.predicted_emotion ?? "Not active"}
                </p>
                <p className="text-[10px] text-slate-400">
                  {prediction.individual_predictions.video
                    ? `${(prediction.individual_predictions.video.confidence * 100).toFixed(1)}% conf`
                    : prediction.exclusion_reasons.video ?? "Not provided"}
                </p>
              </div>
            </div>
          </div>

          <UncertaintyNotice confidence={prediction.confidence} signalNote={getSignalNote(prediction.quality)} />
        </div>
      )}
    </section>
  );
});

const ResultCard = memo(function ResultCard({ prediction, loading }: { prediction: Prediction | null; loading: boolean }) {
  return (
    <section aria-live="polite" className="panel relative overflow-hidden p-6 sm:p-8">
      <div className="absolute -right-20 -top-20 h-48 w-48 rounded-full bg-lavender/10 blur-3xl" />
      <p className="eyebrow">Live result</p>
      {loading ? (
        <div className="flex h-[330px] flex-col items-center justify-center text-slate-400">
          <RefreshCw className="mb-4 animate-spin text-cyan" size={28} />
          <p className="text-sm">Running the model...</p>
        </div>
      ) : !prediction ? (
        <div className="flex h-[330px] flex-col items-center justify-center text-center">
          <div className="mb-4 rounded-2xl bg-white/5 p-4 text-slate-400">
            <Sparkles size={27} />
          </div>
          <p className="text-sm text-slate-300">Your prediction will appear here.</p>
          <p className="mt-2 text-xs text-slate-400">Select a modality to begin</p>
        </div>
      ) : (
        <div className="relative mt-8 space-y-6">
          <div>
            <p className="text-xs uppercase tracking-[.2em] text-slate-400">Predicted emotion</p>
            <h2 className="mt-2 font-display text-5xl font-extrabold capitalize text-white">
              {prediction.predicted_emotion}
            </h2>
            <div className="mt-5 flex items-center gap-3">
              <span className="rounded-full bg-lavender/15 px-3 py-1 text-xs font-bold text-lavender">
                {(prediction.confidence * 100).toFixed(1)}% confidence
              </span>
              <span className="text-xs text-slate-400">{prediction.inference_duration_ms.toFixed(1)} ms</span>
            </div>
          </div>
          <ProbabilityBars probabilities={prediction.probabilities} />
          <UncertaintyNotice confidence={prediction.confidence} signalNote={getSignalNote(prediction.quality)} />
        </div>
      )}
    </section>
  );
});

const ProbabilityBars = memo(function ProbabilityBars({ probabilities }: { probabilities: Record<Emotion, number> }) {
  return (
    <div className="mt-6 space-y-2.5">
      {Object.entries(probabilities).map(([label, value]) => (
        <div key={label} className="flex items-center gap-3 text-xs">
          <span className="w-16 capitalize text-slate-400">{label}</span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/10">
            <div className={`h-full rounded-full ${barColors[label as Emotion]}`} style={{ width: `${value * 100}%` }} />
          </div>
          <span className="w-12 text-right font-semibold text-slate-300">{formatProbabilityPercent(value)}</span>
        </div>
      ))}
    </div>
  );
});

const StatusCard = memo(function StatusCard({
  status,
  error,
  onRefresh,
}: {
  status: ModelsStatus | null;
  error: string;
  onRefresh: () => void;
}) {
  return (
    <section className="panel p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="eyebrow">Model registry</p>
          <h2 className="mt-2 font-display text-xl font-bold text-white">Live backend status</h2>
        </div>
        <button onClick={onRefresh} aria-label="Refresh model status" className="rounded-lg p-2 text-slate-500 hover:bg-white/5 hover:text-cyan">
          <RefreshCw size={17} />
        </button>
      </div>
      {error ? (
        <p className="mt-7 text-sm text-coral">{error}</p>
      ) : (
        <div className="mt-6 space-y-3">
          {(["text", "audio", "video"] as const).map((name) => {
            const model = status?.models[name];
            return (
              <div key={name} className="flex items-center justify-between rounded-xl bg-ink/60 px-4 py-3">
                <div className="flex items-center gap-3">
                  <span className={`flex h-7 w-7 items-center justify-center rounded-lg ${model?.loaded ? "bg-cyan/10 text-cyan" : "bg-white/5 text-slate-600"}`}>
                    {model?.loaded ? <Check size={15} /> : <CircleAlert size={15} />}
                  </span>
                  <span className="text-sm font-medium capitalize text-slate-300">{name}</span>
                </div>
                <span className="text-xs text-slate-500">{model?.loaded ? model.device ?? "ready" : model?.error ?? "offline"}</span>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
});

const QualityCard = memo(function QualityCard({
  prediction,
  multimodal,
}: {
  prediction: Prediction | null;
  multimodal: MultimodalPrediction | null;
}) {
  const quality = useMemo(() => {
    if (multimodal?.quality) return multimodal.quality;
    return prediction?.quality ?? {};
  }, [prediction, multimodal]);

  return (
    <section className="panel p-6">
      <p className="eyebrow">Signal quality</p>
      <h2 className="mt-2 font-display text-xl font-bold text-white">Reliability context</h2>
      {!prediction && !multimodal ? (
        <p className="muted mt-6">Quality and acoustic reliability details become available after prediction.</p>
      ) : (
        <div className="mt-6 grid grid-cols-2 gap-3">
          {Object.entries(quality).length ? (
            Object.entries(quality).map(([key, value]) => (
              <div key={key} className="rounded-xl bg-ink/60 p-4">
                <p className="text-[10px] uppercase tracking-wider text-slate-500">{key.replaceAll("_", " ")}</p>
                <p className="mt-2 text-lg font-bold text-white">
                  {value === null || value === undefined
                    ? "N/A"
                    : key.includes("coverage")
                    ? `${(value * 100).toFixed(1)}%`
                    : key.endsWith("_rms_db")
                    ? `${Number(value).toFixed(1)} dB`
                    : key.endsWith("_s")
                    ? `${Number(value).toFixed(2)} s`
                    : key.includes("low_volume_warning")
                    ? Number(value) >= 0.5
                      ? "Low (< -40 dB)"
                      : "Normal"
                    : Number(value).toFixed(3)}
                </p>
              </div>
            ))
          ) : (
            <p className="col-span-2 text-sm text-slate-500">No modality-specific quality fields were returned.</p>
          )}
        </div>
      )}
    </section>
  );
});

export default App;
