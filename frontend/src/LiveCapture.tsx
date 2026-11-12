import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, ChevronDown, CircleAlert, Mic2, RefreshCw, RotateCcw, Square, Trash2, Video } from "lucide-react";
import {
  blobToWavFile,
  formatDuration,
  listMediaInputDevices,
  measureWavRmsDbFromBuffer,
  pickSupportedAudioMimeType,
  pickSupportedVideoMimeType,
  validateUploadFileSize,
  type AudioLevelReport,
  type MediaDeviceOption,
} from "./media";

const MAX_AUDIO_SECONDS = 45;
const MAX_VIDEO_SECONDS = 25;

function MediaConsentNotice({
  modality,
  checked,
  onChange,
  disabled = false,
}: {
  modality: "microphone" | "camera";
  checked: boolean;
  onChange: (next: boolean) => void;
  disabled?: boolean;
}) {
  const signal = modality === "microphone" ? "audio" : "video";
  return (
    <div className="mt-3 rounded-xl border border-white/10 bg-ink/50 p-2.5 text-[11px] text-slate-300">
      <p className="leading-relaxed text-slate-300">
        <strong className="text-cyan">Transient session processing:</strong> captured {signal} is sent
        solely to the configured Emotion Studio backend for this session, deleted immediately after
        inference, and never stored permanently or shared with third parties.
      </p>
      <label className="mt-2 flex cursor-pointer items-center gap-2 text-xs text-white">
        <input
          type="checkbox"
          aria-label={`Consent to temporary local ${modality} capture`}
          checked={checked}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
          className="h-3.5 w-3.5 rounded border-white/30 bg-ink text-cyan accent-cyan"
        />
        <span>I consent to temporary local {signal} capture for emotion analysis</span>
      </label>
    </div>
  );
}

function describeMediaError(err: unknown, deviceLabel: string): string {
  if (err instanceof DOMException) {
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      return `${deviceLabel} access was denied. Allow browser permission and retry.`;
    }
    if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
      return `No ${deviceLabel.toLowerCase()} device was found on this system.`;
    }
    if (err.name === "NotReadableError" || err.name === "TrackStartError") {
      return `${deviceLabel} is already in use by another application.`;
    }
    if (err.name === "OverconstrainedError") {
      return `Selected ${deviceLabel.toLowerCase()} is unavailable or does not match constraints. Try another device.`;
    }
    if (err.name === "AbortError") {
      return `${deviceLabel} capture was aborted before completing.`;
    }
    if (err.name === "SecurityError") {
      return `${deviceLabel} access requires a secure browser context (localhost or HTTPS).`;
    }
  }
  return err instanceof Error ? err.message : `Unable to access ${deviceLabel.toLowerCase()}.`;
}

function useMediaDevices(kind: "audioinput" | "videoinput") {
  const [devices, setDevices] = useState<MediaDeviceOption[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");

  const refreshDevices = useCallback(async () => {
    const items = await listMediaInputDevices(kind);
    setDevices(items);
    setSelectedDeviceId((prev) => {
      if (prev && items.some((d) => d.deviceId === prev)) {
        return prev;
      }
      return items[0]?.deviceId ?? "";
    });
  }, [kind]);

  useEffect(() => {
    void refreshDevices();
    const mediaDevices = navigator.mediaDevices;
    if (!mediaDevices?.addEventListener) return undefined;
    const onDeviceChange = () => {
      void refreshDevices();
    };
    mediaDevices.addEventListener("devicechange", onDeviceChange);
    return () => {
      mediaDevices.removeEventListener("devicechange", onDeviceChange);
    };
  }, [refreshDevices]);

  return { devices, selectedDeviceId, setSelectedDeviceId, refreshDevices };
}

function DeviceSelect({
  ariaLabel,
  devices,
  selectedDeviceId,
  disabled,
  onChange,
}: {
  ariaLabel: string;
  devices: MediaDeviceOption[];
  selectedDeviceId: string;
  disabled: boolean;
  onChange: (deviceId: string) => void;
}) {
  if (devices.length === 0) return null;
  return (
    <div className="relative inline-flex max-w-full items-center">
      <select
        aria-label={ariaLabel}
        value={selectedDeviceId}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="max-w-[230px] cursor-pointer truncate appearance-none rounded-lg border border-white/10 bg-ink/80 py-1.5 pl-2.5 pr-7 text-[11px] font-medium text-slate-200 outline-none transition hover:border-white/20 focus:border-cyan/60 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {devices.map((dev) => (
          <option key={dev.deviceId} value={dev.deviceId} className="bg-[#0b1526] text-white">
            {dev.label}
          </option>
        ))}
      </select>
      <ChevronDown size={13} className="pointer-events-none absolute right-2 text-slate-400" />
    </div>
  );
}

export function LiveAudioCapture({
  file,
  onChange,
  onCapturingChange,
  compact = false,
  disabled = false,
}: {
  file: File | null;
  onChange: (file: File | null) => void;
  onCapturingChange?: (capturing: boolean) => void;
  compact?: boolean;
  disabled?: boolean;
}) {
  const [recording, setRecording] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [level, setLevel] = useState(0);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [captureError, setCaptureError] = useState("");
  const [audioLevelReport, setAudioLevelReport] = useState<AudioLevelReport | null>(null);
  const [privacyConsented, setPrivacyConsented] = useState(false);
  const {
    devices: micDevices,
    selectedDeviceId: selectedMicId,
    setSelectedDeviceId: setSelectedMicId,
    refreshDevices: refreshMicDevices,
  } = useMediaDevices("audioinput");

  const mountedRef = useRef(true);
  const interruptedRef = useRef(false);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const meterCtxRef = useRef<AudioContext | null>(null);
  const previewUrlRef = useRef<string | null>(null);

  const stopTracksAndMeter = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (rafRef.current !== null) {
      window.cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (meterCtxRef.current) {
      void meterCtxRef.current.close().catch(() => undefined);
      meterCtxRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setLevel(0);
  }, []);

  const revokePreview = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setPreviewUrl(null);
  }, []);

  useEffect(() => {
    if (!file && previewUrlRef.current) {
      revokePreview();
    }
  }, [file, revokePreview]);

  useEffect(() => {
    onCapturingChange?.(recording || processing);
  }, [recording, processing, onCapturingChange]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        try {
          recorderRef.current.stop();
        } catch {
          // Ignore teardown errors
        }
      }
      stopTracksAndMeter();
      revokePreview();
      onCapturingChange?.(false);
    };
  }, [stopTracksAndMeter, revokePreview, onCapturingChange]);

  const stopRecording = useCallback(() => {
    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") {
      rec.stop();
    }
  }, []);

  const startRecording = async () => {
    if (disabled || recording || processing) return;
    if (!privacyConsented) {
      setCaptureError("Please acknowledge the local capture privacy notice before activating the microphone.");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setCaptureError("Live microphone recording is unavailable in this browser context.");
      return;
    }

    setCaptureError("");
    setAudioLevelReport(null);
    interruptedRef.current = false;
    revokePreview();
    onChange(null);
    chunksRef.current = [];

    try {
      const audioConstraint: boolean | MediaTrackConstraints = selectedMicId
        ? { deviceId: { exact: selectedMicId } }
        : true;
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraint });
      } catch (innerErr) {
        if (innerErr instanceof DOMException && innerErr.name === "OverconstrainedError" && selectedMicId) {
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } else {
          throw innerErr;
        }
      }
      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      void refreshMicDevices();

      const track = stream.getAudioTracks()[0];
      if (track) {
        track.addEventListener("ended", () => {
          interruptedRef.current = true;
          stopRecording();
        });
      }

      // Live audio level meter (best-effort visual feedback; never block recording if meter creation fails)
      const AudioCtx =
        window.AudioContext ||
        (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (AudioCtx) {
        try {
          const ctx = new AudioCtx();
          meterCtxRef.current = ctx;
          const source = ctx.createMediaStreamSource(stream);
          const analyser = ctx.createAnalyser();
          analyser.fftSize = 256;
          source.connect(analyser);
          const buffer = new Uint8Array(analyser.fftSize);
          const updateMeter = () => {
            analyser.getByteTimeDomainData(buffer);
            let sumSq = 0;
            for (let i = 0; i < buffer.length; i += 1) {
              const norm = (buffer[i] - 128) / 128;
              sumSq += norm * norm;
            }
            const rms = Math.sqrt(sumSq / buffer.length);
            setLevel(Math.min(100, Math.round(rms * 260)));
            rafRef.current = window.requestAnimationFrame(updateMeter);
          };
          rafRef.current = window.requestAnimationFrame(updateMeter);
        } catch {
          if (meterCtxRef.current) {
            void meterCtxRef.current.close().catch(() => undefined);
            meterCtxRef.current = null;
          }
        }
      }

      const mimeType = pickSupportedAudioMimeType();
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      recorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        stopTracksAndMeter();
        if (!mountedRef.current) return;
        setRecording(false);
        setCaptureError("Microphone recording failed unexpectedly.");
      };

      recorder.onstop = () => {
        const recordedMime = recorder.mimeType || mimeType || "audio/webm";
        const recordedBlob = new Blob(chunksRef.current, { type: recordedMime });
        const wasInterrupted = interruptedRef.current;
        stopTracksAndMeter();
        if (!mountedRef.current) return;
        setRecording(false);

        if (wasInterrupted && recordedBlob.size === 0) {
          setCaptureError("Microphone disconnected during capture. Reconnect a microphone and retry.");
          onChange(null);
          return;
        }
        setProcessing(true);

        void blobToWavFile(recordedBlob, `live-speech-${Date.now()}.wav`)
          .then(async (wavFile) => {
            if (!mountedRef.current) return;
            const sizeErr = validateUploadFileSize(wavFile);
            if (sizeErr) {
              setCaptureError(sizeErr);
              onChange(null);
              return;
            }
            const levelReport = measureWavRmsDbFromBuffer(await wavFile.arrayBuffer());
            if (!mountedRef.current) return;
            setAudioLevelReport(levelReport);
            const url = URL.createObjectURL(wavFile);
            previewUrlRef.current = url;
            setPreviewUrl(url);
            if (wasInterrupted) {
              setCaptureError("Microphone disconnected mid-capture; partial audio was kept below.");
            }
            onChange(wavFile);
          })
          .catch((err) => {
            if (!mountedRef.current) return;
            setCaptureError(
              wasInterrupted
                ? "Microphone disconnected before usable audio was captured."
                : err instanceof Error
                ? err.message
                : "Could not convert captured audio to WAV.",
            );
            onChange(null);
          })
          .finally(() => {
            if (mountedRef.current) setProcessing(false);
          });
      };

      setSeconds(0);
      setRecording(true);
      recorder.start(250);

      const startedAt = Date.now();
      timerRef.current = window.setInterval(() => {
        const elapsed = Math.floor((Date.now() - startedAt) / 1000);
        setSeconds(elapsed);
        if (elapsed >= MAX_AUDIO_SECONDS) {
          stopRecording();
        }
      }, 250);
    } catch (err) {
      stopTracksAndMeter();
      setRecording(false);
      setCaptureError(describeMediaError(err, "Microphone"));
    }
  };

  const clearRecording = () => {
    revokePreview();
    setSeconds(0);
    setCaptureError("");
    setAudioLevelReport(null);
    onChange(null);
  };

  return (
    <div
      className={`rounded-2xl border border-dashed border-white/15 bg-ink/40 ${
        compact ? "p-3.5" : "p-5"
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div
            className={`flex h-9 w-9 items-center justify-center rounded-xl ${
              recording ? "bg-coral/20 text-coral" : "bg-cyan/10 text-cyan"
            }`}
          >
            <Mic2 size={18} />
          </div>
          <div>
            <p className="text-xs font-semibold text-white">
              {recording
                ? "Listening on live microphone..."
                : processing
                ? "Encoding 16 kHz mono WAV..."
                : file && previewUrl
                ? "Speech captured — listen below"
                : "Capture live speech via microphone"}
            </p>
            <p className="text-[10px] text-slate-500">
              {recording
                ? `Recording ${formatDuration(seconds)} / ${formatDuration(MAX_AUDIO_SECONDS)}`
                : file
                ? `${file.name} · ${(file.size / 1024).toFixed(1)} KB (16 kHz WAV)`
                : "Recorded directly to 16 kHz mono PCM WAV"}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <DeviceSelect
            ariaLabel="Select microphone device"
            devices={micDevices}
            selectedDeviceId={selectedMicId}
            disabled={disabled || recording || processing}
            onChange={setSelectedMicId}
          />
          {!recording ? (
            <button
              type="button"
              disabled={disabled || processing}
              onClick={() => void startRecording()}
              className="flex items-center gap-1.5 rounded-xl bg-cyan px-3.5 py-2 text-xs font-bold text-ink transition hover:bg-cyan/80 disabled:opacity-50"
            >
              {processing ? (
                <RefreshCw size={14} className="animate-spin" />
              ) : file ? (
                <RotateCcw size={14} />
              ) : (
                <Mic2 size={14} />
              )}
              {file ? "Re-record" : "Start recording"}
            </button>
          ) : (
            <button
              type="button"
              onClick={stopRecording}
              className="flex items-center gap-1.5 rounded-xl bg-coral px-3.5 py-2 text-xs font-bold text-ink transition hover:bg-coral/85"
            >
              <Square size={13} fill="currentColor" />
              Stop capture
            </button>
          )}

          {file && !recording && !processing && (
            <button
              type="button"
              onClick={clearRecording}
              aria-label="Clear captured audio"
              className="rounded-xl border border-white/10 bg-white/5 p-2 text-slate-400 transition hover:text-coral"
            >
              <Trash2 size={14} />
            </button>
          )}
        </div>
      </div>

      {recording && (
        <div className="mt-3 space-y-1.5">
          <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-slate-400">
            <span className="flex items-center gap-1.5 text-coral">
              <span className="h-2 w-2 animate-pulse rounded-full bg-coral" /> Live mic level
            </span>
            <span>{formatDuration(seconds)}</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full rounded-full bg-cyan transition-[width] duration-75"
              style={{ width: `${Math.max(4, level)}%` }}
            />
          </div>
          {level < 5 && (
            <p className="text-[11px] text-amber-200/90">
              Very quiet signal — speak clearly closer to your microphone.
            </p>
          )}
        </div>
      )}

      {previewUrl && !recording && (
        <div className="mt-3 space-y-2 rounded-xl border border-white/10 bg-ink/70 p-2.5">
          <div className="flex items-center justify-between gap-2">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
              Listen to captured audio
            </p>
            {audioLevelReport && (
              <span
                className={`text-[10px] font-semibold ${
                  audioLevelReport.isNearSilent
                    ? "text-coral"
                    : audioLevelReport.isLowVolume
                    ? "text-amber-300"
                    : "text-cyan"
                }`}
              >
                {audioLevelReport.rmsDb.toFixed(1)} dB RMS
              </span>
            )}
          </div>
          <audio controls src={previewUrl} className="h-9 w-full" />
          {audioLevelReport?.isNearSilent ? (
            <p className="text-[11px] text-coral">
              Captured audio is near-silent ({audioLevelReport.rmsDb.toFixed(1)} dB RMS). Check your
              microphone input — audio below -60 dB will be rejected as silent rather than scored.
            </p>
          ) : audioLevelReport?.isLowVolume ? (
            <p className="text-[11px] text-amber-200">
              Low microphone volume detected ({audioLevelReport.rmsDb.toFixed(1)} dB RMS). You can still
              analyze it, or re-record closer to the mic for clearer acoustic prosody.
            </p>
          ) : null}
        </div>
      )}

      <MediaConsentNotice
        modality="microphone"
        checked={privacyConsented}
        disabled={disabled || recording || processing}
        onChange={(next) => {
          setPrivacyConsented(next);
          if (next) setCaptureError("");
        }}
      />

      {captureError && (
        <div className="mt-3 flex items-start gap-2 rounded-xl border border-coral/25 bg-coral/10 p-2.5 text-xs text-coral">
          <CircleAlert size={14} className="mt-0.5 shrink-0" />
          <span>{captureError}</span>
        </div>
      )}
    </div>
  );
}

export function LiveVideoCapture({
  file,
  onChange,
  onCapturingChange,
  disabled = false,
  compact = false,
}: {
  file: File | null;
  onChange: (file: File | null) => void;
  onCapturingChange?: (capturing: boolean) => void;
  disabled?: boolean;
  compact?: boolean;
}) {
  const [cameraActive, setCameraActive] = useState(false);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [micMutedNote, setMicMutedNote] = useState("");
  const [captureError, setCaptureError] = useState("");
  const [privacyConsented, setPrivacyConsented] = useState(false);
  const {
    devices: camDevices,
    selectedDeviceId: selectedCamId,
    setSelectedDeviceId: setSelectedCamId,
    refreshDevices: refreshCamDevices,
  } = useMediaDevices("videoinput");

  const mountedRef = useRef(true);
  const interruptedRef = useRef(false);
  const recordingRef = useRef(false);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const liveVideoRef = useRef<HTMLVideoElement | null>(null);
  const previewUrlRef = useRef<string | null>(null);

  const stopTimerAndStream = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (liveVideoRef.current) {
      liveVideoRef.current.srcObject = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  }, []);

  const revokePreview = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setPreviewUrl(null);
  }, []);

  useEffect(() => {
    if (!file && previewUrlRef.current) {
      revokePreview();
    }
  }, [file, revokePreview]);

  useEffect(() => {
    if (cameraActive && liveVideoRef.current && streamRef.current) {
      try {
        liveVideoRef.current.srcObject = streamRef.current;
      } catch {
        // Ignore preview binding error on non-native stream wrappers
      }
    }
  }, [cameraActive]);

  useEffect(() => {
    recordingRef.current = recording;
    onCapturingChange?.(recording);
  }, [recording, onCapturingChange]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        try {
          recorderRef.current.stop();
        } catch {
          // Ignore teardown errors
        }
      }
      stopTimerAndStream();
      revokePreview();
      onCapturingChange?.(false);
    };
  }, [stopTimerAndStream, revokePreview, onCapturingChange]);

  const acquireCameraStream = async (overrideDeviceId?: string): Promise<MediaStream | null> => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setCaptureError("Live camera capture is unavailable in this browser context.");
      return null;
    }

    setCaptureError("");
    setMicMutedNote("");
    const activeCamId = overrideDeviceId ?? selectedCamId;
    const baseVideo: MediaTrackConstraints = { width: { ideal: 640 }, height: { ideal: 360 } };
    const preferredVideo: MediaTrackConstraints = activeCamId
      ? { ...baseVideo, deviceId: { exact: activeCamId } }
      : { ...baseVideo, facingMode: "user" };
    const fallbackVideo: MediaTrackConstraints = { ...baseVideo, facingMode: "user" };

    const getStream = async (constraints: MediaStreamConstraints) => {
      try {
        return await navigator.mediaDevices.getUserMedia(constraints);
      } catch (err) {
        if (err instanceof DOMException && err.name === "OverconstrainedError" && activeCamId) {
          return navigator.mediaDevices.getUserMedia({ ...constraints, video: fallbackVideo });
        }
        throw err;
      }
    };

    const activateStream = (stream: MediaStream, mutedNote = ""): MediaStream | null => {
      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return null;
      }
      const videoTrack = stream.getVideoTracks().find((track) => track.readyState === "live");
      if (!videoTrack) {
        stream.getTracks().forEach((track) => track.stop());
        setCaptureError("Camera video stream is not active. Check that your camera is connected and not in use.");
        return null;
      }
      videoTrack.addEventListener("ended", () => {
        if (recordingRef.current) {
          interruptedRef.current = true;
          const rec = recorderRef.current;
          if (rec && rec.state !== "inactive") {
            rec.stop();
          }
          return;
        }
        stopTimerAndStream();
        if (mountedRef.current) {
          setCaptureError("Camera disconnected or stream interrupted. Check device connection and retry.");
        }
      });
      streamRef.current = stream;
      setCameraActive(true);
      if (mutedNote) setMicMutedNote(mutedNote);
      void refreshCamDevices();
      return stream;
    };

    try {
      const stream = await getStream({ video: preferredVideo, audio: true });
      return activateStream(stream);
    } catch {
      try {
        const videoOnly = await getStream({ video: preferredVideo, audio: false });
        return activateStream(videoOnly, "Microphone unavailable — capturing silent video.");
      } catch (err) {
        if (mountedRef.current) {
          setCaptureError(describeMediaError(err, "Camera"));
        }
        return null;
      }
    }
  };

  const openCameraPreview = async (overrideDeviceId?: string) => {
    if (disabled || recording) return;
    if (!privacyConsented) {
      setCaptureError("Please acknowledge the local capture privacy notice before activating the camera.");
      return;
    }
    revokePreview();
    onChange(null);
    if (streamRef.current) {
      stopTimerAndStream();
    }
    await acquireCameraStream(overrideDeviceId);
  };

  const handleSelectCamera = (nextDeviceId: string) => {
    setSelectedCamId(nextDeviceId);
    if (cameraActive && !recording) {
      void openCameraPreview(nextDeviceId);
    }
  };

  const stopRecording = useCallback(() => {
    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") {
      rec.stop();
    }
  }, []);

  const startRecording = async () => {
    if (disabled || recording) return;
    if (!privacyConsented) {
      setCaptureError("Please acknowledge the local capture privacy notice before activating the camera.");
      return;
    }
    setCaptureError("");
    interruptedRef.current = false;
    revokePreview();
    onChange(null);

    const stream = streamRef.current ?? (await acquireCameraStream());
    if (!stream) return;

    chunksRef.current = [];
    const { mimeType, extension } = pickSupportedVideoMimeType();
    try {
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      recorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        recordingRef.current = false;
        stopTimerAndStream();
        if (!mountedRef.current) return;
        setRecording(false);
        setCaptureError("Camera recording failed unexpectedly.");
      };

      recorder.onstop = () => {
        recordingRef.current = false;
        const resolvedMime = recorder.mimeType || mimeType || (extension === ".mp4" ? "video/mp4" : "video/webm");
        const resolvedExt = resolvedMime.includes("mp4") ? ".mp4" : ".webm";
        const blob = new Blob(chunksRef.current, { type: resolvedMime });
        const wasInterrupted = interruptedRef.current;
        stopTimerAndStream();
        if (!mountedRef.current) return;
        setRecording(false);

        if (blob.size === 0) {
          setCaptureError(
            wasInterrupted
              ? "Camera disconnected during capture before any video data was saved."
              : "Captured video was empty. Try recording at least 1 second.",
          );
          onChange(null);
          return;
        }

        const capturedFile = new File([blob], `live-video-${Date.now()}${resolvedExt}`, {
          type: resolvedMime,
        });
        const sizeErr = validateUploadFileSize(capturedFile);
        if (sizeErr) {
          setCaptureError(sizeErr);
          onChange(null);
          return;
        }
        const url = URL.createObjectURL(capturedFile);
        previewUrlRef.current = url;
        setPreviewUrl(url);
        if (wasInterrupted) {
          setCaptureError("Camera disconnected mid-capture; partial clip was kept below.");
        }
        onChange(capturedFile);
      };

      setSeconds(0);
      recordingRef.current = true;
      setRecording(true);
      recorder.start(250);

      const startedAt = Date.now();
      timerRef.current = window.setInterval(() => {
        const elapsed = Math.floor((Date.now() - startedAt) / 1000);
        setSeconds(elapsed);
        if (elapsed >= MAX_VIDEO_SECONDS) {
          stopRecording();
        }
      }, 250);
    } catch (err) {
      stopTimerAndStream();
      setRecording(false);
      setCaptureError(describeMediaError(err, "Camera"));
    }
  };

  const resetCapture = () => {
    if (recording) stopRecording();
    stopTimerAndStream();
    revokePreview();
    setSeconds(0);
    setCaptureError("");
    setMicMutedNote("");
    onChange(null);
  };

  return (
    <div
      className={`space-y-3 rounded-2xl border border-dashed border-white/15 bg-ink/40 ${
        compact ? "p-3" : "p-4"
      }`}
    >
      <div
        className={`relative flex w-full items-center justify-center overflow-hidden rounded-xl border border-white/10 bg-black/70 ${
          compact ? "h-40" : "h-56"
        }`}
      >
        {cameraActive ? (
          <>
            <video
              ref={liveVideoRef}
              muted
              autoPlay
              playsInline
              className="h-full w-full -scale-x-100 object-cover"
            />
            {recording && (
              <div className="absolute left-3 top-3 flex items-center gap-1.5 rounded-full bg-black/75 px-3 py-1 text-[11px] font-bold text-coral backdrop-blur">
                <span className="h-2 w-2 animate-pulse rounded-full bg-coral" />
                REC {formatDuration(seconds)} / {formatDuration(MAX_VIDEO_SECONDS)}
              </div>
            )}
          </>
        ) : previewUrl ? (
          <video controls playsInline src={previewUrl} className="h-full w-full object-contain" />
        ) : (
          <div className="flex flex-col items-center px-4 text-center">
            <Video size={28} className="mb-3 text-cyan" />
            <p className="text-sm font-semibold text-white">Capture live face video &amp; audio</p>
            <p className="mt-1 text-xs text-slate-500">
              Preview camera, record your clip, watch &amp; listen, then analyze facial emotion.
            </p>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs text-slate-400">
          {file ? (
            <span>
              Ready: <strong className="text-white">{file.name}</strong> ({(file.size / (1024 * 1024)).toFixed(2)} MB)
            </span>
          ) : cameraActive ? (
            <span>{recording ? "Recording in progress..." : "Camera preview ready — start recording when ready"}</span>
          ) : (
            <span>Recommended: 3–10 seconds facing the camera</span>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <DeviceSelect
            ariaLabel="Select camera device"
            devices={camDevices}
            selectedDeviceId={selectedCamId}
            disabled={disabled || recording}
            onChange={handleSelectCamera}
          />
          {!cameraActive && !recording && (
            <button
              type="button"
              disabled={disabled}
              onClick={() => void openCameraPreview()}
              className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-white/10 disabled:opacity-50"
            >
              <Camera size={14} />
              Preview camera
            </button>
          )}

          {!recording ? (
            <button
              type="button"
              disabled={disabled}
              onClick={() => void startRecording()}
              className="flex items-center gap-1.5 rounded-xl bg-cyan px-3.5 py-2 text-xs font-bold text-ink transition hover:bg-cyan/80 disabled:opacity-50"
            >
              {file ? <RotateCcw size={14} /> : <Video size={14} />}
              {file ? "Record again" : "Start recording"}
            </button>
          ) : (
            <button
              type="button"
              onClick={stopRecording}
              className="flex items-center gap-1.5 rounded-xl bg-coral px-3.5 py-2 text-xs font-bold text-ink transition hover:bg-coral/85"
            >
              <Square size={13} fill="currentColor" />
              Stop capture
            </button>
          )}

          {(cameraActive || file) && !recording && (
            <button
              type="button"
              onClick={resetCapture}
              aria-label="Clear captured video"
              className="rounded-xl border border-white/10 bg-white/5 p-2 text-slate-400 transition hover:text-coral"
            >
              <Trash2 size={14} />
            </button>
          )}
        </div>
      </div>

      {micMutedNote && <p className="text-[11px] text-slate-400">{micMutedNote}</p>}

      <MediaConsentNotice
        modality="camera"
        checked={privacyConsented}
        disabled={disabled || recording}
        onChange={(next) => {
          setPrivacyConsented(next);
          if (next) setCaptureError("");
        }}
      />

      {captureError && (
        <div className="flex items-start gap-2 rounded-xl border border-coral/25 bg-coral/10 p-2.5 text-xs text-coral">
          <CircleAlert size={14} className="mt-0.5 shrink-0" />
          <span>{captureError}</span>
        </div>
      )}
    </div>
  );
}

