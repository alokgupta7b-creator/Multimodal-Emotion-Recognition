const TARGET_SAMPLE_RATE = 16_000;
export const MAX_CLIENT_UPLOAD_BYTES = 50 * 1024 * 1024;

export type MediaDeviceOption = {
  deviceId: string;
  label: string;
};

export async function listMediaInputDevices(
  kind: "audioinput" | "videoinput",
): Promise<MediaDeviceOption[]> {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return [];
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter((d) => d.kind === kind && d.deviceId)
      .map((d, index) => ({
        deviceId: d.deviceId,
        label:
          d.label.trim() ||
          `${kind === "audioinput" ? "Microphone" : "Camera"} ${index + 1}`,
      }));
  } catch {
    return [];
  }
}

export function validateUploadFileSize(
  file: File,
  maxBytes: number = MAX_CLIENT_UPLOAD_BYTES,
): string | null {
  if (file.size === 0) {
    return "Selected file is empty (0 bytes).";
  }
  if (file.size > maxBytes) {
    const mb = (maxBytes / (1024 * 1024)).toFixed(0);
    const actualMb = (file.size / (1024 * 1024)).toFixed(1);
    return `File "${file.name}" (${actualMb} MB) exceeds the ${mb} MB upload limit.`;
  }
  return null;
}

export function formatDuration(totalSeconds: number): string {
  const safe = Math.max(0, Math.floor(totalSeconds));
  const mins = String(Math.floor(safe / 60)).padStart(2, "0");
  const secs = String(safe % 60).padStart(2, "0");
  return `${mins}:${secs}`;
}

export function pickSupportedAudioMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type));
}

export function pickSupportedVideoMimeType(): { mimeType?: string; extension: string } {
  if (typeof MediaRecorder === "undefined") {
    return { extension: ".webm" };
  }
  const candidates: Array<{ mimeType: string; extension: string }> = [
    { mimeType: "video/mp4;codecs=avc1.42E01E,mp4a.40.2", extension: ".mp4" },
    { mimeType: "video/mp4;codecs=avc1", extension: ".mp4" },
    { mimeType: "video/mp4", extension: ".mp4" },
    { mimeType: "video/webm;codecs=vp9,opus", extension: ".webm" },
    { mimeType: "video/webm;codecs=vp8,opus", extension: ".webm" },
    { mimeType: "video/webm;codecs=vp8", extension: ".webm" },
    { mimeType: "video/webm", extension: ".webm" },
  ];
  const matched = candidates.find((item) => MediaRecorder.isTypeSupported(item.mimeType));
  return matched ?? { extension: ".webm" };
}

function writeAscii(view: DataView, offset: number, text: string): void {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

export function encodeWavPcm16(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const dataBytes = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true); // PCM subchunk size
  view.setUint16(20, 1, true); // AudioFormat = 1 (PCM)
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); // BlockAlign
  view.setUint16(34, 16, true); // BitsPerSample
  writeAscii(view, 36, "data");
  view.setUint32(40, dataBytes, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return buffer;
}

export type AudioLevelReport = {
  rmsDb: number;
  peakAbs: number;
  isLowVolume: boolean;
  isNearSilent: boolean;
};

export function analyzeWaveformLevel(samples: Float32Array): AudioLevelReport {
  if (samples.length === 0) {
    return { rmsDb: -100, peakAbs: 0, isLowVolume: true, isNearSilent: true };
  }
  let sumSq = 0;
  let peakAbs = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const s = samples[i];
    sumSq += s * s;
    const abs = Math.abs(s);
    if (abs > peakAbs) peakAbs = abs;
  }
  const meanSq = sumSq / samples.length;
  const rms = Math.sqrt(Math.max(meanSq, 1e-16));
  const rmsDb = 20 * Math.log10(rms);
  return {
    rmsDb,
    peakAbs,
    isLowVolume: rmsDb < -40,
    isNearSilent: rmsDb < -58,
  };
}

export function measureWavRmsDbFromBuffer(buffer: ArrayBuffer): AudioLevelReport | null {
  if (buffer.byteLength < 44) return null;
  const view = new DataView(buffer);
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  const wave = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
  if (riff !== "RIFF" || wave !== "WAVE") return null;

  let bitsPerSample = 16;
  let offset = 12;
  while (offset + 8 <= buffer.byteLength) {
    const chunkId = String.fromCharCode(
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
      view.getUint8(offset + 3),
    );
    const chunkSize = view.getUint32(offset + 4, true);
    if (chunkId === "fmt " && offset + 24 <= buffer.byteLength) {
      bitsPerSample = view.getUint16(offset + 22, true);
    } else if (chunkId === "data") {
      if (bitsPerSample !== 16) return null;
      const dataStart = offset + 8;
      const dataEnd = Math.min(buffer.byteLength, dataStart + chunkSize);
      const count = Math.floor((dataEnd - dataStart) / 2);
      const samples = new Float32Array(count);
      for (let i = 0; i < count; i += 1) {
        samples[i] = view.getInt16(dataStart + i * 2, true) / 32768;
      }
      return analyzeWaveformLevel(samples);
    }
    offset += 8 + chunkSize + (chunkSize % 2);
  }
  return null;
}

export function formatProbabilityPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export async function blobToWavFile(blob: Blob, filename = "live-speech.wav"): Promise<File> {
  if (!blob || blob.size === 0) {
    throw new Error("Captured audio was empty. Try speaking slightly longer.");
  }

  const AudioCtx =
    window.AudioContext ||
    (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioCtx) {
    throw new Error("Web Audio API is not supported in this browser.");
  }

  const audioCtx = new AudioCtx();
  try {
    const rawBuffer = await blob.arrayBuffer();
    const decoded = await audioCtx.decodeAudioData(rawBuffer.slice(0));
    if (decoded.length === 0 || decoded.duration < 0.05) {
      throw new Error("Recording was too short. Please record at least 0.2 seconds.");
    }

    // Average multi-channel input deterministically into mono before resampling
    const numChannels = decoded.numberOfChannels;
    const monoSamples = new Float32Array(decoded.length);
    for (let ch = 0; ch < numChannels; ch += 1) {
      const channelData = decoded.getChannelData(ch);
      for (let i = 0; i < decoded.length; i += 1) {
        monoSamples[i] += channelData[i] / numChannels;
      }
    }

    let finalSamples = monoSamples;
    if (decoded.sampleRate !== TARGET_SAMPLE_RATE) {
      const targetLength = Math.max(1, Math.ceil(decoded.duration * TARGET_SAMPLE_RATE));
      const offline = new OfflineAudioContext(1, targetLength, TARGET_SAMPLE_RATE);
      const monoBuffer = offline.createBuffer(1, monoSamples.length, decoded.sampleRate);
      monoBuffer.copyToChannel(monoSamples, 0);
      const source = offline.createBufferSource();
      source.buffer = monoBuffer;
      source.connect(offline.destination);
      source.start(0);
      const rendered = await offline.startRendering();
      finalSamples = rendered.getChannelData(0);
    }

    const wavBuffer = encodeWavPcm16(finalSamples, TARGET_SAMPLE_RATE);
    return new File([wavBuffer], filename, { type: "audio/wav" });
  } finally {
    await audioCtx.close().catch(() => undefined);
  }
}

