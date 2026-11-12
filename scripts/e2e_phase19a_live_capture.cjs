const path = require('path');
const fs = require('fs');
const puppeteer = require('../frontend/node_modules/puppeteer-core');

const FRONTEND_URL = process.env.FRONTEND_URL || 'http://localhost:5173';
const CHROME_PATH = process.env.CHROME_PATH || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const OUT_JSON = process.env.OUT_JSON || path.resolve(__dirname, '..', 'models', 'facial_emotion', 'resnet18', 'phase19a_browser_validation.json');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function clickTab(page, categoryNeedle) {
  const ok = await page.evaluate((needle) => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const btn = buttons.find((b) => (b.innerText || '').toLowerCase().includes(needle.toLowerCase()));
    if (!btn) return false;
    btn.click();
    return true;
  }, categoryNeedle);
  if (!ok) throw new Error(`Tab containing "${categoryNeedle}" not found`);
  await sleep(250);
}

async function clickButtonContaining(page, needle) {
  const ok = await page.evaluate((text) => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const btn = buttons.find((b) => (b.innerText || '').toLowerCase().includes(text.toLowerCase()) && !b.disabled);
    if (!btn) return false;
    btn.click();
    return true;
  }, needle);
  if (!ok) throw new Error(`Enabled button containing "${needle}" not found`);
}

async function waitBodyText(page, needle, timeoutMs = 8000) {
  await page.waitForFunction(
    (s) => document.body.innerText.toLowerCase().includes(s.toLowerCase()),
    { timeout: timeoutMs },
    needle
  );
}

async function run() {
  const consoleErrors = [];
  const pageErrors = [];

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu', '--window-size=1440,960'],
  });

  try {
    // 1. Enumerate real browser mediaDevices cleanly (without fake flags or simulation)
    const probePage = await browser.newPage();
    await probePage.goto(FRONTEND_URL, { waitUntil: 'networkidle0', timeout: 30000 });
    const osEnumeration = await probePage.evaluate(async () => {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        return { supported: false, counts: {} };
      }
      const list = await navigator.mediaDevices.enumerateDevices();
      const counts = { audioinput: 0, videoinput: 0, audiooutput: 0 };
      const entries = list.map((d) => {
        counts[d.kind] = (counts[d.kind] || 0) + 1;
        return { kind: d.kind, hasId: Boolean(d.deviceId), labelPresent: Boolean(d.label) };
      });
      return { supported: true, counts, entriesCount: entries.length };
    });
    await probePage.close();

    // 2. Automated LiveCapture UI state, device dropdown, disconnect, and >50MB upload guard tests
    const page = await browser.newPage();
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });
    page.on('pageerror', (err) => {
      pageErrors.push(err.message || String(err));
    });

    await page.evaluateOnNewDocument(() => {
      window.__phase19aFaultMode = 'ok';
      window.__activeTrackEndedListeners = [];
      const mockDevices = [
        { deviceId: 'mic-1', kind: 'audioinput', label: 'Simulated Mic 1', groupId: 'g1', toJSON() { return this; } },
        { deviceId: 'mic-2', kind: 'audioinput', label: 'Simulated Mic 2', groupId: 'g2', toJSON() { return this; } },
        { deviceId: 'cam-1', kind: 'videoinput', label: 'Simulated Cam 1', groupId: 'g3', toJSON() { return this; } },
        { deviceId: 'cam-2', kind: 'videoinput', label: 'Simulated Cam 2', groupId: 'g4', toJSON() { return this; } },
      ];

      function makeMockTrack(kind) {
        const listeners = {};
        const track = {
          kind,
          readyState: 'live',
          stop() {
            this.readyState = 'ended';
          },
          addEventListener(ev, fn) {
            listeners[ev] = listeners[ev] || [];
            listeners[ev].push(fn);
            if (ev === 'ended') window.__activeTrackEndedListeners.push(fn);
          },
          removeEventListener() {},
        };
        return track;
      }

      class MockMediaRecorder {
        constructor(stream, opts) {
          this.stream = stream;
          this.mimeType = (opts && opts.mimeType) || 'video/webm';
          this.state = 'inactive';
          this.ondataavailable = null;
          this.onstop = null;
          this.onerror = null;
        }
        static isTypeSupported(t) {
          return t.includes('webm') || t.includes('mp4');
        }
        start() {
          this.state = 'recording';
        }
        stop() {
          if (this.state === 'inactive') return;
          this.state = 'inactive';
          if (this.onstop) this.onstop();
        }
      }
      window.MediaRecorder = MockMediaRecorder;

      Object.defineProperty(navigator, 'mediaDevices', {
        configurable: true,
        value: {
          enumerateDevices: async () => mockDevices,
          addEventListener: () => {},
          removeEventListener: () => {},
          getUserMedia: async (constraints) => {
            const mode = window.__phase19aFaultMode;
            if (mode === 'denied') throw new DOMException('Permission denied', 'NotAllowedError');
            if (mode === 'notfound') throw new DOMException('Device missing', 'NotFoundError');
            if (mode === 'busy') throw new DOMException('Busy', 'NotReadableError');
            if (mode === 'overconstrained') throw new DOMException('Bad constraints', 'OverconstrainedError');
            const tracks = [];
            if (constraints.audio) tracks.push(makeMockTrack('audio'));
            if (constraints.video) tracks.push(makeMockTrack('video'));
            return {
              getTracks: () => tracks,
              getAudioTracks: () => tracks.filter((t) => t.kind === 'audio'),
              getVideoTracks: () => tracks.filter((t) => t.kind === 'video'),
            };
          },
        },
      });
    });

    await page.goto(FRONTEND_URL, { waitUntil: 'networkidle0', timeout: 30000 });

    // A. Audio tab: permission denied + missing + device dropdown + mid-recording disconnect
    await clickTab(page, 'VOICE');
    await page.select('select', 'live');
    await waitBodyText(page, 'capture live speech via microphone');

    const micOptions = await page.$$eval('select[aria-label="Select microphone device"] option', (opts) =>
      opts.map((o) => ({ value: o.value, label: o.textContent }))
    );
    await page.select('select[aria-label="Select microphone device"]', 'mic-2');

    await page.evaluate(() => { window.__phase19aFaultMode = 'denied'; });
    await clickButtonContaining(page, 'Start recording');
    await waitBodyText(page, 'acknowledge the local capture privacy notice');
    await page.click('input[type="checkbox"][aria-label="Consent to temporary local microphone capture"]');
    await clickButtonContaining(page, 'Start recording');
    await waitBodyText(page, 'microphone access was denied');

    await page.evaluate(() => { window.__phase19aFaultMode = 'notfound'; });
    await clickButtonContaining(page, 'Start recording');
    await waitBodyText(page, 'no microphone device was found');

    // Simulate disconnect mid-recording (empty chunk -> friendly disconnect error)
    await page.evaluate(() => {
      window.__phase19aFaultMode = 'ok';
      window.__activeTrackEndedListeners = [];
    });
    await clickButtonContaining(page, 'Start recording');
    await waitBodyText(page, 'listening on live microphone');
    await page.evaluate(() => {
      window.__activeTrackEndedListeners.forEach((fn) => fn());
    });
    await waitBodyText(page, 'microphone disconnected during capture');

    // B. Video tab: camera dropdown switching + permission denied + busy + mid-recording disconnect
    await clickTab(page, 'VISION');
    await page.select('select', 'live');
    await waitBodyText(page, 'capture live face video');

    const camOptions = await page.$$eval('select[aria-label="Select camera device"] option', (opts) =>
      opts.map((o) => ({ value: o.value, label: o.textContent }))
    );
    await page.select('select[aria-label="Select camera device"]', 'cam-2');

    await page.evaluate(() => { window.__phase19aFaultMode = 'denied'; });
    await clickButtonContaining(page, 'Preview camera');
    await waitBodyText(page, 'acknowledge the local capture privacy notice');
    await page.click('input[type="checkbox"][aria-label="Consent to temporary local camera capture"]');
    await clickButtonContaining(page, 'Preview camera');
    await waitBodyText(page, 'camera access was denied');

    await page.evaluate(() => { window.__phase19aFaultMode = 'busy'; });
    await clickButtonContaining(page, 'Preview camera');
    await waitBodyText(page, 'already in use by another application');

    await page.evaluate(() => {
      window.__phase19aFaultMode = 'ok';
      window.__activeTrackEndedListeners = [];
    });
    await clickButtonContaining(page, 'Start recording');
    await waitBodyText(page, 'recording in progress');
    await page.evaluate(() => {
      window.__activeTrackEndedListeners.forEach((fn) => fn());
    });
    await waitBodyText(page, 'camera disconnected during capture');

    // C. Client-side >50MB upload size check on Video upload mode without hitting backend
    await page.select('select[aria-label="Video input source"]', 'upload');
    const hugeRejected = await page.evaluate(() => {
      const input = document.querySelector('input[type="file"]');
      if (!input) return false;
      const big = new File([new Uint8Array(16)], 'oversized.mp4', { type: 'video/mp4' });
      Object.defineProperty(big, 'size', { value: 52 * 1024 * 1024 });
      const dt = new DataTransfer();
      dt.items.add(big);
      input.files = dt.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    });
    await sleep(200);
    const hugeMsgVisible = await page.evaluate(() =>
      document.body.innerText.toLowerCase().includes('exceeds the 50 mb upload limit')
    );

    const summary = {
      passed:
        micOptions.length === 2 &&
        camOptions.length === 2 &&
        hugeRejected &&
        hugeMsgVisible &&
        consoleErrors.length === 0 &&
        pageErrors.length === 0,
      osDeviceEnumeration: osEnumeration,
      uiFaultInjectionChecks: {
        micDeviceSelectorCount: micOptions.length,
        camDeviceSelectorCount: camOptions.length,
        micDeniedHandled: true,
        micMissingHandled: true,
        micDisconnectMidCaptureHandled: true,
        camDeniedHandled: true,
        camBusyHandled: true,
        camDisconnectMidCaptureHandled: true,
        oversized50MbClientGuardTriggered: hugeMsgVisible,
      },
      consoleErrors,
      pageErrors,
    };
    fs.writeFileSync(OUT_JSON, JSON.stringify(summary, null, 2), 'utf-8');
    console.log(JSON.stringify(summary, null, 2));
    if (!summary.passed) process.exitCode = 1;
  } finally {
    await browser.close();
  }
}

run().catch((err) => {
  console.error('Phase 19A browser validation failed:', err);
  process.exit(1);
});

