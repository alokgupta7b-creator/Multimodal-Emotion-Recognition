const path = require('path');
const fs = require('fs');
const puppeteer = require('../frontend/node_modules/puppeteer-core');

const FRONTEND_URL = process.env.FRONTEND_URL || 'http://localhost:5174';
const CHROME_PATH = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const DEV_WAV_PATH = process.env.DEV_WAV_PATH;
const DEV_MP4_PATH = process.env.DEV_MP4_PATH;
const NOFACE_MP4_PATH = process.env.NOFACE_MP4_PATH;
const OUT_JSON = process.env.OUT_JSON;
const DESKTOP_SHOT = process.env.DESKTOP_SHOT;
const MOBILE_SHOT = process.env.MOBILE_SHOT;

if (!DEV_WAV_PATH || !DEV_MP4_PATH || !OUT_JSON) {
  throw new Error('DEV_WAV_PATH, DEV_MP4_PATH, and OUT_JSON env vars are required');
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function clickTab(page, tabName) {
  const clicked = await page.evaluate((needle) => {
    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
    const btn = tabs.find((b) => (b.innerText || '').toLowerCase().includes(needle.toLowerCase()));
    if (!btn) return false;
    btn.click();
    return true;
  }, tabName);
  if (!clicked) throw new Error(`Tab "${tabName}" not found`);
  await sleep(250);
}

async function waitForPrediction(page, kind) {
  await page.waitForFunction(
    (k) => {
      const txt = document.body.innerText.toLowerCase();
      if (k === 'multimodal') {
        return (
          txt.includes('individual modality breakdown') &&
          (txt.includes('probabilistic ai estimate') || txt.includes('low-confidence estimate'))
        );
      }
      return (
        txt.includes('confidence') &&
        (txt.includes('probabilistic ai estimate') || txt.includes('low-confidence estimate'))
      );
    },
    { timeout: 60000 },
    kind
  );
}

async function setTextarea(page, value) {
  await page.evaluate(() => {
    const ta = document.querySelector('textarea');
    if (ta) ta.focus();
  });
  await page.click('textarea', { clickCount: 3 });
  await page.keyboard.press('Backspace');
  if (value) {
    await page.type('textarea', value);
  }
}

async function attachByAccept(page, acceptNeedle, filePath) {
  const handles = await page.$$('input[type="file"]');
  for (const h of handles) {
    const acc = await page.evaluate((el) => el.getAttribute('accept') || '', h);
    if (acc.includes(acceptNeedle)) {
      await h.uploadFile(filePath);
      await sleep(200);
      return;
    }
  }
  throw new Error(`File input matching accept="${acceptNeedle}" not found`);
}

async function submitPrimary(page) {
  const ok = await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const btn = btns.find(
      (b) =>
        !b.disabled &&
        ((b.innerText || '').includes('Run Candidate 4 Multimodal Prediction') ||
          (b.innerText || '').includes('Run Multimodal Fusion') ||
          (b.innerText || '').includes('Analyze text') ||
          (b.innerText || '').includes('Analyze audio') ||
          (b.innerText || '').includes('Analyze video'))
    );
    if (!btn) return false;
    btn.click();
    return true;
  });
  if (!ok) throw new Error('Primary submit button not found or disabled');
}

async function submitAndCaptureResponse(page, urlPart) {
  const responsePromise = page.waitForResponse(
    (res) => res.url().includes(urlPart) && res.request().method() === 'POST',
    { timeout: 60000 }
  );
  await submitPrimary(page);
  const res = await responsePromise;
  let body = null;
  try {
    body = await res.json();
  } catch (_) {}
  return { url: res.url(), status: res.status(), body };
}

async function run() {
  const consoleErrors = [];
  const pageErrors = [];
  let interceptMode = null;

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu', '--window-size=1920,1080'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });
    await page.setRequestInterception(true);
    page.on('request', (req) => {
      if (interceptMode === '503' && req.method() === 'POST' && req.url().includes('/predict/text')) {
        return req.respond({
          status: 503,
          headers: { 'Access-Control-Allow-Origin': '*' },
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'Simulated backend unavailable during staging retry test' }),
        });
      }
      return req.continue();
    });
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text());
    });
    page.on('pageerror', (err) => {
      pageErrors.push(err.message || String(err));
    });

    await page.goto(FRONTEND_URL, { waitUntil: 'networkidle0', timeout: 30000 });
    await page.waitForFunction(() => document.body.innerText.includes('API connected'), { timeout: 15000 });

    const accessibilityChecks = await page.evaluate(() => {
      const skipLink = document.querySelector('a[href="#main-content"]');
      const mainEl = document.getElementById('main-content');
      const tablist = document.querySelector('[role="tablist"]');
      const tabs = document.querySelectorAll('[role="tab"]');
      const selectedTabs = Array.from(tabs).filter((t) => t.getAttribute('aria-selected') === 'true');
      return {
        hasSkipLink: Boolean(skipLink),
        hasMainLandmark: Boolean(mainEl),
        hasTablist: Boolean(tablist),
        tabCount: tabs.length,
        selectedTabCount: selectedTabs.length,
        uncertaintyPresentBeforeRun: document.body.innerText.toLowerCase().includes('non-diagnostic'),
      };
    });

    const singleModalityResults = {};

    // Words tab + empty guard + 503 Retry recovery
    await clickTab(page, 'Words');
    await submitPrimary(page);
    await page.waitForFunction(
      () => (document.querySelector('[role="alert"]')?.innerText || '').includes('Enter some text before analyzing.'),
      { timeout: 5000 }
    );
    await setTextarea(page, 'I am thrilled the staging verification passed cleanly!');
    interceptMode = '503';
    await submitAndCaptureResponse(page, '/predict/text');
    await page.waitForFunction(
      () => (document.querySelector('[role="alert"]')?.innerText || '').includes('Simulated backend unavailable'),
      { timeout: 5000 }
    );
    interceptMode = null;
    const retryPromise = page.waitForResponse((res) => res.url().includes('/predict/text'), { timeout: 60000 });
    const retryClicked = await page.evaluate(() => {
      const retry = Array.from(document.querySelectorAll('[role="alert"] button')).find((b) =>
        (b.innerText || '').includes('Retry')
      );
      if (!retry) return false;
      retry.click();
      return true;
    });
    if (!retryClicked) throw new Error('Retry button not rendered inside alert');
    const retryRes = await retryPromise;
    const retryJson = await retryRes.json();
    await waitForPrediction(page, 'single');
    singleModalityResults.text_tab = {
      status: retryRes.status(),
      predicted: retryJson.predicted_emotion,
      confidence: retryJson.confidence,
      retryRecovered: true,
    };

    // Voice tab
    await clickTab(page, 'Voice');
    await attachByAccept(page, '.wav', DEV_WAV_PATH);
    const audioPayload = await submitAndCaptureResponse(page, '/predict/audio');
    await waitForPrediction(page, 'single');
    singleModalityResults.speech_tab = {
      status: audioPayload.status,
      predicted: audioPayload.body?.predicted_emotion,
      confidence: audioPayload.body?.confidence,
    };

    // Vision tab + optional no-face rejection check
    await clickTab(page, 'Vision');
    if (NOFACE_MP4_PATH && fs.existsSync(NOFACE_MP4_PATH)) {
      await attachByAccept(page, '.mp4', NOFACE_MP4_PATH);
      await submitAndCaptureResponse(page, '/predict/video');
      await page.waitForFunction(() => Boolean(document.querySelector('[role="alert"]')), { timeout: 20000 });
      singleModalityResults.no_face_error = await page.$eval('[role="alert"]', (el) => el.innerText.trim());
    }
    await attachByAccept(page, '.mp4', DEV_MP4_PATH);
    const videoPayload = await submitAndCaptureResponse(page, '/predict/video');
    await waitForPrediction(page, 'single');
    singleModalityResults.face_tab = {
      status: videoPayload.status,
      predicted: videoPayload.body?.predicted_emotion,
      confidence: videoPayload.body?.confidence,
    };

    // Adaptive Fusion: all 7 combinations
    const comboResults = {};
    const combos = [
      { key: '1_text_only', text: 'Get out of here right now!', wav: false, mp4: false },
      { key: '2_audio_only', text: '', wav: true, mp4: false },
      { key: '3_video_only', text: '', wav: false, mp4: true },
      { key: '4_text_audio', text: 'Get out of here right now!', wav: true, mp4: false },
      { key: '5_text_video', text: 'Wait, is this really happening?', wav: false, mp4: true },
      { key: '6_audio_video', text: '', wav: true, mp4: true },
      { key: '7_text_audio_video', text: 'I am so happy and excited!', wav: true, mp4: true },
    ];

    for (const c of combos) {
      await clickTab(page, 'Words');
      await clickTab(page, 'Adaptive Fusion');
      await setTextarea(page, c.text || '');
      if (c.wav) await attachByAccept(page, '.wav', DEV_WAV_PATH);
      if (c.mp4) await attachByAccept(page, '.mp4', DEV_MP4_PATH);

      const payload = await submitAndCaptureResponse(page, '/predict/multimodal');
      await waitForPrediction(page, 'multimodal');
      const b = payload.body;
      if (!b) throw new Error(`Missing JSON body for ${c.key}`);
      const weights = b.modality_weights || { text: 0, audio: 0, video: 0 };
      const weightSum = Number((weights.text + weights.audio + weights.video).toFixed(4));
      const uiText = await page.evaluate(() => document.body.innerText);
      comboResults[c.key] = {
        status: payload.status,
        predicted: b.predicted_emotion,
        confidence: b.confidence,
        fusion_method: b.fusion_method,
        rule_applied: b.rule_applied,
        modalities_available: b.modalities_available,
        modality_weights: weights,
        weight_sum: weightSum,
        exclusion_reasons: b.exclusion_reasons,
        uiShowsUncertaintyNotice: uiText.toLowerCase().includes('not a medical or psychological diagnosis'),
        uiShowsPredictedEmotion: uiText.toLowerCase().includes(String(b.predicted_emotion).toLowerCase()),
      };
    }

    if (DESKTOP_SHOT) {
      await page.setViewport({ width: 1366, height: 768 });
      await sleep(200);
      fs.mkdirSync(path.dirname(DESKTOP_SHOT), { recursive: true });
      await page.screenshot({ path: DESKTOP_SHOT, fullPage: true });
    }

    const viewports = [
      { width: 1920, height: 1080 },
      { width: 1366, height: 768 },
      { width: 768, height: 1024 },
      { width: 390, height: 844 },
      { width: 360, height: 800 },
    ];
    const responsiveResults = [];
    for (const vp of viewports) {
      await page.setViewport(vp);
      await sleep(180);
      const layout = await page.evaluate(() => {
        const doc = document.documentElement;
        const primaryBtn = Array.from(document.querySelectorAll('button')).find((b) => {
          const t = b.innerText || '';
          return t.includes('Run Multimodal Fusion') || t.includes('Run Candidate 4');
        });
        const rect = primaryBtn ? primaryBtn.getBoundingClientRect() : null;
        return {
          innerWidth: window.innerWidth,
          scrollWidth: doc.scrollWidth,
          horizontalOverflowPx: Math.max(0, doc.scrollWidth - window.innerWidth),
          submitWidthOk: rect ? rect.right <= window.innerWidth + 1 && rect.left >= -1 : false,
        };
      });
      responsiveResults.push({ viewport: `${vp.width}x${vp.height}`, ...layout });
      if (MOBILE_SHOT && vp.width === 390) {
        fs.mkdirSync(path.dirname(MOBILE_SHOT), { recursive: true });
        await page.screenshot({ path: MOBILE_SHOT, fullPage: true });
      }
    }

    const unexpectedConsoleErrors = consoleErrors.filter(
      (txt) => !txt.includes('status of 503') && !txt.includes('status of 422')
    );

    const allCombosOk = Object.values(comboResults).every(
      (r) => r.status === 200 && Math.abs(r.weight_sum - 1.0) < 0.01 && r.uiShowsUncertaintyNotice && r.uiShowsPredictedEmotion
    );
    const allViewportsOk = responsiveResults.every((r) => r.horizontalOverflowPx === 0 && r.submitWidthOk);

    const passed =
      accessibilityChecks.hasSkipLink &&
      accessibilityChecks.hasMainLandmark &&
      accessibilityChecks.hasTablist &&
      accessibilityChecks.tabCount === 4 &&
      accessibilityChecks.selectedTabCount === 1 &&
      singleModalityResults.text_tab?.retryRecovered === true &&
      singleModalityResults.speech_tab?.status === 200 &&
      singleModalityResults.face_tab?.status === 200 &&
      allCombosOk &&
      allViewportsOk &&
      unexpectedConsoleErrors.length === 0 &&
      pageErrors.length === 0;

    const output = {
      passed,
      accessibility: accessibilityChecks,
      singleModalityResults,
      multimodalSevenCombinations: comboResults,
      responsiveViewports: responsiveResults,
      unexpectedConsoleErrors,
      allConsoleErrorsCount: consoleErrors.length,
      pageErrors,
    };

    fs.mkdirSync(path.dirname(OUT_JSON), { recursive: true });
    fs.writeFileSync(OUT_JSON, JSON.stringify(output, null, 2), 'utf-8');
    console.log(JSON.stringify({ passed, viewports: responsiveResults.length, combos: Object.keys(comboResults).length }));
    if (!passed) process.exitCode = 1;
  } finally {
    await browser.close();
  }
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});

