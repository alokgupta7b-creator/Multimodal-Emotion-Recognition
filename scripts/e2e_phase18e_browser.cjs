const path = require('path');
const fs = require('fs');
const puppeteer = require('../frontend/node_modules/puppeteer-core');

const FRONTEND_URL = process.env.FRONTEND_URL || 'http://127.0.0.1:5173';
const CHROME_PATH = process.env.CHROME_PATH || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const TEST_WAV = process.env.TEST_WAV || '';
const TEST_MP4 = process.env.TEST_MP4 || '';
const OUT_JSON = path.resolve(__dirname, '..', 'models', 'facial_emotion', 'resnet18', 'phase18e_browser_validation.json');
const SCREENSHOT_DIR = process.env.SCREENSHOT_DIR || path.resolve(__dirname, '..', 'models', 'facial_emotion', 'resnet18', 'browser_screenshots');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function setTextareaValue(page, value) {
  await page.evaluate((text) => {
    const el = document.querySelector('textarea');
    if (!el) throw new Error('Textarea not found');
    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(el, text);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await sleep(150);
}

async function clickButtonContaining(page, needle) {
  const clicked = await page.evaluate((text) => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const btn = buttons.find((b) => (b.innerText || '').toLowerCase().includes(text.toLowerCase()) && !b.disabled);
    if (!btn) return false;
    btn.click();
    return true;
  }, needle);
  if (!clicked) {
    throw new Error(`Button containing "${needle}" not found or disabled`);
  }
}

async function waitForBodyIncludes(page, needles, timeoutMs = 30000) {
  await page.waitForFunction(
    (items) => {
      const body = document.body.innerText.toLowerCase();
      return items.every((item) => body.includes(item.toLowerCase()));
    },
    { timeout: timeoutMs },
    needles
  );
}

async function checkOverflow(page) {
  return page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    hasOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
  }));
}

async function run() {
  if (!TEST_WAV || !fs.existsSync(TEST_WAV)) {
    throw new Error(`TEST_WAV missing or not found: ${TEST_WAV}`);
  }
  if (!TEST_MP4 || !fs.existsSync(TEST_MP4)) {
    throw new Error(`TEST_MP4 missing or not found: ${TEST_MP4}`);
  }
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

  const consoleErrors = [];
  const pageErrors = [];
  const serverErrors = [];
  const failedRequests = [];

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu', '--window-size=1920,1080'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });

    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });
    page.on('pageerror', (err) => {
      pageErrors.push(err.message || String(err));
    });
    page.on('response', (res) => {
      if (res.status() >= 500) {
        serverErrors.push({ url: res.url(), status: res.status() });
      }
    });
    page.on('requestfailed', (req) => {
      failedRequests.push({ url: req.url(), failure: req.failure()?.errorText || 'unknown' });
    });

    await page.goto(FRONTEND_URL, { waitUntil: 'networkidle0', timeout: 30000 });
    await waitForBodyIncludes(page, ['api connected', 'live backend status', 'multimodal studio'], 20000);

    const statusText = await page.evaluate(() => document.body.innerText);
    const desktopInitialOverflow = await checkOverflow(page);
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_initial.png'), fullPage: true });

    // Verify Live mic & Live camera dropdown toggles in Multimodal Studio work cleanly
    await page.select('select[aria-label="Multimodal speech source"]', 'live');
    await waitForBodyIncludes(page, ['capture live speech via microphone'], 5000);
    await page.select('select[aria-label="Multimodal speech source"]', 'upload');

    await page.select('select[aria-label="Multimodal video source"]', 'live');
    await waitForBodyIncludes(page, ['capture live face video'], 5000);
    await page.select('select[aria-label="Multimodal video source"]', 'upload');

    // A. Multimodal Text-Only (Candidate 4 preserved)
    await setTextareaValue(page, 'I am completely delighted by this wonderful news!');
    await clickButtonContaining(page, 'Run Candidate 4 Multimodal Prediction');
    await waitForBodyIncludes(page, ['text_anchored_class_conditional_bounded_fusion_v1', 'distilbert text', 'resnet18 face']);
    const textOnlyCheck = await page.evaluate(() => ({
      bodyHasCandidate4: document.body.innerText.includes('text_anchored_class_conditional_bounded_fusion_v1'),
      videoExcluded: document.body.innerText.includes('No video file provided'),
    }));

    // B. Multimodal Text + Video (ResNet18 integrated)
    await setTextareaValue(page, 'Wait, seriously?! What on earth just happened?!');
    const mp4Input = await page.$('input[type="file"][accept*=".mp4"]');
    await mp4Input.uploadFile(TEST_MP4);
    await sleep(300);
    await clickButtonContaining(page, 'Run Multimodal Fusion');
    await page.waitForFunction(
      () => {
        const body = document.body.innerText.toLowerCase();
        return body.includes('candidate4_plus_validated_resnet18_video_v1') && !body.includes('no video file provided');
      },
      { timeout: 30000 }
    );
    const textVideoCheck = await page.evaluate(() => ({
      bodyHasExtendedId: document.body.innerText.includes('candidate4_plus_validated_resnet18_video_v1'),
      hasResnetBreakdown: document.body.innerText.includes('ResNet18 Face'),
    }));

    // C. Multimodal Text + Audio + Video (all 3 active)
    const wavInput = await page.$('input[type="file"][accept*=".wav"]');
    await wavInput.uploadFile(TEST_WAV);
    await sleep(300);
    await clickButtonContaining(page, 'Run Multimodal Fusion');
    await page.waitForFunction(
      () => {
        const body = document.body.innerText.toLowerCase();
        return (
          body.includes('candidate4_plus_validated_resnet18_video_v1') &&
          !body.includes('no audio file provided') &&
          !body.includes('no video file provided')
        );
      },
      { timeout: 30000 }
    );
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_multimodal_tav.png'), fullPage: true });
    const tavCheck = await page.evaluate(() => {
      const text = document.body.innerText.toLowerCase();
      return {
        hasDistilBert: text.includes('distilbert text'),
        hasWav2Vec2: text.includes('wav2vec2 speech'),
        hasResNet18: text.includes('resnet18 face'),
        activeWeightCount: (text.match(/active \(w:/g) || []).length,
        hasNotActive: text.includes('not active'),
      };
    });

    // D. Multimodal Video-Only (Text cleared, Audio removed)
    await setTextareaValue(page, '');
    await clickButtonContaining(page, 'Remove audio');
    await sleep(200);
    await clickButtonContaining(page, 'Run Multimodal Fusion');
    await waitForBodyIncludes(page, ['single_modality_video_resnet18_v1', 'active (w: 1.00)'], 30000);

    // E. Single-modality tabs (Text, Speech, Face) + live toggles
    await clickButtonContaining(page, 'WORDS');
    await sleep(300);
    await setTextareaValue(page, 'I feel peaceful and calm today.');
    await clickButtonContaining(page, 'Analyze text');
    await sleep(200);
    await page.waitForFunction(
      () => !document.querySelector('.animate-spin') && document.body.innerText.toLowerCase().includes('confidence'),
      { timeout: 25000 }
    );

    await clickButtonContaining(page, 'VOICE');
    await sleep(300);
    await page.select('select', 'live');
    await waitForBodyIncludes(page, ['capture live speech via microphone'], 5000);
    await page.select('select', 'upload');
    const speechUpload = await page.$('input[type="file"][accept*=".wav"]');
    await speechUpload.uploadFile(TEST_WAV);
    await sleep(200);
    await clickButtonContaining(page, 'Analyze audio');
    await sleep(200);
    await page.waitForFunction(
      () => !document.querySelector('.animate-spin') && document.body.innerText.toLowerCase().includes('confidence'),
      { timeout: 25000 }
    );

    await clickButtonContaining(page, 'VISION');
    await sleep(300);
    await page.select('select', 'live');
    await waitForBodyIncludes(page, ['capture live face video'], 5000);
    await page.select('select', 'upload');
    const faceUpload = await page.$('input[type="file"][accept*=".mp4"]');
    await faceUpload.uploadFile(TEST_MP4);
    await sleep(200);
    await clickButtonContaining(page, 'Analyze video');
    await sleep(200);
    await page.waitForFunction(
      () => !document.querySelector('.animate-spin') && document.body.innerText.toLowerCase().includes('confidence'),
      { timeout: 30000 }
    );
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'desktop_face_tab.png'), fullPage: true });

    // Mobile viewport check (390x844) on Multimodal Studio
    await clickButtonContaining(page, 'Multimodal Studio');
    await sleep(300);
    await page.setViewport({ width: 390, height: 844 });
    await sleep(400);
    const mobileOverflow = await checkOverflow(page);
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, 'mobile_multimodal_390x844.png'), fullPage: true });

    const summary = {
      passed:
        consoleErrors.length === 0 &&
        pageErrors.length === 0 &&
        serverErrors.length === 0 &&
        failedRequests.length === 0 &&
        !desktopInitialOverflow.hasOverflow &&
        !mobileOverflow.hasOverflow &&
        textOnlyCheck.bodyHasCandidate4 &&
        textVideoCheck.bodyHasExtendedId &&
        tavCheck.hasDistilBert &&
        tavCheck.hasWav2Vec2 &&
        tavCheck.hasResNet18 &&
        !tavCheck.hasNotActive &&
        tavCheck.activeWeightCount >= 2,
      consoleErrors,
      pageErrors,
      serverErrors,
      failedRequests,
      desktop_1920x1080: desktopInitialOverflow,
      mobile_390x844: mobileOverflow,
      textOnlyCheck,
      textVideoCheck,
      tavCheck,
      backendStatusContainsResNetNote: statusText.toLowerCase().includes('cuda') || statusText.toLowerCase().includes('cpu'),
    };
    fs.writeFileSync(OUT_JSON, JSON.stringify(summary, null, 2), 'utf-8');
    console.log(JSON.stringify(summary, null, 2));
    if (!summary.passed) {
      process.exitCode = 1;
    }
  } finally {
    await browser.close();
  }
}

run().catch((err) => {
  console.error('Phase 18E browser verification failed:', err);
  process.exit(1);
});
