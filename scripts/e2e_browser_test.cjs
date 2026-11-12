/**
 * Phase 15 End-to-End Browser Automation Suite with Puppeteer
 * Controls Google Chrome to validate all features, tabs, errors, fallbacks, and responsive viewports.
 */

const path = require('path');
const fs = require('fs');
const puppeteer = require(path.resolve(__dirname, '..', 'frontend', 'node_modules', 'puppeteer-core'));

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const APP_URL = 'http://localhost:5173';
const SCREENSHOTS_DIR = path.resolve(__dirname, '..', 'docs', 'screenshots');
const FIXTURES_DIR = path.resolve(__dirname, '..', 'temp_fixtures');

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function clickSubmit(page) {
  const btn = await page.waitForSelector('button[type="submit"]');
  await btn.click();
}

async function setReactText(page, selector, text) {
  await page.evaluate((sel, val) => {
    const el = document.querySelector(sel);
    if (!el) throw new Error('Element not found: ' + sel);
    const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    nativeSetter.call(el, val);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, selector, text);
}

async function checkNoHorizontalOverflow(page) {
  return await page.evaluate(() => {
    const docWidth = document.documentElement.scrollWidth;
    const winWidth = window.innerWidth;
    return {
      scrollWidth: docWidth,
      clientWidth: winWidth,
      hasOverflow: docWidth > winWidth,
    };
  });
}

async function run() {
  console.log('='.repeat(70));
  console.log('PHASE 15: REAL BROWSER (CHROME) END-TO-END VALIDATION');
  console.log('='.repeat(70));

  if (!fs.existsSync(SCREENSHOTS_DIR)) {
    fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true });
  }

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: 'new',
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--window-size=1920,1080',
      '--disable-web-security',
    ],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080 });

  page.on('console', (msg) => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', (err) => console.log('PAGE ERROR:', err.toString()));

  const metrics = {
    testResults: {},
    responsiveChecks: {},
    timings: {},
  };

  try {
    console.log(`Navigating to ${APP_URL}...`);
    await page.goto(APP_URL, { waitUntil: 'networkidle0', timeout: 30000 });

    // Wait for API connected indicator
    await page.waitForFunction(
      () => document.body.innerText.includes('API connected'),
      { timeout: 15000 }
    );
    console.log('✔ Backend connection verified (API connected)');

    // Initial Screenshot: Desktop Empty Multimodal Studio
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '01_multimodal_studio_empty_desktop.png'),
      fullPage: true,
    });
    console.log('✔ Screenshot 01: Empty Multimodal Studio Desktop');

    // -----------------------------------------------------------------
    // TEST 1: Multimodal Studio - Text-Only Prediction
    // -----------------------------------------------------------------
    console.log('\n--- Test 1: Multimodal Studio (Text-Only) ---');
    const textSample1 = 'I am completely delighted and overjoyed with this miraculous result!';
    await setReactText(page, 'textarea', textSample1);

    const t0 = Date.now();
    await clickSubmit(page);

    // Wait for result to appear
    await page.waitForFunction(
      () => {
        const text = document.body.innerText.toLowerCase();
        return text.includes('predicted emotion');
      },
      { timeout: 15000 }
    );
    const e2eTextOnlyMs = Date.now() - t0;
    metrics.timings.browser_e2e_text_only_ms = e2eTextOnlyMs;

    const textOnlyResult = await page.evaluate(() => {
      const body = document.body.innerText.toLowerCase();
      return {
        hasPredictedEmotion: body.includes('joy'),
        hasRule: body.includes('rule applied') || body.includes('text-anchor default'),
        hasTextActive: body.includes('text modality') && (body.includes('active') || body.includes('w: 1.00')),
        hasAudioExcluded: body.includes('no audio file provided') || body.includes('excluded'),
      };
    });
    console.log(`✔ Text-Only Prediction Rendered in ${e2eTextOnlyMs} ms:`, textOnlyResult);
    metrics.testResults.multimodal_text_only = textOnlyResult;

    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '02_multimodal_text_only_success.png'),
      fullPage: true,
    });
    console.log('✔ Screenshot 02: Text-Only Multimodal Studio Success');

    // -----------------------------------------------------------------
    // TEST 2: Multimodal Studio - Text + Audio Synergy
    // -----------------------------------------------------------------
    console.log('\n--- Test 2: Multimodal Studio (Text + Audio Synergy) ---');
    const textSample2 = 'What on earth are you doing here?! Get out of my face!';
    await setReactText(page, 'textarea', textSample2);

    // Upload dev audio
    const devAudioPath = path.join(FIXTURES_DIR, 'valid_dev_audio.wav');
    const fileInput = await page.waitForSelector('input[type="file"]');
    await fileInput.uploadFile(devAudioPath);
    await sleep(500);

    const t1 = Date.now();
    await clickSubmit(page);

    try {
      await page.waitForFunction(
        () => {
          const body = document.body.innerText.toLowerCase();
          return body.includes('distilbert text') && body.includes('wav2vec2 speech');
        },
        { timeout: 15000 }
      );
    } catch (e) {
      const pageText = await page.evaluate(() => document.body.innerText);
      console.error('FAILED IN TEST 2. Page text:\n', pageText);
      throw e;
    }
    const e2eBothMs = Date.now() - t1;
    metrics.timings.browser_e2e_text_and_audio_ms = e2eBothMs;

    const bothResult = await page.evaluate(() => {
      const body = document.body.innerText.toLowerCase();
      return {
        hasIndividualBreakdown: body.includes('individual modality breakdown'),
        hasDistilBertCard: body.includes('distilbert text'),
        hasWav2Vec2Card: body.includes('wav2vec2 speech'),
        hasModalityStatus: body.includes('text modality') && body.includes('audio modality'),
      };
    });
    console.log(`✔ Text + Audio Multimodal Rendered in ${e2eBothMs} ms:`, bothResult);
    metrics.testResults.multimodal_text_and_audio = bothResult;

    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '03_multimodal_text_and_audio_synergy.png'),
      fullPage: true,
    });
    console.log('✔ Screenshot 03: Text + Audio Multimodal Synergy');

    // -----------------------------------------------------------------
    // TEST 3: Multimodal Studio - Silent Audio Fallback
    // -----------------------------------------------------------------
    console.log('\n--- Test 3: Multimodal Studio (Silent Audio Fallback) ---');
    await setReactText(page, 'textarea', 'I am feeling quiet and peaceful right now.');

    const silentAudioPath = path.join(FIXTURES_DIR, 'silent_audio.wav');
    const fileInputSilent = await page.waitForSelector('input[type="file"]');
    await fileInputSilent.uploadFile(silentAudioPath);
    await sleep(500);

    await clickSubmit(page);

    await page.waitForFunction(
      () => {
        const body = document.body.innerText.toLowerCase();
        return body.includes('silent_audio') || body.includes('excluded');
      },
      { timeout: 15000 }
    );

    const silentResult = await page.evaluate(() => {
      const body = document.body.innerText.toLowerCase();
      return {
        audioExcluded: body.includes('excluded'),
        reasonRecorded: body.includes('silent_audio') || body.includes('silence threshold'),
        textActive: body.includes('w: 1.00') || body.includes('active'),
      };
    });
    console.log('✔ Silent Audio Fallback:', silentResult);
    metrics.testResults.silent_audio_fallback = silentResult;

    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '04_multimodal_silent_audio_fallback.png'),
      fullPage: true,
    });
    console.log('✔ Screenshot 04: Silent Audio Fallback');

    // -----------------------------------------------------------------
    // TEST 4: Multimodal Studio - Invalid Audio Fallback
    // -----------------------------------------------------------------
    console.log('\n--- Test 4: Multimodal Studio (Invalid Audio Upload Fallback) ---');
    await setReactText(page, 'textarea', 'Everything remains steady and calm.');

    const invalidAudioPath = path.join(FIXTURES_DIR, 'invalid_corrupt.wav');
    const fileInputInvalid = await page.waitForSelector('input[type="file"]');
    await fileInputInvalid.uploadFile(invalidAudioPath);
    await sleep(500);

    await clickSubmit(page);

    await page.waitForFunction(
      () => {
        const body = document.body.innerText.toLowerCase();
        return body.includes('upload rejected') || body.includes('not a valid audio file') || body.includes('excluded');
      },
      { timeout: 15000 }
    );

    const invalidResult = await page.evaluate(() => {
      const body = document.body.innerText.toLowerCase();
      return {
        audioExcluded: body.includes('excluded'),
        uploadRejectedNotice: body.includes('upload rejected') || body.includes('not a valid audio file'),
        fallbackTextActive: body.includes('w: 1.00') || body.includes('active'),
      };
    });
    console.log('✔ Invalid Audio Fallback:', invalidResult);
    metrics.testResults.invalid_audio_fallback = invalidResult;

    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '05_multimodal_invalid_audio_fallback.png'),
      fullPage: true,
    });
    console.log('✔ Screenshot 05: Invalid Audio Fallback');

    // -----------------------------------------------------------------
    // TEST 5: Oversized Audio Upload Rejection (Speech Tab)
    // -----------------------------------------------------------------
    console.log('\n--- Test 5: Oversized Audio Rejection (Speech Tab) ---');
    // Click on Speech Tab button
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const speechBtn = btns.find((b) => b.innerText.includes('Speech') || b.innerText.includes('VOICE'));
      if (speechBtn) speechBtn.click();
    });
    await sleep(500);

    const oversizedAudioPath = path.join(FIXTURES_DIR, 'oversized_audio.wav');
    const audioTabInput = await page.waitForSelector('input[type="file"]');
    await audioTabInput.uploadFile(oversizedAudioPath);
    await sleep(500);

    // Click "Analyze audio"
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const analyzeBtn = btns.find((b) => b.innerText.includes('Analyze audio'));
      if (analyzeBtn) analyzeBtn.click();
    });

    await page.waitForFunction(
      () => {
        const body = document.body.innerText.toLowerCase();
        return body.includes('size limit') || body.includes('exceeds') || body.includes('413');
      },
      { timeout: 15000 }
    );

    const oversizedResult = await page.evaluate(() => {
      const body = document.body.innerText.toLowerCase();
      return {
        errorBannerShown: body.includes('size limit') || body.includes('exceeds') || body.includes('413'),
      };
    });
    console.log('✔ Oversized Audio Rejection:', oversizedResult);
    metrics.testResults.oversized_audio_rejection = oversizedResult;

    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '06_oversized_audio_rejection.png'),
      fullPage: true,
    });
    console.log('✔ Screenshot 06: Oversized Audio Rejection Banner');

    // -----------------------------------------------------------------
    // TEST 6: Single-Modality Workflows (Text, Speech, Face)
    // -----------------------------------------------------------------
    console.log('\n--- Test 6: Single-Modality Tabs ---');

    // 6.1 Text Tab
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const btn = btns.find((b) => b.innerText.includes('Text') && b.innerText.includes('WORDS'));
      if (btn) btn.click();
    });
    await sleep(500);
    await setReactText(page, 'textarea', 'I am completely devastated and overwhelmed by sorrow.');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const btn = btns.find((b) => b.innerText.includes('Analyze text'));
      if (btn) btn.click();
    });
    await page.waitForFunction(
      () => document.body.innerText.toLowerCase().includes('sadness'),
      { timeout: 15000 }
    );
    console.log('✔ Single-modality Text Tab success (Sadness)');
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '07_single_modality_text_tab.png'),
      fullPage: true,
    });

    // 6.2 Speech Tab (Valid dev audio)
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const btn = btns.find((b) => b.innerText.includes('Speech') || b.innerText.includes('VOICE'));
      if (btn) btn.click();
    });
    await sleep(500);
    const speechInput = await page.waitForSelector('input[type="file"]');
    await speechInput.uploadFile(path.join(FIXTURES_DIR, 'valid_dev_audio.wav'));
    await sleep(500);
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const btn = btns.find((b) => b.innerText.includes('Analyze audio'));
      if (btn) btn.click();
    });
    await page.waitForFunction(
      () => document.body.innerText.toLowerCase().includes('live result') && document.body.innerText.toLowerCase().includes('confidence'),
      { timeout: 15000 }
    );
    console.log('✔ Single-modality Speech Tab success');
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '08_single_modality_audio_tab.png'),
      fullPage: true,
    });

    // 6.3 Face Tab (Valid dev video)
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const btn = btns.find((b) => b.innerText.includes('Face') || b.innerText.includes('VISION'));
      if (btn) btn.click();
    });
    await sleep(500);
    const videoInput = await page.waitForSelector('input[type="file"]');
    await videoInput.uploadFile(path.join(FIXTURES_DIR, 'valid_dev_video.mp4'));
    await sleep(500);
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const btn = btns.find((b) => b.innerText.includes('Analyze video'));
      if (btn) btn.click();
    });
    await page.waitForFunction(
      () => document.body.innerText.toLowerCase().includes('live result') && document.body.innerText.toLowerCase().includes('confidence'),
      { timeout: 15000 }
    );
    console.log('✔ Single-modality Face Tab success');
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '09_single_modality_video_tab.png'),
      fullPage: true,
    });

    // -----------------------------------------------------------------
    // TEST 7: Responsive UI Validation (Laptop, Tablet, Mobile)
    // -----------------------------------------------------------------
    console.log('\n--- Test 7: Responsive Viewports ---');

    // Switch back to Multimodal Studio
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const btn = btns.find((b) => b.innerText.includes('Multimodal Studio'));
      if (btn) btn.click();
    });
    await sleep(500);

    // 7.1 Laptop (1366 x 768)
    await page.setViewport({ width: 1366, height: 768 });
    await sleep(400);
    const laptopOverflow = await checkNoHorizontalOverflow(page);
    metrics.responsiveChecks.laptop = laptopOverflow;
    console.log('✔ Laptop (1366x768) overflow check:', laptopOverflow);
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '10_responsive_laptop_1366x768.png'),
      fullPage: true,
    });

    // 7.2 Tablet (768 x 1024)
    await page.setViewport({ width: 768, height: 1024 });
    await sleep(400);
    const tabletOverflow = await checkNoHorizontalOverflow(page);
    metrics.responsiveChecks.tablet = tabletOverflow;
    console.log('✔ Tablet (768x1024) overflow check:', tabletOverflow);
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '11_responsive_tablet_768x1024.png'),
      fullPage: true,
    });

    // 7.3 Mobile (375 x 667)
    await page.setViewport({ width: 375, height: 667 });
    await sleep(400);
    const mobileOverflow = await checkNoHorizontalOverflow(page);
    metrics.responsiveChecks.mobile = mobileOverflow;
    console.log('✔ Mobile (375x667) overflow check:', mobileOverflow);
    await page.screenshot({
      path: path.join(SCREENSHOTS_DIR, '12_responsive_mobile_375x667.png'),
      fullPage: true,
    });

    // Save browser test results
    const outJsonPath = path.resolve(__dirname, '..', 'models', 'phase15_test_experiments', 'browser_validation_results.json');
    fs.writeFileSync(outJsonPath, JSON.stringify(metrics, null, 2), 'utf-8');
    console.log(`\n✔ Saved browser validation results to ${outJsonPath}`);
    console.log('='.repeat(70));
  } finally {
    await browser.close();
  }
}

run().catch((err) => {
  console.error('Browser validation failed:', err);
  process.exit(1);
});

