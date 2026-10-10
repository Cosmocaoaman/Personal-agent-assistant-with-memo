/* Read-only checks against tools/memory_lab.py --serve. No real API calls. */
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const fs = require('node:fs/promises');

(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.MORI_TEST_BROWSER || 'C:/Program Files/Google/Chrome/Application/chrome.exe'});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  const artifacts = path.resolve('artifacts/memory-lab');
  try {
    await page.goto('http://127.0.0.1:8767/#flow');
    await page.locator('#test-dataset-banner').filter({hasText: '500 条记忆'}).waitFor();
    const state = await page.evaluate(async () => {
      const started = performance.now();
      const response = await fetch('/api/state');
      const body = await response.text();
      return {data: JSON.parse(body), utf8_bytes: new TextEncoder().encode(body).length, response_ms: performance.now() - started};
    });
    assert.equal(state.data.settings.dataset.synthetic, true, 'Refuse to test against a personal database');
    assert.equal(state.data.memories.length, 500);
    assert.equal(state.data.personal_context.memories.length, 24);
    assert.equal(state.data.personal_context.omitted, 151);
    assert.equal(await page.locator('#memory-count').textContent(), '500');
    assert.equal(await page.locator('#key-indicator').textContent(), '待配置');
    await page.locator('[data-view=memory]').click();
    await page.waitForFunction(() => document.querySelectorAll('.memory-card').length === 500);
    await page.locator('#memory-search').fill('Python aiohttp');
    await page.locator('.memory-card').filter({hasText: '异步爬虫'}).waitFor();
    await page.screenshot({path: path.join(artifacts, 'memory-search-desktop.png')});
    await page.locator('[data-view=flow]').click();
    await page.locator('#flow-run-status').filter({hasText: /^已完成$/}).waitFor();
    assert.match(await page.locator('#flow-config-summary').textContent(), /scripted-capacity-probe/);
    await page.locator('.node-recall').click();
    assert.match(await page.locator('#flow-inspector').textContent(), /读入记忆容量/);
    assert.match(await page.locator('#flow-inspector').textContent(), /查询与浏览记录/);
    assert.match(await page.locator('#flow-inspector').textContent(), /容量省略/);
    await page.screenshot({path: path.join(artifacts, 'scale-flow-desktop.png')});
    await page.locator('#model-button').click();
    assert.equal(await page.locator('#memory-read-chars').inputValue(), '24000');
    assert.equal(await page.locator('#handoff-chars').inputValue(), '16000');
    await page.locator('#settings-form .close-dialog').click();
    await page.setViewportSize({width: 390, height: 844});
    await page.reload();
    await page.locator('#test-dataset-banner').waitFor();
    await page.locator('#flow-run-status').filter({hasText: /^已完成$/}).waitFor();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.locator('.node-recall').click();
    await page.screenshot({path: path.join(artifacts, 'scale-flow-mobile.png')});
    assert.deepEqual(errors, []);
    await fs.writeFile(path.join(artifacts, 'browser-validation.json'), JSON.stringify({synthetic: true, desktop: 'passed', mobile: 'passed', errors,
      state_response_bytes: state.utf8_bytes, state_response_ms: state.response_ms,
      note: 'Read-only UI validation; scripted trace, no real model. /api/state still sends the full memory list to the management UI, not to the model.'}, null, 2));
    await page.goto(pathToFileURL(path.join(artifacts, 'report.html')).href);
    assert.equal(await page.locator('tbody tr').count(), 4);
    await page.screenshot({path: path.join(artifacts, 'report-mobile.png')});
    console.log('PASS: isolated 500-record lab, keyword search, visible capacity/query traces, settings, desktop/mobile, offline report. No real model called.');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
