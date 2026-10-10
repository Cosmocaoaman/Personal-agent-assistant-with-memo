/* Use the isolated ui_fixture.py server; no real API calls. */
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.MORI_TEST_BROWSER || 'C:/Program Files/Google/Chrome/Application/chrome.exe'});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  const artifacts = path.resolve('artifacts/streaming-v2.7'); await fs.mkdir(artifacts, {recursive: true});
  try {
    await page.goto('http://127.0.0.1:8766/');
    await page.evaluate(async () => {
      const state = await (await fetch('/api/state')).json();
      const result = await fetch('/api/settings', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Mori-Token': state.csrf},
        body: JSON.stringify({provider: 'openai', model: 'stream-fixture', base_url: 'http://127.0.0.1:9876', api_key: 'test-only-not-real', options: {reasoning_effort: 'none'}})});
      if (!result.ok) throw new Error(await result.text());
    });
    await page.reload();
    for (const viewport of [{width: 1440, height: 1000}, {width: 390, height: 844}]) {
      await page.setViewportSize(viewport);
      if (viewport.width < 760) await page.locator('#menu-button').click();
      await page.locator('#new-chat').click();
      await page.locator('#chat-input').fill('流式测试：回答几个字即可');
      await page.locator('#send-button').click();
      await page.waitForFunction(() => document.querySelector('.message.streaming .message-body')?.textContent.length > 0);
      assert.match(await page.locator('.message.streaming .message-body').textContent(), /测试模拟/);
      assert.equal(await page.locator('#send-button').isDisabled(), true);
      assert.equal(await page.locator('.message.assistant:not(.streaming)').count(), 0);
      await page.screenshot({path: path.join(artifacts, `streaming-${viewport.width}.png`), fullPage: true});
      await page.waitForFunction(() => !document.querySelector('#send-button').disabled, {timeout: 20000});
      assert.equal(await page.locator('.message.streaming').count(), 0);
      assert.equal(await page.locator('.message.assistant').count(), 1);
      const report = await page.evaluate(async () => {
        const runs = await (await fetch('/api/runs')).json();
        return (await fetch(`/api/runs/detail?id=${runs[0].id}`)).json();
      });
      const requests = Object.values(report.nodes).flatMap(n => n.requests || []);
      assert.equal(requests.length >= 3, true);
      assert.equal(requests.every(r => r.streaming), true);
      assert.equal(requests.some(r => r.connection_reused), true);
      assert.equal(report.nodes.answer.requests[0].first_output_ms < report.nodes.answer.requests[0].total_ms, true);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      await page.locator('.message.assistant .message-meta button').filter({hasText: '查看流程'}).click();
      await page.locator('.node-answer').click();
      await page.locator('#flow-inspector .request-timings').waitFor();
      assert.match(await page.locator('#flow-inspector').textContent(), /首输出后.*总计/);
      await page.locator('#flow-inspector .request-timings').scrollIntoViewIfNeeded();
      await page.locator('#flow-inspector .request-timings').screenshot({path: path.join(artifacts, `request-details-${viewport.width}.png`)});
      await page.screenshot({path: path.join(artifacts, `timings-${viewport.width}.png`), fullPage: true});
      await fs.writeFile(path.join(artifacts, `ui-trace-${viewport.width}.json`), JSON.stringify(report, null, 2));
      await page.locator('#flow-chat').click();
    }
    assert.deepEqual(errors, []);
    console.log('Streaming UI verified: real SSE from mock, desktop/mobile partial output, completed replacement, connection reuse, and timing inspector.');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
