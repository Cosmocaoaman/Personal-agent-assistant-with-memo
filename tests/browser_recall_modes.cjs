/* Isolated ui_fixture.py; all API responses are explicitly scripted. */
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.MORI_TEST_BROWSER || 'C:/Program Files/Google/Chrome/Application/chrome.exe'});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  const artifacts = path.resolve('artifacts/recall-v2.8'); await fs.mkdir(artifacts, {recursive: true});
  try {
    await page.goto('http://127.0.0.1:8766/');
    await page.evaluate(async () => {
      const state = await (await fetch('/api/state')).json();
      async function post(url, body) {
        const response = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Mori-Token': state.csrf}, body: JSON.stringify(body)});
        if (!response.ok) throw new Error(await response.text());
      }
      await post('/api/settings', {provider: 'openai', model: 'recall-modes-fixture', base_url: 'http://127.0.0.1:9876', api_key: 'fixture-only', options: {reasoning_effort: 'none', web_search: false}, runtime: {recall_mode: 'specialist'}});
      await post('/api/memories/save', {content: 'Aurora 使用 PostgreSQL 和 Redis。', category: 'project', key: 'ui.aurora'});
    });
    await page.reload();
    for (const viewport of [{width: 1440, height: 1000}, {width: 390, height: 844}]) {
      await page.setViewportSize(viewport);
      for (const mode of ['specialist', 'main']) {
        await page.locator('#model-button').click();
        await page.locator('#recall-mode').selectOption(mode);
        await page.locator('#handoff-summary-chars').scrollIntoViewIfNeeded();
        assert.equal(await page.locator('#prefetch-candidates').isDisabled(), mode === 'main');
        assert.equal(await page.locator('#handoff-summary-chars').isDisabled(), mode === 'main');
        assert.match(await page.locator('#recall-mode-note').textContent(), mode === 'main' ? /自主选择直接回答/ : /一次筛选交接/);
        await page.screenshot({path: path.join(artifacts, `settings-${mode}-${viewport.width}.png`)});
        await page.locator('#save-settings').click();
        await page.locator('#settings-dialog').waitFor({state: 'hidden'});
        const stored = await page.evaluate(async () => (await (await fetch('/api/state')).json()).settings.runtime);
        assert.equal(stored.recall_mode, mode);
        for (const retrieval of mode === 'main' ? [false, true] : [true]) {
          if (viewport.width < 760) await page.locator('#menu-button').click();
          await page.locator('#new-chat').click();
          await page.locator('#chat-input').fill(retrieval ? '测试按需检索：Aurora 用什么数据库？' : '解释列表与元组的区别。');
          await page.locator('#send-button').click();
          await page.waitForFunction(() => !document.querySelector('#send-button').disabled, {timeout: 25000});
          assert.equal(await page.locator('.message.assistant').count(), 1);
          const body = await page.locator('.message.assistant .message-body').textContent();
          assert.doesNotMatch(body, /模拟临时说明/);
          if (retrieval && mode === 'main') assert.match(body, /PostgreSQL/);
          const report = await page.evaluate(async () => {
            const runs = await (await fetch('/api/runs')).json();
            return (await fetch(`/api/runs/detail?id=${runs[0].id}`)).json();
          });
          assert.equal(report.nodes.recall.model_calls, mode === 'main' ? 0 : 1);
          assert.equal(report.nodes.answer.model_calls, mode === 'main' && retrieval ? 2 : 1);
          assert.equal(report.nodes.consolidate.model_calls, 1);
          assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
          await page.locator('.message.assistant .message-meta button').filter({hasText: '查看流程'}).click();
          await page.locator('.node-recall').click();
          assert.match(await page.locator('#flow-inspector').textContent(), mode === 'main' ? /无模型调用/ : /简短交接备注/);
          await page.locator('.node-answer').click();
          if (mode === 'main') assert.match(await page.locator('#flow-inspector').textContent(), retrieval ? /按需检索后直接回答/ : /本轮只使用基础个人上下文/);
          await page.screenshot({path: path.join(artifacts, `flow-${mode}-${retrieval ? 'search' : 'direct'}-${viewport.width}.png`), fullPage: true});
          await page.locator('#flow-inspector').scrollIntoViewIfNeeded();
          await page.locator('#flow-inspector').screenshot({path: path.join(artifacts, `inspector-${mode}-${retrieval ? 'search' : 'direct'}-${viewport.width}.png`)});
          await fs.writeFile(path.join(artifacts, `ui-${mode}-${retrieval ? 'search' : 'direct'}-${viewport.width}.json`), JSON.stringify(report, null, 2));
          await page.locator('#flow-chat').click();
        }
      }
    }
    assert.deepEqual(errors, []);
    console.log('PASS: Desktop/mobile saved recall modes, one-pass prefetch handoff, zero-model baseline, main direct/search paths, streamed tool-preamble reset, parallel maintenance and flow inspection (MOCK).');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
