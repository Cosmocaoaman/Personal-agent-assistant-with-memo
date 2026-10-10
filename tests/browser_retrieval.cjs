/* Run after browser.cjs against the disposable UI fixture. Vectors are MOCK. */
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({headless:true, executablePath:process.env.MORI_TEST_BROWSER || 'C:/Program Files/Google/Chrome/Application/chrome.exe'});
  const page = await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[]; page.on('pageerror', e=>errors.push(e.message));
  try {
    await page.goto('http://127.0.0.1:8766');
    await page.locator('#key-indicator').filter({hasText:'已配置'}).waitFor();
    await page.locator('#settings-button').click();
    assert.equal(await page.locator('#embedding-model').inputValue(),'text-embedding-3-small');
    assert.equal(await page.locator('#embedding-key').isDisabled(),true);
    assert.equal(await page.locator('#embedding-base').inputValue(),'http://127.0.0.1:9876');
    await page.locator('details').filter({has:page.locator('#embedding-dimensions')}).locator('summary').click();
    await page.locator('#embedding-dimensions').fill('8');
    await page.locator('#embedding-min-similarity').fill('0');
    await page.locator('#embedding-enabled').check();
    await page.locator('#embedding-probe').click();
    await page.locator('#settings-feedback').filter({hasText:'Embedding 连接成功'}).waitFor();
    assert.match(await page.locator('#settings-feedback').textContent(),/8 维/);
    await page.locator('#embedding-status').filter({hasText:/语义索引 \d+ \/ \d+ 条/}).waitFor();
    await page.locator('#embedding-index').click();
    await page.waitForFunction(() => !document.querySelector('#embedding-index').disabled);
    await page.locator('#settings-dialog').evaluate(el=>el.scrollTop=document.querySelector('#embedding-enabled').offsetTop-30);
    await page.screenshot({path:'artifacts/embedding-settings.png'});
    await page.locator('#save-settings').click();
    await page.locator('#settings-dialog').waitFor({state:'hidden'});
    assert.match(await page.locator('#model-settings-summary').textContent(),/Embedding text-embedding-3-small/);
    await page.locator('[data-view=memory]').click();
    await page.locator('.retrieval-panel summary').click();
    await page.locator('#retrieval-query').fill('简洁中文');
    await page.locator('#retrieval-test').click();
    await page.locator('#retrieval-feedback').filter({hasText:'混合召回'}).waitFor();
    assert.ok(await page.locator('.retrieval-result').count()>0);
    await page.locator('#retrieval-test').click();
    await page.locator('#retrieval-feedback').filter({hasText:'查询缓存命中'}).waitFor();
    await page.screenshot({path:'artifacts/hybrid-retrieval.png',fullPage:true});

    // External router data remains available on the daily route. Model is MOCK.
    const run = await page.evaluate(async()=>{
      const state=await (await fetch('/api/state')).json();
      const headers={'Content-Type':'application/json','X-Mori-Token':state.csrf};
      const session=await (await fetch('/api/sessions',{method:'POST',headers,body:'{}'})).json();
      const response=await fetch('/api/chat',{method:'POST',headers,body:JSON.stringify({session_id:session.id,message:'模拟路由上下文测试',router_context:{route:'daily',context:{home:{source:'router-fixture',note:'虚构资料'}}}})});
      const rows=(await response.text()).trim().split('\n').map(line=>JSON.parse(line));
      return rows.find(row=>row.event==='done').message.meta.run_id;
    });
    await page.locator('[data-view=flow]').click();
    await page.locator('#flow-run-status').filter({hasText:'已完成'}).waitFor();
    await page.locator('#flow-run-select').selectOption(run);
    await page.locator('.node-recall').click();
    assert.match(await page.locator('#flow-inspector').textContent(),/日常问答.*router-fixture/s);
    await page.screenshot({path:'artifacts/router-context-flow.png',fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    await page.locator('#model-button').click();
    await page.locator('#settings-dialog').evaluate(el=>el.scrollTop=document.querySelector('#embedding-enabled').offsetTop-30);
    assert.equal(await page.locator('#embedding-enabled').isChecked(),true);
    assert.equal(await page.locator('#embedding-key').inputValue(),'');
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    await page.screenshot({path:'artifacts/embedding-settings-mobile.png'});
    assert.deepEqual(errors,[]);
    console.log('PASS: MOCK Embedding settings/probe/index, reused key, hybrid recall/cache, daily router handoff/flow, desktop/mobile.');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
