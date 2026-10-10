'use strict';
window.FlowView = (() => {
  const $ = selector => document.querySelector(selector);
  const names = {prepare: '接收消息', recall: '整理上下文', answer: '生成回答', consolidate: '判断与维护记忆', finish: '汇总结果'};
  const roles = {prepare: '系统', recall: '记忆子 Agent', answer: '主 Agent', consolidate: '记忆子 Agent', finish: '系统'};
  const hints = {prepare: '读取近期对话、准备最新的基础个人上下文，并保存当前消息。', recall: '记忆子 Agent 接收基础上下文与预取候选，尽量一次筛选交接，必要时补查。', answer: '主 Agent 流式回答；按需检索模式下可自行只读搜索记忆。', consolidate: '与主 Agent 并行，自主决定新增、更新、删除或不改动记忆。', finish: '等待两个分支完成，保存各自的结果与用量。'};
  const nodeLabels = {waiting: '等待执行', running: '执行中', succeeded: '已完成', failed: '未完成', interrupted: '已中断', cancelled: '未执行'};
  const runLabels = {running: '执行中', completed: '已完成', partial: '部分异常', failed: '执行失败', interrupted: '已中断'};
  const tools = {memory_list: '浏览记忆', memory_search: '搜索记忆', memory_context: '交接上下文', memory_decide: '提交记忆判断', web_search: '联网搜索'};
  const outcomeLabels = {changed: '已维护记忆', skipped: '决定不作改动', readonly: '本轮只读', error: '维护未完成'};
  let api, openChat, active = false, polling = null, ticking = null, playback = null;
  let pending = false, generation = 0, runList = [], selectedId = null, data = null, following = true;
  let selectedNode = 'recall', cursor = null, selectorSignature = '';

  function el(tag, className, value) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value !== undefined) node.textContent = value;
    return node;
  }
  function duration(ms) {
    if (ms === null || ms === undefined) return '—';
    if (ms < 1000) return `${Math.max(0, Math.round(ms))} ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(2)} s`;
    return `${Math.floor(ms / 60000)}分 ${Math.floor(ms / 1000) % 60}秒`;
  }
  function instant(value) {return new Date(value).toLocaleString('zh-CN', {month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit'});}
  function stopPlayback() {clearInterval(playback); playback = null; $('#flow-play').textContent = '▶ 逐步回放';}
  function notice(text, error = false) {$('#flow-notice').textContent = text; $('#flow-notice').classList.toggle('error', error);}
  function initialNodes() {
    return Object.fromEntries(Object.keys(names).map(id => [id, {id, label: names[id], status: 'waiting', started_ms: null, ended_ms: null, duration_ms: null, tokens: 0, reasoning_tokens: null, model_calls: 0, tools: 0, detail: {}}]));
  }
  function displayState() {
    if (!data) return {nodes: initialNodes(), elapsed: 0, events: [], status: null};
    if (cursor === null) {
      const elapsed = data.status === 'running' ? Math.max(data.elapsed_ms, Date.now() - Date.parse(data.started_at)) : data.elapsed_ms;
      return {nodes: data.nodes, elapsed, events: data.events, status: data.status};
    }
    const nodes = initialNodes();
    const events = data.events.slice(0, cursor);
    let status = 'running';
    for (const event of events) {
      if (event.node && event.patch) Object.assign(nodes[event.node], event.patch);
      if (event.kind === 'run_end') status = event.status;
    }
    return {nodes, elapsed: events.at(-1)?.elapsed_ms || 0, events, status};
  }
  function renderSelector() {
    const signature = JSON.stringify([selectedId, runList.map(r => [r.id, r.status])]);
    if (selectorSignature === signature) return;
    selectorSignature = signature;
    const select = $('#flow-run-select'); select.replaceChildren();
    if (!runList.length) select.append(el('option', '', '尚无运行记录'));
    for (const run of runList) {
      const option = el('option', '', `${instant(run.started_at)} · ${run.preview} · ${runLabels[run.status] || run.status}`);
      option.value = run.id;
      select.append(option);
    }
    if (selectedId) select.value = selectedId;
    select.disabled = !runList.length;
  }
  function adopt(trace) {
    const row = {id: trace.id, session_id: trace.session_id, status: trace.status, preview: trace.preview, started_at: trace.started_at, elapsed_ms: trace.elapsed_ms};
    const index = runList.findIndex(r => r.id === row.id);
    if (index >= 0) runList[index] = row;
    else runList.unshift(row);
    if (!selectedId || (following && row.id === runList[0]?.id)) selectedId = row.id;
    if (selectedId !== trace.id) return;
    if (data?.id === trace.id && data.events.length > trace.events.length) return;
    if (data?.id !== trace.id) {cursor = null; stopPlayback();}
    data = trace;
    if (active) render();
  }
  async function refresh() {
    if (pending) return;
    pending = true;
    const request = ++generation;
    try {
      const list = await api('/api/runs');
      if (request !== generation) return;
      $('#flow-nav-dot').hidden = !list.some(run => run.status === 'running');
      // Preserve an explicitly opened older run outside the latest-50 menu.
      if (!following && data && !list.some(r => r.id === data.id)) list.push({id: data.id, session_id: data.session_id, preview: data.preview, status: data.status, started_at: data.started_at});
      runList = list;
      if (following || !selectedId) selectedId = list[0]?.id || null;
      if (!selectedId) {data = null; render(); return;}
      const wantedId = selectedId;
      const trace = await api(`/api/runs/detail?id=${encodeURIComponent(wantedId)}`);
      if (request !== generation || wantedId !== selectedId) return;
      adopt(trace);
    } catch (error) {
      if (request === generation) notice(error.message || '暂时无法加载流程记录', true);
    } finally {pending = false;}
  }
  function metrics(view) {
    const nodes = Object.values(view.nodes);
    $('#flow-run-status').textContent = data ? (cursor !== null ? `回放 · ${runLabels[view.status]}` : runLabels[view.status]) : '结构预览';
    $('#flow-duration').textContent = data ? duration(view.elapsed) : '—';
    $('#flow-counts').textContent = data ? `${nodes.reduce((n, x) => n + x.model_calls, 0)} / ${nodes.reduce((n, x) => n + x.tools, 0)}` : '—';
    $('#flow-tokens').textContent = data ? `${view.nodes.answer.tokens.toLocaleString()} / ${(view.nodes.recall.tokens + view.nodes.consolidate.tokens).toLocaleString()}` : '—';
    $('#flow-mode').textContent = !data ? '流程结构' : cursor !== null ? '历史步骤' : data.status === 'running' ? '实时运行' : '执行记录';
    $('#flow-follow').classList.toggle('flow-following', following && cursor === null);
    $('#flow-follow').setAttribute('aria-pressed', String(following && cursor === null));
    $('#flow-chat').disabled = !data;
  }
  function renderNodes(view) {
    const main = data?.configuration?.runtime?.recall_mode === 'main';
    for (const node of document.querySelectorAll('.flow-node')) {
      const item = view.nodes[node.dataset.node];
      if (item.id === 'recall') {
        node.querySelector('strong').textContent = main ? '载入基础上下文' : '整理与交接上下文';
        node.querySelector('.flow-node-eyebrow').textContent = main ? '02 · 系统 · 无模型调用' : '02 · 记忆子 Agent';
      }
      node.className = `flow-node node-${item.id} ${data ? item.status : 'preview'}${selectedNode === item.id ? ' selected' : ''}`;
      node.setAttribute('aria-pressed', String(selectedNode === item.id));
      let label = data ? nodeLabels[item.status] : '未运行';
      if (item.id === 'consolidate' && item.status === 'succeeded') label = outcomeLabels[item.detail.outcome] || label;
      node.querySelector('.flow-node-status').textContent = label;
      const ms = item.status === 'running' ? Math.max(0, view.elapsed - item.started_ms) : item.duration_ms;
      node.querySelector('.flow-node-time').textContent = duration(ms);
    }
    for (const path of document.querySelectorAll('[data-edge]')) {
      const [a, b] = path.dataset.edge.split('-');
      const before = view.nodes[a], after = view.nodes[b];
      const reached = before.ended_ms !== null && after.started_ms !== null;
      path.classList.toggle('active', reached && after.status === 'running');
      path.classList.toggle('done', reached && after.status !== 'running');
    }
  }
  function detailSection(parent, title, text) {
    const section = el('div', 'flow-detail-section');
    section.append(el('h3', '', title), el('p', '', text));
    parent.append(section);
  }
  function renderInspector(view) {
    const item = view.nodes[selectedNode];
    const box = $('#flow-inspector'); box.replaceChildren();
    const main = data?.configuration?.runtime?.recall_mode === 'main';
    box.append(el('span', 'tiny-label', main && selectedNode === 'recall' ? '系统' : roles[selectedNode]), el('h2', '', main && selectedNode === 'recall' ? '载入基础上下文' : names[selectedNode]), el('p', '', main && selectedNode === 'recall' ? '直接载入基础个人上下文与外部路由资料，无模型调用。是否额外检索由主 Agent 在回答时自主决定。' : hints[selectedNode]));
    box.append(el('span', `flow-detail-status ${item.status}`, data ? nodeLabels[item.status] : '尚未运行'));
    if (!data) {detailSection(box, '查看真实执行', '发送消息后会记录节点耗时、工具动作和完成状态。点击图中的节点，可以查看该阶段的结果。'); return;}
    const grid = el('div', 'flow-details-grid');
    const values = [['开始于', duration(item.started_ms)], ['节点耗时', duration(item.status === 'running' ? view.elapsed - item.started_ms : item.duration_ms)], ['模型调用', `${item.model_calls} 次`], ['Token 用量', item.tokens.toLocaleString()]];
    for (const [label, value] of values) {const cell = el('div'); cell.append(el('span', '', label), el('strong', '', value)); grid.append(cell);}
    box.append(grid);
    if (item.first_output_ms !== undefined) detailSection(box, '首次输出', `本轮开始后 ${duration(item.first_output_ms)} 收到该节点的首个文本或工具参数片段。`);
    if (item.requests?.length) {
      const section = el('details', 'flow-detail-section request-timings'); section.open = true;
      section.append(el('summary', '', `模型请求耗时 · ${item.requests.length} 次`));
      item.requests.forEach((request, index) => {
        const connection = request.connection_reused === true ? '复用连接' : request.connection_reused === false ? '新建连接' : '连接信息未提供';
        section.append(el('h3', '', `请求 ${index + 1} · ${request.transport} · ${connection} · ${request.streaming ? '流式' : '整包返回'}${request.ok ? '' : ' · 未完成'}`));
        section.append(el('p', '', `连接建立 ${duration(request.connection_ms)}；响应头 ${duration(request.headers_ms)}；首输出 ${duration(request.first_output_ms)}；首输出后 ${duration(request.output_ms)}；总计 ${duration(request.total_ms)}。`));
      });
      section.append(el('p', 'muted', '响应头和首输出均从请求发起计时。首输出等待包含服务端处理与网络等待，首输出后包含后续生成与传输；无法精确拆出纯模型计算。'));
      box.append(section);
    }
    if (['answer', 'recall', 'consolidate'].includes(selectedNode)) {
      detailSection(box, '接口已报告的思考 Token', item.reasoning_tokens == null ? '接口未提供此项统计，无法得知实际大小。' : `${item.reasoning_tokens.toLocaleString()} tokens（已计入总用量）`);
      const options = data.configuration?.model_options;
      if (options) {
        const params = el('details', 'flow-detail-section'); params.append(el('summary', '', '本轮模型配置'));
        const entries = [['模型', data.model], ['思考强度', options.reasoning_effort], ['生成 Token 上限', options.max_output_tokens], ['请求超时', `${options.timeout_seconds} 秒`], ['Temperature', options.temperature ?? '服务商默认'], ['Top-P', options.top_p ?? '服务商默认'], ['回答详略', selectedNode === 'answer' ? options.verbosity : '服务商默认'], ['网页搜索', selectedNode === 'answer' && options.web_search ? '按需搜索' : '关闭']];
        const embedding = data.configuration?.embedding;
        if (embedding && (['recall','consolidate'].includes(selectedNode) || main)) entries.push(['语义检索', embedding.enabled ? `${embedding.model} · ${embedding.dimensions || '默认'} 维 · API 超时 ${embedding.timeout_seconds} 秒` : '未启用，使用 FTS5 关键词']);
        for (const [key, value] of entries) params.append(el('p', '', `${key}：${value}`));
        box.append(params);
      }
    }
    const detail = item.detail;
    if (selectedNode === 'recall' && !main && detail.summary_chars !== undefined) detailSection(box, '简短交接备注', `${detail.summary_chars} / ${data.configuration.runtime.handoff_summary_chars} 字符；原始事实按 ID 传递。`);
    if (selectedNode === 'answer' && main && item.status === 'succeeded') detailSection(box, '额外记忆检索', detail.extra_memory ? '主 Agent 按需检索后直接回答，未经过子 Agent 整理交接。' : '主 Agent 选择直接回答；本轮只使用基础个人上下文、近期对话与路由资料。');
    if (detail.error) detailSection(box, '本阶段异常', detail.error);
    if (detail.read_chars !== undefined) {
      const budget = data.configuration?.runtime?.memory_read_chars;
      detailSection(box, '读入记忆容量', `${detail.read_chars.toLocaleString()}${budget ? ` / ${budget.toLocaleString()}` : ''} JSON 字符，包含本阶段预载与工具返回的完整记忆条目。`);
    }
    if (detail.handoff_chars !== undefined) detailSection(box, '交接原始条目', `${detail.handoff_chars.toLocaleString()} JSON 字符。${detail.handoff_omitted ? `容量限制另省略 ${detail.handoff_omitted} 条；摘要仍独立提供。` : '条目均完整保留。'}`);
    if (detail.retrievals?.length) {
      const section = el('details', 'flow-detail-section'); section.open = detail.retrievals.length <= 4;
      section.append(el('summary', '', `查询与浏览记录 · ${detail.retrievals.length} 次`));
      for (const read of detail.retrievals) {
        section.append(el('p', '', `${read.name === 'memory_prefetch' ? `系统预取「${read.query}」` : read.name === 'memory_search' ? `搜索「${read.query}」` : `浏览偏移 ${read.offset}`} · ${read.returned_count} 条 · ${read.lookup_ms} ms · 剩余 ${read.budget_remaining_chars.toLocaleString()} 字符${read.omitted_due_to_budget ? ` · 容量省略 ${read.omitted_due_to_budget} 条` : ''}${read.error ? ` · ${read.error}` : ''}`));
        const r = read.retrieval;
        if (r) section.append(el('p', 'muted', `${r.mode === 'hybrid' ? '混合召回 · RRF' : '关键词 · FTS5'}：关键词 ${r.lexical_count} 条，语义 ${r.semantic_count} 条${r.embedding_ms !== undefined ? `；Embedding ${r.embedding_ms} ms${r.query_cache_hit ? '（缓存命中）' : ''}` : ''}${r.indexed !== undefined ? `；语义索引 ${r.indexed}/${r.total}` : ''}${r.fallback ? `；${r.fallback}` : ''}`));
      }
      box.append(section);
    }
    if (selectedNode === 'prepare' && item.status === 'succeeded') detailSection(box, '本轮输入', `${detail.input_chars} 字符，连同 ${detail.message_count - 1} 条近期消息进入流程。`);
    if (detail.router_context) detailSection(box, `路由 · ${detail.router_context.route === 'home' ? '家居问答' : '日常问答'}`, Object.keys(detail.router_context.context).length ? JSON.stringify(detail.router_context.context, null, 2) : '没有额外路由资料。所有领域的长期记忆均可按需召回。');
    if (detail.personal_count !== undefined) {
      detailSection(box, `基础上下文 · ${detail.personal_count} 条`, `关于我、偏好习惯与置顶记忆默认载入。${detail.personal_omitted ? `另有 ${detail.personal_omitted} 条未预载，可按需查询。` : ''}`);
      const memories = detail.personal_memories || [];
      if (memories.length > 6) {
        const section = el('details', 'flow-detail-section'); section.append(el('summary', '', `查看 ${memories.length} 条预载事实`));
        for (const memory of memories) section.append(el('div', 'flow-detail-change', memory.content));
        box.append(section);
      } else {
        for (const memory of memories) box.append(el('div', 'flow-detail-change', memory.content));
      }
    }
    if (selectedNode === 'recall' && detail.summary !== undefined) detailSection(box, `交接上下文 · ${detail.memory_count} 条记忆`, detail.summary || (detail.memory_count ? '记忆条目已交接，没有额外摘要。' : '没有需要交接的长期记忆。'));
    if (selectedNode === 'answer' && item.status === 'succeeded') detailSection(box, '回答已返回', `本轮回答共 ${detail.output_chars} 字符，已独立发送至对话界面。`);
    if (selectedNode === 'consolidate' && detail.outcome) {
      detailSection(box, outcomeLabels[detail.outcome], detail.reason || '没有额外说明。');
      for (const change of detail.changes || []) box.append(el('div', 'flow-detail-change', `${{create: '新增', update: '更新', delete: '删除'}[change.operation]} · ${change.key}`));
    }
    if (selectedNode === 'finish' && item.status === 'succeeded') detailSection(box, '本轮完成', `共使用 ${detail.total_tokens} tokens。记忆结果：${outcomeLabels[detail.memory_status] || detail.memory_status}。`);
    if (item.status === 'running') detailSection(box, '正在执行', selectedNode === 'answer' ? '主 Agent 正在生成回答，记忆分支独立运行。' : '等待该节点返回，结果会自动更新。');
    if (item.status === 'interrupted') detailSection(box, '记录说明', '服务重启前未收到该节点完成事件，无法得知实际结束时间。');
    const activity = view.events.filter(e => e.node === selectedNode && e.kind === 'tool');
    if (activity.length) {
      const section = el(activity.length > 6 ? 'details' : 'div', 'flow-detail-section'); section.append(el(activity.length > 6 ? 'summary' : 'h3', '', `工具动作 · ${activity.length} 次`));
      const list = el('ul', 'flow-mini-tools');
      for (const event of activity) {const row = el('li'); row.append(el('span', '', `${event.ok ? '✓' : '×'} ${tools[event.name] || event.name}`), el('time', '', duration(event.elapsed_ms))); list.append(row);}
      section.append(list); box.append(section);
    }
  }
  function renderTimeline(view) {
    const events = data?.events || [];
    const progress = cursor === null ? events.length : cursor;
    const range = $('#flow-scrubber'); range.max = events.length; range.value = progress; range.disabled = !events.length;
    $('#flow-play').disabled = !events.length;
    $('#flow-latest').disabled = !events.length || cursor === null;
    $('#flow-replay-label').textContent = events.length ? `第 ${progress} / ${events.length} 步 · ${duration(view.elapsed)}` : '等待第一条记录';
    const box = $('#flow-timeline'); const scroll = box.scrollTop;
    box.replaceChildren();
    if (!events.length) box.append(el('p', 'flow-event-placeholder', '模型调用、工具执行和节点完成时，会在这里留下时间记录。'));
    for (const event of events) {
      const node = el('button', `flow-event${cursor === event.seq ? ' selected' : ''}${event.seq > progress ? ' future' : ''}${event.ok === false || event.patch?.status === 'failed' ? ' failed' : ''}`);
      node.type = 'button';
      let message = event.message;
      if (event.kind === 'tool') message = `${tools[event.name] || event.name} · ${event.ok ? '成功' : '未成功'}`;
      const role = event.node === 'recall' && data.configuration?.runtime?.recall_mode === 'main' ? '系统' : roles[event.node] || '系统';
      node.append(el('time', '', duration(event.elapsed_ms)), el('span', 'flow-event-agent', role), el('span', 'flow-event-message', message));
      node.addEventListener('click', () => {stopPlayback(); cursor = event.seq; following = false; if (event.node) selectedNode = event.node; render();});
      box.append(node);
    }
    box.scrollTop = scroll;
  }
  function render(tick = false) {
    if (!active) return;
    const view = displayState();
    metrics(view); renderNodes(view);
    if (tick) return;
    renderSelector(); renderInspector(view); renderTimeline(view);
    const config = $('#flow-config-summary');
    const options = data?.configuration?.model_options;
    config.hidden = !options;
    if (options) config.textContent = `${data.model} · 思考 ${options.reasoning_effort} · 生成上限 ${options.max_output_tokens.toLocaleString()} tokens · 超时 ${options.timeout_seconds} 秒 · ${data.configuration?.runtime?.recall_mode === 'main' ? '主 Agent 按需检索' : '子 Agent 筛选交接'} · 主 Agent 联网 ${options.web_search ? '按需搜索' : '关闭'}${data.configuration?.embedding?.enabled ? ` · Embedding ${data.configuration.embedding.model}` : ''}`;
    if (!data) notice('结构预览 · 发送一条消息后显示真实执行过程。旧对话没有流程记录。');
    else if (cursor !== null) notice(`逐步回看这次运行 · ${data.preview}`);
    else if (data.status === 'running') notice(`实时跟踪 · ${data.preview}`);
    else if (data.status === 'interrupted') notice(`这次运行未正常结束；显示的是中断前的末次记录。`, true);
    else notice(`${instant(data.started_at)} · ${data.model} · ${data.preview}`);
  }
  function init(options) {
    api = options.api; openChat = options.openChat;
    for (const node of document.querySelectorAll('.flow-node')) node.addEventListener('click', () => {selectedNode = node.dataset.node; render();});
    $('#flow-run-select').addEventListener('change', async event => {await select(event.target.value);});
    $('#flow-follow').addEventListener('click', follow);
    $('#flow-latest').addEventListener('click', () => {cursor = null; stopPlayback(); render();});
    $('#flow-chat').addEventListener('click', () => {if (data) openChat(data.session_id);});
    $('#flow-scrubber').addEventListener('input', event => {stopPlayback(); following = false; cursor = Number(event.target.value); render();});
    $('#flow-play').addEventListener('click', () => {
      if (playback) {stopPlayback(); return;}
      following = false;
      if (cursor === null || cursor >= data.events.length) cursor = 0;
      $('#flow-play').textContent = 'Ⅱ 暂停回放';
      render();
      playback = setInterval(() => {cursor = Math.min(cursor + 1, data.events.length); render(); if (cursor === data.events.length) stopPlayback();}, 450);
    });
  }
  async function open() {
    active = true; render();
    clearInterval(polling); clearInterval(ticking);
    polling = setInterval(() => {if (following || data?.status === 'running') refresh();}, 1500);
    ticking = setInterval(() => {if (data?.status === 'running' && cursor === null) render(true);}, 250);
    await refresh();
  }
  function close() {active = false; clearInterval(polling); clearInterval(ticking); stopPlayback();}
  async function follow() {following = true; cursor = null; stopPlayback(); await refresh(); render();}
  function forgetSession(sessionId) {
    generation++;
    runList = runList.filter(run => run.session_id !== sessionId);
    if (data?.session_id === sessionId) {
      data = null; selectedId = null; cursor = null; following = true; stopPlayback();
    }
    render();
    if (active) refresh();
  }
  async function select(id) {
    following = false; selectedId = id; cursor = null; stopPlayback();
    generation++;
    try {const trace = await api(`/api/runs/detail?id=${encodeURIComponent(id)}`); if (selectedId === id) adopt(trace);}
    catch (error) {notice(error.message, true);}
  }
  function receive(trace) {
    $('#flow-nav-dot').hidden = trace.status !== 'running';
    adopt(trace);
  }
  return {init, open, close, refresh, select, receive, follow, forgetSession};
})();
