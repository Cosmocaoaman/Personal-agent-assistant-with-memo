'use strict';
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = {settings: null, csrf: '', sessions: [], memories: [], tasks: [], active: null, messages: [], view: 'chat', category: 'all', busy: false};
const categoryNames = {profile: '关于我', preference: '偏好习惯', project: '长期项目', note: '其他笔记'};
const toolNames = {memory_search: '查询记忆', memory_list: '浏览记忆', memory_context: '交接上下文', memory_decide: '提交记忆判断', memory_save: '保存记忆', web_search: '联网搜索'};
const effortNames = {auto: '服务商默认', none: '关闭', minimal: '极低', low: '低', medium: '中', high: '高', xhigh: '更高', max: '最大'};
const runtimeFields = {history_messages: 'history-messages', history_chars: 'history-chars', personal_items: 'personal-items', personal_chars: 'personal-chars', memory_steps: 'memory-steps', tool_calls_per_step: 'tool-calls-per-step', memory_read_chars: 'memory-read-chars', handoff_chars: 'handoff-chars', handoff_summary_chars: 'handoff-summary-chars'};
const fallbackRuntime = {history_messages: 30, history_chars: 24000, personal_items: 24, personal_chars: 12000, memory_steps: 6, tool_calls_per_step: 4, memory_read_chars: 24000, handoff_chars: 16000, recall_mode: 'specialist', prefetch_candidates: true, handoff_summary_chars: 240};
function optionsFor(config, provider) {return {reasoning_effort: provider === 'deepseek' ? 'none' : 'auto', max_output_tokens: 4096, timeout_seconds: 90, temperature: null, top_p: null, verbosity: 'auto', web_search: false, search_context_size: 'medium', ...config.options};}
function optionSummary(options, provider) {return `思考 ${effortNames[options.reasoning_effort]} · ${options.max_output_tokens.toLocaleString()} Token 上限 · ${provider === 'deepseek' ? '联网未接入' : options.web_search ? '联网可用' : '联网关闭'}`;}
let toastTimer;
let settingsProvider = 'deepseek';
let settingsDraft = {};
let searchTimer;
let searchGeneration = 0;
let embeddingTimer;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function button(text, className, action) {
  const node = el('button', className, text);
  node.type = 'button';
  node.addEventListener('click', () => safe(action));
  return node;
}
function toast(message, error = false) {
  clearTimeout(toastTimer);
  $('#toast').textContent = message;
  $('#toast').classList.toggle('error', error);
  $('#toast').hidden = false;
  toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 4500);
}
async function safe(action) {
  try { await action(); } catch (error) { toast(error.message || '操作失败，请重试', true); }
}
async function api(route, body) {
  const options = body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Mori-Token': state.csrf}, body: JSON.stringify(body)};
  let response;
  try { response = await fetch(route, options); } catch (_) { throw new Error('无法连接本机服务，请运行 start.cmd 后刷新页面。'); }
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}
async function refresh() {
  const data = await api('/api/state');
  Object.assign(state, data);
  const dataset = state.settings.dataset;
  $('#test-dataset-banner').hidden = !dataset?.synthetic;
  if (dataset?.synthetic) {
    $('#test-dataset-banner').textContent = `虚构记忆测试库 · 当前 ${state.memories.length} 条记忆 · 数据与日常工作空间分开保存`;
    document.title = '木记 Mori · 虚构记忆测试';
  }
  renderSidebar();
  renderModel();
  renderPersonalContext();
  await renderEmbeddingStatus();
  if (state.view === 'memory') await renderMemories();
  if (state.view === 'tasks') renderTasks();
}
function renderModel() {
  const name = state.settings.provider;
  const config = state.settings[name];
  $('#current-provider').textContent = name === 'openai' ? 'OpenAI · GPT' : 'DeepSeek';
  $('#current-model').textContent = config.model;
  const summary = optionSummary(optionsFor(config, name), name);
  $('#model-button').title = `${config.model} · ${summary}`;
  const embedding = state.settings.embedding;
  $('#model-settings-summary').textContent = `${summary} · ${state.settings.runtime?.recall_mode === 'main' ? '主 Agent 按需检索' : '子 Agent 筛选交接'} · ${embedding?.enabled ? `Embedding ${embedding.model}` : 'FTS5 关键词'} · 设置 ↗`;
  $('#key-indicator').textContent = state.settings.keys[name] ? '已配置' : '待配置';
  $('#memory-mode').textContent = state.settings.auto_memory ? '记忆子 Agent 自主维护' : '记忆子 Agent 只读';
}
function renderSidebar() {
  $('#memory-count').textContent = state.memories.length;
  $('#task-count').textContent = state.tasks.filter(t => !t.done).length;
  const list = $('#session-list');
  list.replaceChildren();
  if (!state.sessions.length) list.append(el('p', 'session-empty', '还没有对话。\n从今天的小事开始吧。'));
  for (const session of state.sessions) {
    const row = el('div', `session-row${state.active === session.id ? ' active' : ''}`);
    const open = button(session.title, '', () => loadSession(session.id));
    open.title = session.title;
    const remove = button('×', 'delete-session', async () => {
      if (state.busy) return toast('请等待当前回复结束');
      if (!confirm('删除这个会话、聊天及流程记录？已保存的长期记忆会保留，可在记忆页单独删除。')) return;
      await api('/api/sessions/delete', {id: session.id});
      FlowView.forgetSession(session.id);
      if (state.active === session.id) {state.active = null; state.messages = []; renderMessages();}
      await refresh();
      toast('会话已删除');
    });
    remove.setAttribute('aria-label', `删除会话：${session.title}`);
    row.append(open, remove);
    list.append(row);
  }
}
async function setView(view) {
  state.view = view;
  $$('.view').forEach(node => {node.hidden = node.id !== `view-${view}`;});
  $$('.nav-item').forEach(node => node.classList.toggle('active', node.dataset.view === view));
  $('#page-label').textContent = {chat: '对话空间', memory: '长期记忆', tasks: '我的待办', flow: 'Agent 流程'}[view];
  $('#sidebar').classList.remove('open');
  if (view === 'memory') await renderMemories();
  if (view === 'tasks') renderTasks();
  if (view === 'flow') await FlowView.open();
  else FlowView.close();
}
async function newChat() {
  if (state.busy) return toast('木记正在回复，请稍等');
  await refresh();
  if (state.busy) return toast('木记正在回复，请稍等');
  state.active = null;
  state.messages = [];
  await setView('chat');
  renderMessages();
  renderSidebar();
  $('#chat-input').focus();
}
async function loadSession(id) {
  if (state.busy) return toast('木记正在回复，请稍等');
  const messages = await api(`/api/messages?session=${encodeURIComponent(id)}`);
  state.active = id;
  state.messages = messages;
  await setView('chat');
  renderMessages();
  renderSidebar();
  scrollChat();
}

// Small safe Markdown renderer: construct DOM nodes, never inject model HTML.
function inline(parent, text) {
  const pattern = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    parent.append(document.createTextNode(text.slice(cursor, match.index)));
    const part = match[0];
    if (part.startsWith('`')) parent.append(el('code', '', part.slice(1, -1)));
    else if (part.startsWith('**')) parent.append(el('strong', '', part.slice(2, -2)));
    else {
      const parsed = part.match(/^\[([^\]]+)\]\((.+)\)$/);
      const link = el('a', '', parsed[1]);
      link.href = parsed[2]; link.target = '_blank'; link.rel = 'noopener noreferrer';
      parent.append(link);
    }
    cursor = match.index + part.length;
  }
  parent.append(document.createTextNode(text.slice(cursor)));
}
function markdown(container, text) {
  const blocks = text.split(/```[^\n]*\n?/);
  blocks.forEach((block, index) => {
    if (index % 2) {const pre = el('pre'); pre.append(el('code', '', block.replace(/\n$/, ''))); container.append(pre); return;}
    let paragraph = [], list = null;
    const flush = () => {if (paragraph.length) {const p = el('p'); inline(p, paragraph.join('\n')); container.append(p); paragraph = [];}};
    for (const line of block.split('\n')) {
      const heading = line.match(/^(#{1,4})\s+(.+)$/);
      const bullet = line.match(/^\s*(?:[-*]|\d+[.)])\s+(.+)$/);
      if (heading) {flush(); list = null; const h = el(`h${Math.min(heading[1].length + 1, 4)}`); inline(h, heading[2]); container.append(h);}
      else if (bullet) {flush(); const tag = /^\s*\d/.test(line) ? 'OL' : 'UL'; if (!list || list.tagName !== tag) {list = el(tag.toLowerCase()); container.append(list);} const li = el('li'); inline(li, bullet[1]); list.append(li);}
      else if (!line.trim()) {flush(); list = null;}
      else {list = null; paragraph.push(line);}
    }
    flush();
  });
}
function renderMessages() {
  $('#welcome').hidden = state.messages.length > 0;
  const box = $('#messages');
  box.hidden = !state.messages.length;
  box.replaceChildren();
  for (const message of state.messages) {
    const meta = message.meta || {};
    const node = el('article', `message ${message.role}${meta.error ? ' error' : ''}${meta.streaming ? ' streaming' : ''}`);
    if (meta.run_id) node.dataset.runId = meta.run_id;
    const head = el('div', 'message-head');
    if (message.role === 'assistant') {const img = el('img'); img.src = '/favicon.svg'; img.alt = ''; head.append(img);}
    head.append(document.createTextNode(message.role === 'user' ? '你' : '木记 Mori'));
    const body = el('div', 'message-body');
    if (message.role === 'user' || meta.error || meta.streaming) body.textContent = message.content;
    else markdown(body, message.content);
    node.append(head, body);
    if (meta.streaming) node.append(el('div', 'message-meta', '正在流式生成…'));
    if (message.role === 'assistant' && !meta.streaming) {
      const footer = el('div', 'message-meta');
      if (meta.model) footer.append(el('span', '', meta.model));
      if (meta.recalled?.length) footer.append(button(`✳ 参考 ${meta.recalled.length} 条记忆`, '', () => setView('memory')));
      if (meta.personal_context?.memory_ids.length) footer.append(el('span', '', `基础上下文 ${meta.personal_context.memory_ids.length} 条`));
      for (const tool of meta.tools || []) footer.append(el('span', `trace${tool.ok ? '' : ' failed'}`, `${tool.ok ? '✓' : '×'} ${toolNames[tool.name] || tool.name}`));
      if (meta.total_tokens) footer.append(el('span', '', `${meta.total_tokens.toLocaleString()} tokens`));
      if (meta.reasoning_tokens !== undefined) footer.append(el('span', '', meta.reasoning_tokens === null ? '思考 Token：接口未提供' : `主 Agent 思考 ${meta.reasoning_tokens.toLocaleString()} tokens`));
      if (meta.run_id) footer.append(button('查看流程 ↗', '', async () => {await setView('flow'); await FlowView.select(meta.run_id);}));
      footer.append(button('复制', '', async () => {await navigator.clipboard.writeText(message.content); toast('已复制');}));
      node.append(footer);
      if (meta.configuration?.model_options) {
        const config = el('details', 'message-configuration');
        config.append(el('summary', '', '本轮模型参数'), el('p', '', `${meta.model} · ${optionSummary(meta.configuration.model_options, meta.provider)} · 请求超时 ${meta.configuration.model_options.timeout_seconds} 秒`));
        node.append(config);
      }
      if (meta.sources?.length) {
        const sources = el('div', 'message-sources'); sources.append(el('span', '', '搜索来源'));
        for (const source of meta.sources) {
          if (!/^https?:\/\//i.test(source.url)) continue;
          const link = el('a', '', source.title || source.url); link.href = source.url; link.target = '_blank'; link.rel = 'noopener noreferrer'; sources.append(link);
        }
        node.append(sources);
      }
      if (meta.memory) {
        const memory = meta.memory;
        const panel = el('details', `memory-result ${memory.status}`);
        const labels = {pending: '记忆子 Agent 处理中', changed: '记忆已更新', skipped: '本轮不改动记忆', readonly: '记忆只读', error: '记忆处理未完成'};
        panel.append(el('summary', '', `✳ ${labels[memory.status] || memory.status}`));
        if (memory.reason) panel.append(el('p', '', memory.reason));
        for (const change of memory.changes || []) panel.append(el('div', '', `${{create: '新增', update: '更新', delete: '删除'}[change.operation]} · ${change.key}`));
        if (memory.status === 'pending') panel.append(el('p', '', '回答已返回，记忆子 Agent 仍在独立处理。'));
        node.append(panel);
      }
    }
    box.append(node);
  }
}
function renderPersonalContext() {
  const context = state.personal_context;
  if (!context) return;
  const count = context.memories.length;
  $('#personal-context-count').textContent = count ? `已准备 ${count} 条` : '尚无个人信息与偏好';
  $('#personal-context-note').textContent = count
    ? `第一条消息就会带上这些内容，无需先询问“你记得我吗”。每轮会读取最新版本。${context.omitted ? `另有 ${context.omitted} 条因容量限制未预载，子 Agent 仍可按需查询。` : ''}`
    : '保存为「关于我」「偏好习惯」，或将记忆置顶后，会自动带入新对话。';
  const list = $('#personal-context-list'); list.replaceChildren();
  for (const memory of context.memories) {
    const row = el('li');
    row.append(el('span', 'personal-context-category', `${memory.pinned ? '置顶 · ' : ''}${categoryNames[memory.category]}`), el('span', '', memory.content));
    list.append(row);
  }
}
function scrollChat() {const box = $('#conversation'); box.scrollTop = box.scrollHeight;}
function setBusy(busy) {
  state.busy = busy;
  $('#send-button').disabled = busy;
  $('#chat-input').disabled = busy;
  $('#new-chat').disabled = busy;
  $('#run-status').hidden = !busy;
  $('#show-live-flow').hidden = !busy;
  $('#send-button').textContent = busy ? '·' : '↑';
}
async function sendMessage(event) {
  event.preventDefault();
  const message = $('#chat-input').value.trim();
  if (!message || state.busy) return;
  if (!state.settings.keys[state.settings.provider]) {openSettings(); toast('填写 API Key 后即可开始对话'); return;}
  setBusy(true);
  let sent = false;
  let finished = false;
  let memoryResult = null;
  let currentAnswerId = null;
  let renderFrame = null;
  const statuses = {main: '等待上下文', memory: '准备整理上下文'};
  function renderStatus() {
    $('#run-status').replaceChildren(el('span', 'agent-progress', `主 Agent · ${statuses.main}`), el('span', 'agent-progress', `记忆 Agent · ${statuses.memory}`));
  }
  function upsertMessage(message) {
    currentAnswerId = message.id;
    if (memoryResult && message.meta.memory?.status === 'pending') message.meta.memory = memoryResult;
    const index = state.messages.findIndex(m => m.id === message.id ||
      (m.meta?.streaming && m.meta.run_id === message.meta.run_id));
    if (index >= 0) state.messages[index] = message;
    else state.messages.push(message);
    renderMessages(); scrollChat();
  }
  try {
    if (!state.active) {const session = await api('/api/sessions', {}); state.active = session.id;}
    state.messages.push({role: 'user', content: message});
    $('#chat-input').value = '';
    renderMessages(); scrollChat();
    renderStatus();
    const response = await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Mori-Token': state.csrf}, body: JSON.stringify({session_id: state.active, message})});
    if (!response.ok) {const data = await response.json(); throw new Error(data.error || '请求失败');}
    sent = true;
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    function receive(line) {
      if (!line.trim()) return;
      const data = JSON.parse(line);
      if (data.event === 'flow') FlowView.receive(data.trace);
      if (data.event === 'agent_status') {statuses[data.agent] = data.message; renderStatus();}
      if (data.event === 'tool') {statuses[data.agent || 'memory'] = `${data.ok ? '已执行' : '待纠正'}：${toolNames[data.name] || data.name}`; renderStatus();}
      if (data.event === 'answer_start') upsertMessage({id: `stream-${data.run_id}`, role: 'assistant', content: '', meta: {...data.meta, streaming: true}});
      if (data.event === 'answer_reset') {
        const draft = state.messages.find(m => m.meta?.streaming && m.meta.run_id === data.run_id);
        if (draft) {draft.content = ''; renderMessages();}
      }
      if (data.event === 'answer_delta') {
        const draft = state.messages.find(m => m.meta?.streaming && m.meta.run_id === data.run_id);
        if (draft) {
          draft.content += data.delta;
          statuses.main = '正在流式回答'; renderStatus();
          if (renderFrame === null) renderFrame = requestAnimationFrame(() => {
            renderFrame = null;
            const current = state.messages.find(m => m.meta?.streaming && m.meta.run_id === data.run_id);
            const node = [...document.querySelectorAll('.message.streaming')].find(n => n.dataset.runId === data.run_id);
            if (current && node) {node.querySelector('.message-body').textContent = current.content; scrollChat();}
          });
        }
      }
      if (data.event === 'answer') upsertMessage(data.message);
      if (data.event === 'memory_result') {
        memoryResult = data.result;
        const pending = state.messages.find(m => m.id === currentAnswerId);
        if (pending) {pending.meta.memory = memoryResult; renderMessages();}
      }
      if (data.event === 'done') {finished = true; upsertMessage(data.message);}
      if (data.event === 'error') {finished = true; toast(data.message, true);}
    }
    while (true) {
      const {done, value} = await reader.read();
      if (done) {buffer += decoder.decode(); if (buffer.trim()) receive(buffer); break;}
      buffer += decoder.decode(value, {stream: true});
      let index;
      while ((index = buffer.indexOf('\n')) >= 0) {receive(buffer.slice(0, index)); buffer = buffer.slice(index + 1);}
    }
    if (!finished) throw new Error('连接意外中断。请查看已保存的消息和工具结果后再继续。');
  } catch (error) {
    toast(error.message || '连接已中断，请检查本机服务。', true);
    if (!sent) $('#chat-input').value = message;
  } finally {
    if (renderFrame !== null) cancelAnimationFrame(renderFrame);
    await safe(async () => {await refresh(); if (state.active) {state.messages = await api(`/api/messages?session=${encodeURIComponent(state.active)}`); renderMessages(); scrollChat();}});
    setBusy(false);
    if (state.view === 'chat') $('#chat-input').focus();
  }
}

function empty(icon, title, description) {const node = el('div', 'empty-state'); node.append(el('div', 'empty-icon', icon), el('h3', '', title), el('p', '', description)); return node;}
function date(value) {return new Date(value).toLocaleDateString('zh-CN', {month: 'short', day: 'numeric'});}
async function renderMemories() {
  const generation = ++searchGeneration;
  const q = $('#memory-search').value.trim();
  const found = q ? await api(`/api/memories?q=${encodeURIComponent(q)}`) : state.memories;
  if (generation !== searchGeneration) return;
  const items = found.filter(m => state.category === 'all' || m.category === state.category);
  const grid = $('#memory-grid'); grid.replaceChildren();
  if (!items.length) {grid.append(empty('✳', q || state.category !== 'all' ? '这里还没有匹配的记忆' : '记忆从一句话开始', '在对话中说「请记住……」，或点击「添加记忆」。\n保存后，即使新开对话或重启，木记也能再次找到它。')); return;}
  for (const memory of items) {
    const card = el('article', 'memory-card');
    const head = el('div', 'memory-card-header');
    head.append(el('span', 'category-label', categoryNames[memory.category]), el('span', '', memory.pinned ? '✦ 已置顶' : date(memory.updated_at)));
    card.append(head, el('p', '', memory.content), el('small', '', `${memory.source === 'manual' ? '手动保存' : '对话中保存'} · ${date(memory.updated_at)}`));
    if (memory.evidence) {const details = el('details'); details.append(el('summary', '', '查看保存依据'), el('p', '', memory.evidence)); card.append(details);}
    const actions = el('div', 'memory-actions');
    actions.append(button(memory.pinned ? '取消置顶' : '置顶', '', async () => {
      await api('/api/memories/save', {id: memory.id, key: memory.key, content: memory.content, category: memory.category, pinned: !memory.pinned}); await refresh();
    }), button('编辑', '', () => openMemory(memory)), button('忘记', 'danger', async () => {
      if (!confirm('让木记忘记这条长期记忆？原始聊天仍会保留。如需完全移除这段内容，也请删除对应会话。')) return;
      await api('/api/memories/delete', {id: memory.id}); await refresh(); toast('这条长期记忆已删除');
    }));
    card.append(actions); grid.append(card);
  }
}
function openMemory(memory = null) {
  $('#memory-dialog-title').textContent = memory ? '编辑这条记忆' : '添加一条记忆';
  $('#memory-id').value = memory?.id || '';
  $('#memory-content').value = memory?.content || '';
  $('#memory-key').value = memory?.key || '';
  $('#memory-category').value = memory?.category || 'profile';
  $('#memory-pinned').checked = Boolean(memory?.pinned);
  $('#memory-dialog').showModal();
}
function renderTasks() {
  const list = $('#task-list'); list.replaceChildren();
  if (!state.tasks.length) {list.append(empty('☷', '给下一步，留一点空间', '在上方添加待办，也可以先和木记一起梳理计划。')); return;}
  for (const task of state.tasks) {
    const row = el('div', `task-row${task.done ? ' done' : ''}`);
    const label = el('label'); const check = el('input'); check.type = 'checkbox'; check.checked = Boolean(task.done);
    check.addEventListener('change', () => safe(async () => {
      try {await api('/api/tasks/set', {id: task.id, done: check.checked}); await refresh();}
      catch (error) {check.checked = Boolean(task.done); throw error;}
    }));
    label.append(check, el('span', '', task.title));
    row.append(label, button('删除', '', async () => {if (!confirm('删除这条待办？')) return; await api('/api/tasks/delete', {id: task.id}); await refresh();}));
    list.append(row);
  }
}

function populateProvider() {
  const config = settingsDraft[settingsProvider];
  $('#model').value = config.model;
  $('#base-url').value = config.base_url;
  const options = optionsFor(config, settingsProvider);
  const effort = $('#reasoning-effort'); effort.replaceChildren();
  const values = state.settings.efforts?.[settingsProvider] || (settingsProvider === 'deepseek' ? ['auto', 'none', 'low', 'high', 'max'] : ['auto', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh']);
  for (const value of values) {const option = el('option', '', `${effortNames[value]}${['auto', 'none'].includes(value) ? '' : ` · ${value}`}`); option.value = value; effort.append(option);}
  effort.value = options.reasoning_effort;
  $('#max-output-tokens').value = options.max_output_tokens;
  $('#request-timeout').value = options.timeout_seconds;
  $('#temperature').value = options.temperature ?? '';
  $('#top-p').value = options.top_p ?? '';
  $('#verbosity').value = options.verbosity;
  $('#web-search').checked = options.web_search;
  $('#search-context-size').value = options.search_context_size;
  $('#api-key').value = '';
  $('#clear-key').checked = false;
  renderKeyStatus();
  $('#settings-feedback').textContent = '';
  updateCapabilities();
}
function renderKeyStatus() {
  const saved = state.settings.key_storage?.method === 'local_database';
  $('#key-status').textContent = state.settings.keys[settingsProvider] ? (saved ? '已保存到本机，留空保留' : '已配置，留空保留现有 Key') : '尚未配置';
}
function updateCapabilities() {
  const deepseek = settingsProvider === 'deepseek';
  const thinking = $('#reasoning-effort').value !== 'none';
  $('#verbosity').disabled = deepseek;
  $('#web-search').disabled = deepseek;
  $('#search-context-size').disabled = deepseek || !$('#web-search').checked;
  $('#temperature').disabled = deepseek && thinking;
  $('#top-p').disabled = deepseek && !thinking;
  $('#top-p').min = deepseek ? '0.95' : '0.01';
  $('#reasoning-note').textContent = deepseek ? 'none 关闭思考；auto 不传思考设置；low / high / max 开启对应强度。' : 'auto 不传思考设置，适用于不确定模型能力时；其他强度需模型支持。选择的模型或兼容网关可能不接受某些值，连接测试会返回结果。';
  $('#sampling-note').textContent = deepseek ? 'Temperature 仅关闭思考时生效；Top-P 仅思考开启时生效且范围为 0.95–1。禁用字段不发送。' : '留空时由服务商决定。部分推理模型不接受 Temperature / Top-P，或仅在关闭思考时接受；不确定时留空。一般只调整其中一项。';
  $('#web-search-note').textContent = deepseek ? '当前 DeepSeek 适配器未接入联网搜索' : 'Responses API 内置搜索；默认关闭，开启后按需调用，模型与网关需支持，可能产生额外搜索费用';
}
function readModelOptions() {
  const optional = id => $(id).disabled || $(id).value === '' ? null : Number($(id).value);
  return {reasoning_effort: $('#reasoning-effort').value, max_output_tokens: Number($('#max-output-tokens').value), timeout_seconds: Number($('#request-timeout').value),
    temperature: optional('#temperature'), top_p: optional('#top-p'), verbosity: settingsProvider === 'deepseek' ? 'auto' : $('#verbosity').value,
    web_search: settingsProvider === 'openai' && $('#web-search').checked, search_context_size: $('#search-context-size').value};
}
function openSettings() {
  settingsProvider = state.settings.provider;
  settingsDraft = structuredClone(state.settings);
  $('#provider').value = settingsProvider;
  $('#auto-memory').checked = state.settings.auto_memory;
  $('#persona').value = state.settings.persona;
  const runtime = {...fallbackRuntime, ...state.settings.runtime};
  for (const [key, id] of Object.entries(runtimeFields)) $('#' + id).value = runtime[key];
  $('#recall-mode').value = runtime.recall_mode;
  $('#prefetch-candidates').checked = runtime.prefetch_candidates;
  updateRecallMode();
  populateProvider();
  populateEmbedding();
  $('#settings-dialog').showModal();
}
function updateRecallMode() {
  const main = $('#recall-mode').value === 'main';
  $('#prefetch-candidates').disabled = main;
  $('#handoff-summary-chars').disabled = main;
  $('#recall-mode-note').textContent = main ? '主 Agent 自主选择直接回答或只读检索。基础个人上下文始终载入，记忆子 Agent 并行判断是否维护。' : '先预取候选，由子 Agent 尽量一次筛选交接；候选不足时仍可自主补查。原始事实按选中 ID 传递，备注只说明必要的不确定性。';
}
$('#recall-mode').addEventListener('change', updateRecallMode);
function populateEmbedding() {
  const config = state.settings.embedding;
  $('#embedding-enabled').checked = config.enabled;
  $('#embedding-model').value = config.model;
  $('#embedding-base').value = config.base_url;
  $('#embedding-key-source').value = config.key_source;
  for (const field of ['dimensions', 'batch_size', 'candidate_limit', 'min_similarity']) $('#embedding-' + field.replaceAll('_', '-')).value = config[field];
  $('#embedding-timeout').value = config.timeout_seconds;
  $('#embedding-key').value = '';
  $('#embedding-clear-key').checked = false;
  updateEmbeddingKey();
}
function updateEmbeddingKey() {
  const reuse = $('#embedding-key-source').value === 'openai';
  $('#embedding-key').disabled = reuse;
  $('#embedding-clear-key').disabled = reuse;
  $('#embedding-base').readOnly = reuse;
  if (reuse) $('#embedding-base').value = settingsProvider === 'openai' ? $('#base-url').value : settingsDraft.openai?.base_url || state.settings.openai.base_url;
  $('#embedding-key-status').textContent = reuse ? (state.settings.keys.openai ? '复用已保存的 OpenAI Key' : '需先保存 OpenAI Key') : (state.settings.separate_embedding_key ? '已保存，留空保留' : '尚未配置');
}
function readEmbedding() {
  updateEmbeddingKey();
  return {enabled: $('#embedding-enabled').checked, key_source: $('#embedding-key-source').value,
    base_url: $('#embedding-base').value.trim(), model: $('#embedding-model').value.trim(),
    dimensions: Number($('#embedding-dimensions').value), timeout_seconds: Number($('#embedding-timeout').value),
    batch_size: Number($('#embedding-batch-size').value), candidate_limit: Number($('#embedding-candidate-limit').value),
    min_similarity: Number($('#embedding-min-similarity').value),
    api_key: $('#embedding-key').disabled ? '' : $('#embedding-key').value.trim(), clear_key: !$('#embedding-clear-key').disabled && $('#embedding-clear-key').checked};
}
async function renderEmbeddingStatus() {
  const info = await api('/api/retrieval/status');
  const text = `${info.enabled ? `关键词 + ${info.model}` : 'FTS5 关键词'} · 语义索引 ${info.indexed} / ${info.total} 条${info.pending ? ` · 待更新 ${info.pending} 条` : ''}${info.state === 'running' ? ' · 正在后台建立' : ''}${info.error ? ` · ${info.error}` : ''}`;
  $('#retrieval-mode').textContent = info.enabled ? '混合召回' : '关键词索引';
  $('#embedding-status').textContent = text;
  clearTimeout(embeddingTimer);
  if (info.state === 'running' && $('#settings-dialog').open) embeddingTimer = setTimeout(() => safe(renderEmbeddingStatus), 1200);
}
async function saveSettings(probe = false) {
  if (!$('#settings-form').reportValidity()) return;
  const feedback = $('#settings-feedback'); feedback.classList.remove('error');
  if (state.settings.key_storage?.method !== 'local_database') {feedback.textContent = '请先重启服务，再保存 Key 到本机数据库'; feedback.classList.add('error'); return;}
  const body = {provider: $('#provider').value, model: $('#model').value.trim(), base_url: $('#base-url').value.trim(), api_key: $('#api-key').value.trim(), auto_memory: $('#auto-memory').checked, persona: $('#persona').value.trim(), clear_key: $('#clear-key').checked,
    options: readModelOptions(), embedding: readEmbedding(), runtime: {...Object.fromEntries(Object.entries(runtimeFields).map(([key, id]) => [key, Number($('#' + id).value)])), recall_mode: $('#recall-mode').value, prefetch_candidates: $('#prefetch-candidates').checked}};
  const controls = [...$('#settings-form').elements];
  controls.forEach(node => {node.disabled = true;});
  try {
    await api('/api/settings', body);
    $('#api-key').value = '';
    $('#embedding-key').value = '';
    await refresh();
    if (probe === true) {feedback.textContent = '正在进行真实 API 连接测试…'; const result = await api('/api/probe', {}); feedback.textContent = result.message;}
    else if (probe === 'embedding-probe') {feedback.textContent = '正在测试 Embedding 接口…'; const result = await api('/api/embedding/probe', {}); feedback.textContent = `${result.message} ${result.model} · ${result.dimensions} 维 · ${result.elapsed_ms} ms · ${result.tokens} Token`; await renderEmbeddingStatus();}
    else if (probe === 'embedding-index') {await api('/api/embedding/index', {}); feedback.textContent = '索引在后台处理，可在下方查看进度。'; await renderEmbeddingStatus();}
    else {$('#settings-dialog').close(); toast('设置已保存');}
  } catch (error) {feedback.textContent = error.message; feedback.classList.add('error');}
  finally {controls.forEach(node => {node.disabled = false;}); updateCapabilities(); renderKeyStatus(); updateEmbeddingKey(); updateRecallMode();}
}

async function testRetrieval() {
  const query = $('#retrieval-query').value.trim();
  if (!query) return toast('先输入一个召回测试问题');
  const button = $('#retrieval-test'); button.disabled = true;
  $('#retrieval-feedback').textContent = '正在召回候选记忆…';
  $('#retrieval-results').replaceChildren();
  try {
    const result = await api('/api/retrieval/test', {query});
    const d = result.retrieval;
    $('#retrieval-feedback').textContent = `${d.mode === 'hybrid' ? '混合召回 · RRF' : '关键词召回 · FTS5'} · 关键词 ${d.lexical_count} 条 · 语义 ${d.semantic_count} 条 · 关键词 ${d.keyword_ms} ms · Embedding ${d.embedding_ms} ms${d.query_cache_hit ? '（查询缓存命中）' : ''}${d.fallback ? ` · ${d.fallback}` : ''}`;
    for (const memory of result.memories) {const row = el('div', 'retrieval-result'); row.append(el('strong', '', memory.key), el('p', '', memory.content)); $('#retrieval-results').append(row);}
    if (!result.memories.length) $('#retrieval-results').append(el('p', '', '没有召回候选。'));
  } catch (error) {$('#retrieval-feedback').textContent = error.message;}
  finally {button.disabled = false;}
}

$('#new-chat').addEventListener('click', () => safe(newChat));
$$('[data-view]').forEach(node => node.addEventListener('click', () => safe(() => setView(node.dataset.view))));
$('#explore-memory').addEventListener('click', () => safe(() => setView('memory')));
$('#menu-button').addEventListener('click', () => $('#sidebar').classList.toggle('open'));
$('#settings-button').addEventListener('click', () => safe(openSettings));
$('#model-button').addEventListener('click', () => safe(openSettings));
$('#model-settings-summary').addEventListener('click', () => safe(openSettings));
$('#show-live-flow').addEventListener('click', () => safe(async () => {await setView('flow'); await FlowView.follow();}));
$('#manage-personal-context').addEventListener('click', () => safe(() => setView('memory')));
$$('[data-prompt]').forEach(node => node.addEventListener('click', () => {$('#chat-input').value = node.dataset.prompt; $('#chat-input').focus();}));
$('#chat-form').addEventListener('submit', event => safe(() => sendMessage(event)));
$('#chat-input').addEventListener('keydown', event => {if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {event.preventDefault(); $('#chat-form').requestSubmit();}});
document.addEventListener('keydown', event => {if (event.key.toLowerCase() === 'n' && !event.ctrlKey && !event.metaKey && !event.altKey && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName) && !document.querySelector('dialog[open]')) safe(newChat);});
$$('.close-dialog').forEach(node => node.addEventListener('click', () => node.closest('dialog').close()));
$('#settings-dialog').addEventListener('close', () => {$('#api-key').value = ''; $('#embedding-key').value = ''; clearTimeout(embeddingTimer);});
$('#embedding-key-source').addEventListener('change', updateEmbeddingKey);
$('#base-url').addEventListener('input', updateEmbeddingKey);
$('#embedding-probe').addEventListener('click', () => safe(() => saveSettings('embedding-probe')));
$('#embedding-index').addEventListener('click', () => safe(() => saveSettings('embedding-index')));
$('#retrieval-test').addEventListener('click', () => safe(testRetrieval));
$('#provider').addEventListener('change', () => {settingsDraft[settingsProvider] = {model: $('#model').value, base_url: $('#base-url').value, options: readModelOptions()}; settingsProvider = $('#provider').value; populateProvider(); updateEmbeddingKey();});
$('#reasoning-effort').addEventListener('change', updateCapabilities);
$('#web-search').addEventListener('change', updateCapabilities);
$('#settings-form').addEventListener('submit', event => {event.preventDefault(); safe(() => saveSettings());});
$('#probe-button').addEventListener('click', () => safe(() => saveSettings(true)));
$('#add-memory').addEventListener('click', () => openMemory());
$('#memory-form').addEventListener('submit', event => {event.preventDefault(); safe(async () => {
  await api('/api/memories/save', {id: $('#memory-id').value || undefined, content: $('#memory-content').value.trim(), category: $('#memory-category').value, key: $('#memory-key').value.trim(), pinned: $('#memory-pinned').checked});
  $('#memory-dialog').close(); await refresh(); toast('记忆已保存，下次对话也能找到');
});});
$('#memory-search').addEventListener('input', () => {clearTimeout(searchTimer); searchTimer = setTimeout(() => safe(renderMemories), 180);});
$$('[data-category]').forEach(node => node.addEventListener('click', () => {state.category = node.dataset.category; $$('.filter').forEach(item => item.classList.toggle('active', item === node)); safe(renderMemories);}));
$('#task-form').addEventListener('submit', event => {event.preventDefault(); safe(async () => {await api('/api/tasks/add', {title: $('#task-input').value.trim()}); $('#task-input').value = ''; await refresh();});});
$('#export-button').addEventListener('click', () => safe(async () => {const data = await api('/api/export'); const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'})); const link = el('a'); link.href = url; link.download = `mori-backup-${new Date().toISOString().slice(0, 10)}.json`; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000); toast('已导出聊天、记忆与待办；不含 API Key');}));

FlowView.init({api, openChat: id => safe(async () => {
  if (id === state.active) await setView('chat');
  else await loadSession(id);
})});
safe(async () => {await refresh(); renderMessages(); if (location.hash === '#flow') await setView('flow');});
