const state = { messages: [], controller: null, requestId: null };
const $ = (id) => document.getElementById(id);

function setText(id, value) { $(id).textContent = value == null || value === '' ? '—' : String(value); }
function showError(error) {
  const box = $('error');
  box.hidden = !error;
  box.textContent = error ? `${error.code || 'error'}: ${error.message || 'Request failed'}` : '';
}
function renderMessages() {
  const target = $('messages');
  target.replaceChildren();
  if (!state.messages.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-state'; empty.textContent = 'Select a model and send a prompt to begin.';
    target.append(empty); return;
  }
  for (const message of state.messages) {
    const node = document.createElement('div');
    node.className = `message ${message.role}${message.pending ? ' pending' : ''}`;
    node.textContent = message.content || (message.pending ? 'Generating...' : '');
    target.append(node);
  }
  target.scrollTop = target.scrollHeight;
}
function selectedModel() { return $('model-select').value; }
function updateSelection() {
  const model = selectedModel(); setText('current-model', model || 'No model selected');
  setText('requested-model', model || '—');
}
async function loadRuntime() {
  try {
    const [healthResponse, detailResponse] = await Promise.all([fetch('/health'), fetch('/v1/models/detail')]);
    const health = await healthResponse.json();
    const detail = detailResponse.ok ? await detailResponse.json() : {};
    $('status-dot').className = health.inference_ready ? 'ready' : 'error';
    setText('status-label', health.inference_ready ? 'Ready' : 'Degraded');
    setText('api-status', health.api_ready ? 'Ready' : 'Unavailable');
    setText('inference-status', health.inference_ready ? 'Ready' : (health.reason || 'Degraded'));
    setText('default-model', health.default_model);
    setText('model-count', Array.isArray(health.models) ? health.models.length : 0);
    const selected = selectedModel();
    const modelDetail = (detail.models || []).find((item) => item.id === selected) || {};
    setText('resolved-model', modelDetail.id || selected);
    setText('provider-backend', modelDetail.provider || modelDetail.backend);
    setText('context-limit', modelDetail.context_window || modelDetail.context);
    setText('residency', modelDetail.resident ?? modelDetail.loaded);
  } catch (error) {
    $('status-dot').className = 'error'; setText('status-label', 'Unavailable');
    setText('api-status', 'Unavailable'); setText('inference-status', 'Unavailable');
    showError({ code: 'cloud_unreachable', message: error.message });
  }
}
async function loadModels() {
  try {
    const response = await fetch('/v1/models');
    const payload = await response.json();
    if (!response.ok) throw payload.error || { code: 'model_unavailable', message: 'Model inventory unavailable' };
    const select = $('model-select'); select.replaceChildren();
    for (const model of payload.data || []) {
      const option = document.createElement('option'); option.value = model.id; option.textContent = model.id; select.append(option);
    }
    select.addEventListener('change', () => { updateSelection(); void loadRuntime(); });
    updateSelection(); void loadRuntime();
  } catch (error) {
    showError(error.error || error); setText('model-count', 'Unavailable');
  }
}
function appendAssistant(content) {
  const last = state.messages[state.messages.length - 1];
  if (last && last.role === 'assistant' && last.pending) { last.content += content; renderMessages(); }
}
async function send(event) {
  event.preventDefault(); if (state.controller || !selectedModel()) return;
  const prompt = $('prompt').value.trim(); if (!prompt) return;
  showError(''); $('prompt').value = '';
  state.messages.push({ role: 'user', content: prompt });
  state.messages.push({ role: 'assistant', content: '', pending: true }); renderMessages();
  state.controller = new AbortController(); $('stop-button').disabled = false;
  try {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST', signal: state.controller.signal,
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ model: selectedModel(), messages: state.messages.filter((m) => !m.pending), stream: true, max_tokens: 256, temperature: 0.2 }),
    });
    if (!response.ok) {
      const payload = await response.json(); throw payload.error || { code: 'internal_inference_error', message: `HTTP ${response.status}` };
    }
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = '';
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n'); buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data:')) continue; const raw = line.slice(5).trim();
        if (raw === '[DONE]') continue;
        try { const chunk = JSON.parse(raw); appendAssistant(chunk.choices?.[0]?.delta?.content || ''); if (chunk.id) state.requestId = chunk.id; } catch (_) { /* ignore incomplete frames */ }
      }
    }
    const assistant = state.messages[state.messages.length - 1]; assistant.pending = false;
    setText('request-id', state.requestId); setText('resolved-model', selectedModel()); renderMessages();
  } catch (error) {
    const assistant = state.messages[state.messages.length - 1]; assistant.pending = false;
    if (error.name === 'AbortError') { assistant.content = '[cancelled]'; showError({ code: 'inference_cancelled', message: 'Generation stopped' }); }
    else { assistant.content = ''; showError(error.error || error); }
    renderMessages();
  } finally { state.controller = null; $('stop-button').disabled = true; void loadRuntime(); }
}
$('chat-form').addEventListener('submit', send);
$('stop-button').addEventListener('click', () => state.controller?.abort());
$('clear-button').addEventListener('click', () => { if (!state.controller) { state.messages = []; showError(''); renderMessages(); } });
void loadModels();
