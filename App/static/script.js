/**
 * LinuxGPT — Frontend Script
 * Handles: provider/model dropdown, chat, markdown rendering, file upload, source modal
 */

document.addEventListener('DOMContentLoaded', () => {

    const chatWindow      = document.getElementById('chat-window');
    const userInput       = document.getElementById('user-input');
    const sendBtn         = document.getElementById('send-btn');
    const fileUpload      = document.getElementById('file-upload');
    const uploadStatus    = document.getElementById('upload-status');
    const providerSelect  = document.getElementById('provider-select');
    const modelSelect     = document.getElementById('model-select');
    const modelBadge      = document.getElementById('badge-text');
    const activeModelHint = document.getElementById('active-model-hint');
    const sourceModal     = document.getElementById('source-modal');
    const modalBody       = document.getElementById('modal-body');
    const modalCloseBtn   = document.getElementById('modal-close-btn');

    // ── marked.js config ────────────────────────────────────────────────────
    marked.setOptions({
        breaks: true,
        gfm: true,
    });

    const renderer = new marked.Renderer();
    renderer.code = (code, lang) => {
        const language = lang && hljs.getLanguage(lang) ? lang : 'bash';
        const highlighted = hljs.highlight(code, { language }).value;
        return `<pre><code class="hljs language-${language}">${highlighted}</code></pre>`;
    };
    marked.setOptions({ renderer });

    // ── Provider Registry ────────────────────────────────────────────────────
    let providers = {};

    async function loadProviders() {
        try {
            const res  = await fetch('/llm/providers');
            const data = await res.json();

            providerSelect.innerHTML = '';
            providers = {};

            data.forEach(entry => {
                providers[entry.provider] = entry.models;
                const opt = document.createElement('option');
                opt.value       = entry.provider;
                opt.textContent = entry.provider.charAt(0).toUpperCase() + entry.provider.slice(1);
                providerSelect.appendChild(opt);
            });

            if (data.length > 0) {
                providerSelect.value = data[0].provider;
                updateModelSelect(data[0].provider);
            }
        } catch (err) {
            providerSelect.innerHTML = '<option value="groq">Groq</option>';
            modelSelect.innerHTML    = '<option value="llama-3.1-8b-instant">llama-3.1-8b-instant</option>';
            updateBadge('groq', 'llama-3.1-8b-instant');
        }
    }

    function updateModelSelect(provider) {
        const models = providers[provider] || [];
        modelSelect.innerHTML = models
            .map((m, i) => `<option value="${m}"${i === 0 ? ' selected' : ''}>${m}</option>`)
            .join('');
        updateBadge(provider, models[0] || '');
    }

    function updateBadge(provider, model) {
        modelBadge.textContent      = `${provider} · ${model}`;
        activeModelHint.textContent = `${provider}/${model}`;
    }

    providerSelect.addEventListener('change', () => updateModelSelect(providerSelect.value));
    modelSelect.addEventListener('change', () => updateBadge(providerSelect.value, modelSelect.value));

    loadProviders();

    // ── Welcome Message ──────────────────────────────────────────────────────
    chatWindow.innerHTML = `
        <div class="welcome-card">
            <span class="welcome-icon">🐧</span>
            <h2>$ man linuxgpt</h2>
            <p>
                I'm your Linux documentation assistant, powered by RAG.<br>
                Upload Linux command books and ask me anything — I'll cite my sources.
            </p>
        </div>
    `;

    // ── Textarea auto-expand ─────────────────────────────────────────────────
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 160) + 'px';
    });

    userInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    // ── Quick Queries ────────────────────────────────────────────────────────
    document.querySelectorAll('.quick-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            userInput.value = btn.dataset.query;
            userInput.dispatchEvent(new Event('input'));
            sendMessage();
        });
    });

    // ── Chat ─────────────────────────────────────────────────────────────────
    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        const welcome = chatWindow.querySelector('.welcome-card');
        if (welcome) welcome.remove();

        appendUserMessage(text);
        userInput.value = '';
        userInput.style.height = 'auto';
        sendBtn.disabled = true;

        const typingEl = appendTypingIndicator();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message:  text,
                    provider: providerSelect.value || null,
                    model:    modelSelect.value    || null,
                }),
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${response.status}`);
            }

            const data = await response.json();
            typingEl.remove();
            appendBotMessage(data.answer, data.sources, data.provider, data.model);

            if (data.provider && data.model) {
                updateBadge(data.provider, data.model);
            }
        } catch (err) {
            typingEl.remove();
            appendBotMessage(`**Error:** ${err.message}`, [], '', '');
        } finally {
            sendBtn.disabled = false;
        }
    }

    // ── Message Builders ─────────────────────────────────────────────────────
    function appendUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'message user';
        div.innerHTML = `<div class="message-bubble">${escapeHtml(text)}</div>`;
        chatWindow.appendChild(div);
        scrollBottom();
    }

    function appendTypingIndicator() {
        const div = document.createElement('div');
        div.className = 'message bot typing-indicator';
        div.innerHTML = `
            <div class="bot-header">
                <div class="bot-avatar">$_</div>
                <span class="bot-name">LinuxGPT</span>
            </div>
            <div class="message-bubble">
                <div class="typing-dots"><span></span><span></span><span></span></div>
            </div>
        `;
        chatWindow.appendChild(div);
        scrollBottom();
        return div;
    }

    function appendBotMessage(answer, sources, provider, model) {
        const div = document.createElement('div');
        div.className = 'message bot';

        const label = provider && model ? `${provider}/${model}` : 'LinuxGPT';
        const rendered = marked.parse(answer);

        let sourcesBtn = '';
        if (sources && sources.length > 0) {
            const encoded = encodeURIComponent(JSON.stringify(sources));
            sourcesBtn = `
                <button class="sources-toggle" data-sources="${encoded}">
                    <i class="fas fa-book-open"></i>
                    View ${sources.length} source${sources.length > 1 ? 's' : ''}
                </button>
            `;
        }

        div.innerHTML = `
            <div class="bot-header">
                <div class="bot-avatar">$_</div>
                <span class="bot-name">${escapeHtml(label)}</span>
            </div>
            <div class="message-bubble">
                ${rendered}
                ${sourcesBtn}
            </div>
        `;

        chatWindow.appendChild(div);

        div.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));

        div.querySelectorAll('.sources-toggle').forEach(btn => {
            btn.addEventListener('click', () => {
                const srcs = JSON.parse(decodeURIComponent(btn.dataset.sources));
                showSourcesModal(srcs);
            });
        });

        scrollBottom();
    }

    // ── Source Modal ─────────────────────────────────────────────────────────
    function showSourcesModal(sources) {
        modalBody.innerHTML = '';
        sources.forEach(s => {
            const rerankColor = s.rerank_score > 5
                ? 'var(--green)'
                : s.rerank_score > 0
                    ? 'var(--amber)'
                    : 'var(--text-muted)';
            const card = document.createElement('div');
            card.className = 'source-card';
            card.innerHTML = `
                <div class="source-card-header">
                    <span class="source-filename">
                        <i class="fas fa-file-lines"></i>
                        ${escapeHtml(s.source)}
                    </span>
                    <div class="source-meta">
                        <span class="source-tag page">p.${s.page}</span>
                        <span class="source-tag score" title="FAISS cosine similarity">
                            cos ${Number(s.cosine_score).toFixed(3)}
                        </span>
                        <span class="source-tag score" style="color:${rerankColor}" title="Cross-encoder re-rank score">
                            rank ${Number(s.rerank_score).toFixed(3)}
                        </span>
                    </div>
                </div>
                <div class="source-snippet">${escapeHtml(s.text_snippet)}</div>
            `;
            modalBody.appendChild(card);
        });
        sourceModal.style.display = 'block';
    }

    modalCloseBtn.addEventListener('click', () => { sourceModal.style.display = 'none'; });
    window.addEventListener('click', e => { if (e.target === sourceModal) sourceModal.style.display = 'none'; });
    window.addEventListener('keydown', e => { if (e.key === 'Escape') sourceModal.style.display = 'none'; });

    // ── File Upload ───────────────────────────────────────────────────────────
    fileUpload.addEventListener('change', async e => {
        const file = e.target.files[0];
        if (!file) return;

        setUploadStatus(`⏳ Uploading ${file.name}...`, 'var(--text-secondary)');
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/upload', { method: 'POST', body: formData });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const data = await res.json();
            setUploadStatus(`✓ ${data.message}`, 'var(--green-dim)');
        } catch (err) {
            setUploadStatus(`✗ ${err.message}`, 'var(--red)');
        }

        fileUpload.value = '';
    });

    function setUploadStatus(msg, color) {
        uploadStatus.textContent = msg;
        uploadStatus.style.color  = color;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────
    function scrollBottom() {
        requestAnimationFrame(() => { chatWindow.scrollTop = chatWindow.scrollHeight; });
    }

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }
});
