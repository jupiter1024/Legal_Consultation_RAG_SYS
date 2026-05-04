document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const fileUpload = document.getElementById('file-upload');
    const uploadStatus = document.getElementById('upload-status');
    const sourceModal = document.getElementById('source-modal');
    const modalBody = document.getElementById('modal-body');
    const closeBtn = document.querySelector('.close-btn');

    // Handle Sending Messages
    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        appendMessage('user', text);
        userInput.value = '';
        
        // Bot Typing State
        const botMsgDiv = appendMessage('bot', '...', true);

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });

            if (!response.ok) throw new Error('Failed to get response');

            const data = await response.json();
            updateBotMessage(botMsgDiv, data.answer, data.sources);
        } catch (error) {
            botMsgDiv.querySelector('.message-content').textContent = "Error: " + error.message;
        }
    }

    function appendMessage(type, text, isPending = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerText = text;
        
        msgDiv.appendChild(contentDiv);
        chatWindow.appendChild(msgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        
        return msgDiv;
    }

    function updateBotMessage(div, answer, sources) {
        const contentDiv = div.querySelector('.message-content');
        contentDiv.innerText = answer;

        if (sources && sources.length > 0) {
            const btn = document.createElement('span');
            btn.className = 'sources-btn';
            btn.innerHTML = `<i class="fas fa-info-circle"></i> View ${sources.length} Sources`;
            btn.onclick = () => showSources(sources);
            div.appendChild(btn);
        }
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function showSources(sources) {
        modalBody.innerHTML = '';
        sources.forEach(s => {
            const card = document.createElement('div');
            card.className = 'source-card';
            card.innerHTML = `
                <h4>${s.source}</h4>
                <p><strong>Page:</strong> ${s.page}</p>
                <p><strong>Context:</strong> ${s.text_snippet}</p>
            `;
            modalBody.appendChild(card);
        });
        sourceModal.style.display = 'block';
    }

    // Handle File Uploads
    fileUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        uploadStatus.innerText = `Uploading ${file.name}...`;
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');
            
            const data = await response.json();
            uploadStatus.innerText = "✓ " + data.message;
            uploadStatus.style.color = "#10b981";
        } catch (error) {
            uploadStatus.innerText = "✗ Error: " + error.message;
            uploadStatus.style.color = "#ef4444";
        }
    });

    // Event Listeners
    sendBtn.onclick = sendMessage;
    userInput.onkeypress = (e) => { if (e.key === 'Enter') sendMessage(); };
    closeBtn.onclick = () => { sourceModal.style.display = 'none'; };
    window.onclick = (e) => { if (e.target == sourceModal) sourceModal.style.display = 'none'; };
});
