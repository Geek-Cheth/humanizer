/**
 * AI Text Humanizer - Modern Workstation & SSE EventStream Controller
 * Real-time AI detection analysis, SSE streaming, interactive diff viewer, docx exporter.
 */

// ============================================
// Configuration
// ============================================
const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const API_BASE = isLocalhost ? 'http://localhost:5000' : '';

// ============================================
// Global State
// ============================================
let isAuthenticated = false;
let sessionToken = null;
let activeStyle = 'academic';
let activePasses = 2;
let originalText = '';
let humanizedText = '';
let analyzeDebounceTimer = null;

// ============================================
// DOM Elements
// ============================================
const inputText = document.getElementById('input-text');
const inputHighlight = document.getElementById('input-highlight');
const inputWordCount = document.getElementById('input-word-count');
const inputCharCount = document.getElementById('input-char-count');
const inputReadability = document.getElementById('input-readability');

const outputText = document.getElementById('output-text');
const diffText = document.getElementById('diff-text');
const outputWordCount = document.getElementById('output-word-count');
const outputCharCount = document.getElementById('output-char-count');
const outputReadability = document.getElementById('output-readability');

const humanizeBtn = document.getElementById('humanize-btn');
const scanBtn = document.getElementById('scan-btn');
const pasteBtn = document.getElementById('paste-btn');
const clearBtn = document.getElementById('clear-btn');
const copyBtn = document.getElementById('copy-btn');
const downloadDocxBtn = document.getElementById('download-docx-btn');
const downloadTxtBtn = document.getElementById('download-txt-btn');

const passesSlider = document.getElementById('passes-slider');
const passesVal = document.getElementById('passes-val');
const streamProgressBar = document.getElementById('stream-progress-bar');
const streamProgressFill = document.getElementById('stream-progress-fill');
const stepsList = document.getElementById('steps-list');

const aiScoreVal = document.getElementById('ai-score-val');
const gaugeFill = document.getElementById('gauge-fill');
const metricBurstiness = document.getElementById('metric-burstiness');
const metricPerplexity = document.getElementById('metric-perplexity');

const detGptzero = document.getElementById('det-gptzero');
const detValGptzero = document.getElementById('det-val-gptzero');
const detTurnitin = document.getElementById('det-turnitin');
const detValTurnitin = document.getElementById('det-val-turnitin');
const detCopyleaks = document.getElementById('det-copyleaks');
const detValCopyleaks = document.getElementById('det-val-copyleaks');
const detZerogpt = document.getElementById('det-zerogpt');
const detValZerogpt = document.getElementById('det-val-zerogpt');

const apiStatus = document.getElementById('api-status');
const pageLoader = document.getElementById('page-loader');

// ============================================
// Init Page & Animations
// ============================================
window.addEventListener('load', () => {
    if (pageLoader) {
        setTimeout(() => {
            pageLoader.style.opacity = '0';
            setTimeout(() => pageLoader.style.display = 'none', 500);
        }, 300);
    }
});

// Lenis smooth scroll init
const lenis = new Lenis({ duration: 1.2, easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)) });
function raf(time) {
    lenis.raf(time);
    requestAnimationFrame(raf);
}
requestAnimationFrame(raf);

// Custom Cursor Tracker
const cursorDot = document.getElementById('cursor-dot');
const cursorRing = document.getElementById('cursor-ring');
if (cursorDot && cursorRing) {
    window.addEventListener('mousemove', (e) => {
        cursorDot.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
        cursorRing.animate({ transform: `translate(${e.clientX - 16}px, ${e.clientY - 16}px)` }, { duration: 300, fill: 'forwards' });
    });
}

// Toast Helper
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    if (!toast) return;
    const toastMessage = toast.querySelector('.toast-message');
    const toastIcon = toast.querySelector('.toast-icon');

    toastMessage.textContent = message;
    toastIcon.textContent = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';
    toast.className = `toast toast-${type} show`;

    setTimeout(() => {
        toast.className = 'toast';
    }, 3500);
}

// ============================================
// Preset Pills & Slider Handlers
// ============================================
document.querySelectorAll('.preset-pill').forEach(pill => {
    pill.addEventListener('click', () => {
        document.querySelectorAll('.preset-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        activeStyle = pill.getAttribute('data-style');
        showToast(`Switched to ${pill.textContent.trim()} mode`, 'info');
    });
});

if (passesSlider && passesVal) {
    passesSlider.addEventListener('input', (e) => {
        activePasses = parseInt(e.target.value);
        passesVal.textContent = activePasses;
    });
}

// Tab Switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');

        btn.classList.add('active');
        const targetTab = btn.getAttribute('data-tab');
        const pane = document.getElementById(`tab-${targetTab}`);
        if (pane) pane.style.display = 'block';
    });
});

// ============================================
// Text Stats & Real-Time AI Analyzer
// ============================================
function updateInputStats() {
    const text = inputText.value;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    inputWordCount.textContent = words;
    inputCharCount.textContent = text.length;

    // Synchronize highlight overlay scroll & content
    if (inputHighlight) {
        inputHighlight.scrollTop = inputText.scrollTop;
    }

    // Debounced fast AI detection scanner call
    clearTimeout(analyzeDebounceTimer);
    if (text.trim().length > 30) {
        analyzeDebounceTimer = setTimeout(() => scanInputText(), 600);
    }
}

inputText.addEventListener('input', updateInputStats);
inputText.addEventListener('scroll', () => {
    if (inputHighlight) inputHighlight.scrollTop = inputText.scrollTop;
});

async function scanInputText() {
    const text = inputText.value.trim();
    if (!text) return;

    try {
        const response = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await response.json();
        if (data.success) {
            renderAIScorecard(data.ai_score);
            renderReadability(inputReadability, data.readability);
            highlightAISignals(text, data.signals);
        }
    } catch (err) {
        console.warn('Analysis engine offline or error:', err);
    }
}

if (scanBtn) scanBtn.addEventListener('click', scanInputText);

function renderAIScorecard(scores) {
    if (!scores) return;

    const prob = Math.round(scores.ai_probability);
    aiScoreVal.textContent = `${prob}%`;

    // Radial Gauge ring (314 is full circumference for r=50)
    if (gaugeFill) {
        const offset = 314 - (314 * (prob / 100));
        gaugeFill.style.strokeDashoffset = offset;
        gaugeFill.style.stroke = prob > 70 ? '#f87171' : prob > 35 ? '#fbbf24' : '#34d399';
    }

    // Detectors
    const dets = scores.detectors || {};
    setDetBar(detGptzero, detValGptzero, dets.gptzero);
    setDetBar(detTurnitin, detValTurnitin, dets.turnitin);
    setDetBar(detCopyleaks, detValCopyleaks, dets.copyleaks);
    setDetBar(detZerogpt, detValZerogpt, dets.zerogpt);

    // Metrics
    if (metricBurstiness) metricBurstiness.textContent = `${Math.round(scores.burstiness)}/100`;
    if (metricPerplexity) metricPerplexity.textContent = `${Math.round(scores.perplexity)}/100`;
}

function setDetBar(barEl, valEl, val) {
    if (!barEl || !valEl) return;
    const score = Math.round(val || 0);
    barEl.style.width = `${score}%`;
    barEl.style.backgroundColor = score > 70 ? '#f87171' : score > 35 ? '#fbbf24' : '#34d399';
    valEl.textContent = `${score}%`;
}

function renderReadability(targetEl, r) {
    if (!targetEl || !r) return;
    targetEl.innerHTML = `<span>Grade <strong>${r.grade_level}</strong> (Ease: ${r.reading_ease})</span>`;
}

function highlightAISignals(text, signals) {
    if (!inputHighlight || !signals || signals.length === 0) {
        if (inputHighlight) inputHighlight.innerHTML = '';
        return;
    }
    // Escape text for HTML overlay
    let html = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    signals.forEach(sig => {
        const regex = new RegExp(`\\b${escapeRegExp(sig.phrase)}\\b`, 'gi');
        html = html.replace(regex, `<span class="ai-flagged-word" title="Flagged AI Phrase: '${sig.phrase}' → Suggestions: ${sig.suggestions.join(', ')}">$&</span>`);
    });
    inputHighlight.innerHTML = html + '\n';
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// ============================================
// SSE Streaming Humanization Engine
// ============================================
async function humanizeTextStream() {
    originalText = inputText.value.trim();
    if (!originalText) {
        showToast('Please enter text to humanize', 'error');
        return;
    }

    // Set UI Loading State
    humanizeBtn.disabled = true;
    humanizeBtn.classList.add('loading');
    if (streamProgressBar) streamProgressBar.style.display = 'block';
    if (streamProgressFill) streamProgressFill.style.width = '5%';

    // Clear output panes
    outputText.innerHTML = '<p class="streaming-cursor">Initializing cross-model disruption stream...</p>';
    if (stepsList) stepsList.innerHTML = '';

    try {
        const response = await fetch(`${API_BASE}/api/humanize/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': sessionToken ? `Bearer ${sessionToken}` : ''
            },
            body: JSON.stringify({
                text: originalText,
                style: activeStyle,
                passes: activePasses
            })
        });

        if (response.status === 401) {
            const data = await response.json();
            showToast(data.error || 'Authentication required', 'error');
            openSignInModal();
            resetLoadingState();
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // keep last unfinished line

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const event = JSON.parse(line.replace('data: ', '').trim());
                        handleSSEEvent(event);
                    } catch (e) {
                        console.error('Error parsing SSE event line:', e);
                    }
                }
            }
        }
    } catch (err) {
        console.error('Streaming error:', err);
        showToast('Humanization failed or network error', 'error');
    } finally {
        resetLoadingState();
    }
}

function handleSSEEvent(ev) {
    if (ev.type === 'init') {
        renderAIScorecard(ev.pre_scores);
    } else if (ev.type === 'step_start') {
        if (streamProgressFill) streamProgressFill.style.width = `${ev.progress}%`;
        addStepLog(ev.step, 'active');
    } else if (ev.type === 'step_complete') {
        if (ev.current_text) {
            outputText.textContent = ev.current_text;
            humanizedText = ev.current_text;
            updateOutputStats(ev.current_text);
        }
        addStepLog(ev.step, 'complete');
    } else if (ev.type === 'complete') {
        if (streamProgressFill) streamProgressFill.style.width = '100%';
        humanizedText = ev.humanized;
        outputText.textContent = humanizedText;
        updateOutputStats(humanizedText);
        renderAIScorecard(ev.post_scores);
        renderReadability(outputReadability, ev.post_readability);
        buildDiffView(originalText, humanizedText);
        showToast('Text Humanized Successfully!', 'success');
    } else if (ev.type === 'error') {
        showToast(ev.error || 'An error occurred during pipeline execution', 'error');
    }
}

function resetLoadingState() {
    humanizeBtn.disabled = false;
    humanizeBtn.classList.remove('loading');
    setTimeout(() => {
        if (streamProgressBar) streamProgressBar.style.display = 'none';
    }, 1000);
}

function addStepLog(text, status) {
    if (!stepsList) return;
    let existing = Array.from(stepsList.children).find(li => li.getAttribute('data-step') === text);
    if (!existing) {
        existing = document.createElement('li');
        existing.className = 'step-item';
        existing.setAttribute('data-step', text);
        stepsList.appendChild(existing);
    }
    existing.className = `step-item ${status}`;
    const icon = status === 'complete' ? '✓' : '⚡';
    existing.innerHTML = `<span class="step-icon">${icon}</span> <span>${text}</span>`;
}

function updateOutputStats(text) {
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    outputWordCount.textContent = words;
    outputCharCount.textContent = text.length;
}

if (humanizeBtn) humanizeBtn.addEventListener('click', humanizeTextStream);

// Keyboard Shortcut Ctrl/Cmd + Enter
inputText.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        humanizeTextStream();
    }
});

// ============================================
// Myers Word-Level Diff Builder
// ============================================
function buildDiffView(oldStr, newStr) {
    if (!diffText) return;

    const oldWords = oldStr.split(/\s+/);
    const newWords = newStr.split(/\s+/);

    // Simple diff rendering using LCS / word comparison
    const diffHtml = computeWordDiff(oldWords, newWords);
    diffText.innerHTML = diffHtml;
}

function computeWordDiff(oldArr, newArr) {
    let i = 0, j = 0;
    let result = [];
    while (i < oldArr.length || j < newArr.length) {
        if (i < oldArr.length && j < newArr.length && oldArr[i] === newArr[j]) {
            result.push(escapeHtml(oldArr[i]));
            i++; j++;
        } else if (j < newArr.length && (!oldArr[i] || !oldArr.includes(newArr[j]))) {
            result.push(`<span class="diff-add">${escapeHtml(newArr[j])}</span>`);
            j++;
        } else if (i < oldArr.length) {
            result.push(`<span class="diff-del">${escapeHtml(oldArr[i])}</span>`);
            i++;
        }
    }
    return result.join(' ');
}

function escapeHtml(str) {
    return str ? str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;") : '';
}

// ============================================
// Clipboard & File Exporters
// ============================================
if (pasteBtn) {
    pasteBtn.addEventListener('click', async () => {
        try {
            const text = await navigator.clipboard.readText();
            inputText.value = text;
            updateInputStats();
            showToast('Pasted text from clipboard', 'success');
        } catch (err) {
            showToast('Unable to read clipboard', 'error');
        }
    });
}

if (clearBtn) {
    clearBtn.addEventListener('click', () => {
        inputText.value = '';
        updateInputStats();
        outputText.innerHTML = '<div class="placeholder-message"><p>Humanized text will stream here in real time...</p></div>';
        if (diffText) diffText.innerHTML = '<div class="placeholder-message"><p>Run humanization to see word-level Diff comparison</p></div>';
        showToast('Cleared workspace', 'info');
    });
}

if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
        const text = outputText.textContent;
        if (!text || text.includes('Humanized text will stream')) {
            showToast('No text available to copy', 'error');
            return;
        }
        await navigator.clipboard.writeText(text);
        showToast('Copied humanized text to clipboard!', 'success');
    });
}

if (downloadTxtBtn) {
    downloadTxtBtn.addEventListener('click', () => {
        const text = outputText.textContent;
        if (!text || text.includes('Humanized text will stream')) {
            showToast('No text available to download', 'error');
            return;
        }
        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `humanized-text-${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('Downloaded .txt file', 'success');
    });
}

if (downloadDocxBtn) {
    downloadDocxBtn.addEventListener('click', () => {
        const text = outputText.textContent;
        if (!text || text.includes('Humanized text will stream')) {
            showToast('No text available to download', 'error');
            return;
        }
        // Create formatted html blob with word doc mime type
        const htmlDoc = `
          <html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
          <head><meta charset='utf-8'><title>Humanized Document</title>
          <style>body { font-family: 'Calibri', sans-serif; font-size: 11pt; line-height: 1.5; color: #111; }</style>
          </head>
          <body>
          <h2>Humanized Document</h2>
          <p>${text.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br/>')}</p>
          </body></html>
        `;
        const blob = new Blob(['\ufeff' + htmlDoc], { type: 'application/msword' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `humanized-doc-${Date.now()}.docx`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('Downloaded .docx document', 'success');
    });
}

// ============================================
// Clerk Authentication & Guest Modal
// ============================================
const authBtn = document.getElementById('auth-btn');
const authOverlay = document.getElementById('auth-overlay');
const authCta = document.getElementById('auth-cta');

function openSignInModal() {
    if (window.Clerk) {
        window.Clerk.openSignIn();
    } else if (authOverlay) {
        authOverlay.style.display = 'flex';
    }
}

if (authBtn) authBtn.addEventListener('click', openSignInModal);
if (authCta) authCta.addEventListener('click', openSignInModal);

if (authOverlay) {
    authOverlay.addEventListener('click', (e) => {
        if (e.target === authOverlay) authOverlay.style.display = 'none';
    });
}
