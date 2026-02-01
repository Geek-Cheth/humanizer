/**
 * Text Humanizer - Frontend Logic
 * Handles UI interactions and API communication
 */

// API Configuration - use relative path for production, localhost for development
const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const API_BASE = isLocalhost ? 'http://localhost:5000' : '';

// DOM Elements
const inputText = document.getElementById('input-text');
const outputText = document.getElementById('output-text');
const humanizeBtn = document.getElementById('humanize-btn');
const pasteBtn = document.getElementById('paste-btn');
const clearBtn = document.getElementById('clear-btn');
const copyBtn = document.getElementById('copy-btn');
const intensitySlider = document.getElementById('intensity');
const processingInfo = document.getElementById('processing-info');
const stepsList = document.getElementById('steps-list');

// Word count elements
const inputWordCount = document.getElementById('input-word-count');
const inputCharCount = document.getElementById('input-char-count');
const outputWordCount = document.getElementById('output-word-count');
const outputCharCount = document.getElementById('output-char-count');

// Toast element
const toast = document.getElementById('toast');

/**
 * Show toast notification
 */
function showToast(message, type = 'success') {
    const toastIcon = toast.querySelector('.toast-icon');
    const toastMessage = toast.querySelector('.toast-message');

    toastIcon.textContent = type === 'success' ? '✓' : '✕';
    toastMessage.textContent = message;

    toast.className = `toast ${type}`;
    toast.classList.add('show');

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

/**
 * Count words in text
 */
function countWords(text) {
    return text.trim() ? text.trim().split(/\s+/).length : 0;
}

/**
 * Update word counts
 */
function updateInputCounts() {
    const text = inputText.value;
    inputWordCount.textContent = countWords(text);
    inputCharCount.textContent = text.length;
}

function updateOutputCounts(text) {
    outputWordCount.textContent = countWords(text);
    outputCharCount.textContent = text.length;
}

/**
 * Get current settings from UI
 */
function getSettings() {
    // Get selected mode
    const modeRadio = document.querySelector('input[name="mode"]:checked');
    const mode = modeRadio ? modeRadio.value : 'balanced';

    // Get intensity
    const intensityMap = ['light', 'medium', 'heavy'];
    const intensity = intensityMap[parseInt(intensitySlider.value)];

    // Get technique options
    const options = {
        synonyms: document.getElementById('opt-synonyms').checked,
        contractions: document.getElementById('opt-contractions').checked,
        vary_length: document.getElementById('opt-vary-length').checked,
        informal: document.getElementById('opt-informal').checked,
        casual_starters: document.getElementById('opt-starters').checked,
        ai_polish: document.getElementById('opt-polish').checked
    };

    return { mode, intensity, options };
}

/**
 * Set loading state
 */
function setLoading(isLoading) {
    if (isLoading) {
        humanizeBtn.classList.add('loading');
        humanizeBtn.disabled = true;
    } else {
        humanizeBtn.classList.remove('loading');
        humanizeBtn.disabled = false;
    }
}

/**
 * Display processing steps
 */
function displaySteps(steps) {
    stepsList.innerHTML = '';
    steps.forEach(step => {
        const li = document.createElement('li');
        li.textContent = step;
        stepsList.appendChild(li);
    });
    processingInfo.style.display = 'block';
}

/**
 * Humanize the input text
 */
async function humanize() {
    const text = inputText.value.trim();

    if (!text) {
        showToast('Please enter some text to humanize', 'error');
        return;
    }

    setLoading(true);
    processingInfo.style.display = 'none';

    try {
        const settings = getSettings();

        const response = await fetch(`${API_BASE}/api/humanize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                mode: settings.mode,
                intensity: settings.intensity,
                options: settings.options
            })
        });

        const data = await response.json();

        if (data.success) {
            // Display humanized text
            outputText.innerHTML = '';
            outputText.textContent = data.humanized;
            updateOutputCounts(data.humanized);

            // Display processing steps
            if (data.steps && data.steps.length > 0) {
                displaySteps(data.steps);
            }

            showToast('Text humanized successfully!', 'success');
        } else {
            throw new Error(data.error || 'Humanization failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast(error.message || 'Failed to humanize text', 'error');
    } finally {
        setLoading(false);
    }
}

/**
 * Paste from clipboard
 */
async function pasteFromClipboard() {
    try {
        const text = await navigator.clipboard.readText();
        inputText.value = text;
        updateInputCounts();
        showToast('Text pasted from clipboard', 'success');
    } catch (error) {
        showToast('Failed to paste from clipboard', 'error');
    }
}

/**
 * Clear input text
 */
function clearInput() {
    inputText.value = '';
    updateInputCounts();
}

/**
 * Copy output to clipboard
 */
async function copyToClipboard() {
    const text = outputText.textContent;

    if (!text || text.includes('Your humanized text will appear here')) {
        showToast('No text to copy', 'error');
        return;
    }

    try {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard!', 'success');
    } catch (error) {
        showToast('Failed to copy to clipboard', 'error');
    }
}

// Event Listeners
inputText.addEventListener('input', updateInputCounts);
humanizeBtn.addEventListener('click', humanize);
pasteBtn.addEventListener('click', pasteFromClipboard);
clearBtn.addEventListener('click', clearInput);
copyBtn.addEventListener('click', copyToClipboard);

// Keyboard shortcut: Ctrl/Cmd + Enter to humanize
inputText.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        humanize();
    }
});

// Initialize counts
updateInputCounts();

// Check API health on load
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        if (!response.ok) {
            console.warn('API server not responding');
        }
    } catch (error) {
        console.warn('API server not available. Make sure to run: python app.py');
    }
}

checkApiHealth();

// Hide page loader when everything is ready
window.addEventListener('load', () => {
    const loader = document.getElementById('page-loader');
    setTimeout(() => {
        loader.classList.add('hidden');
        setTimeout(() => {
            loader.style.display = 'none';
        }, 500);
    }, 800);
});
