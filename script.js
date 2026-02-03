/**
 * Text Humanizer - Premium UI/UX
 * Custom cursor, magnetic effects, smooth scroll, reveal animations
 */

// ============================================
// Configuration
// ============================================
const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const API_BASE = isLocalhost ? 'http://localhost:5000' : '';

// ============================================
// Lenis Smooth Scroll
// ============================================
const lenis = new Lenis({
    duration: 1.2,
    easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
    orientation: 'vertical',
    gestureOrientation: 'vertical',
    smoothWheel: true,
    wheelMultiplier: 1,
    touchMultiplier: 2,
    infinite: false,
});

function raf(time) {
    lenis.raf(time);
    requestAnimationFrame(raf);
}
requestAnimationFrame(raf);

// ============================================
// Custom Cursor
// ============================================
const cursorDot = document.getElementById('cursor-dot');
const cursorRing = document.getElementById('cursor-ring');

let mouseX = 0, mouseY = 0;
let ringX = 0, ringY = 0;
let dotX = 0, dotY = 0;

// Check if touch device
const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

if (!isTouchDevice && cursorDot && cursorRing) {
    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });

    document.addEventListener('mousedown', () => {
        cursorDot.classList.add('clicking');
        cursorRing.classList.add('clicking');
    });

    document.addEventListener('mouseup', () => {
        cursorDot.classList.remove('clicking');
        cursorRing.classList.remove('clicking');
    });

    // Cursor animation loop
    function animateCursor() {
        // Dot follows immediately
        dotX += (mouseX - dotX) * 0.5;
        dotY += (mouseY - dotY) * 0.5;

        // Ring follows with delay
        ringX += (mouseX - ringX) * 0.15;
        ringY += (mouseY - ringY) * 0.15;

        cursorDot.style.left = `${dotX}px`;
        cursorDot.style.top = `${dotY}px`;
        cursorRing.style.left = `${ringX}px`;
        cursorRing.style.top = `${ringY}px`;

        requestAnimationFrame(animateCursor);
    }
    animateCursor();

    // Hover effects
    const hoverElements = document.querySelectorAll('button, a, input, textarea, label, .magnetic-btn');
    hoverElements.forEach(el => {
        el.addEventListener('mouseenter', () => {
            cursorDot.classList.add('hovering');
            cursorRing.classList.add('hovering');
        });
        el.addEventListener('mouseleave', () => {
            cursorDot.classList.remove('hovering');
            cursorRing.classList.remove('hovering');
        });
    });
}

// ============================================
// Magnetic Button Effect
// ============================================
const magneticButtons = document.querySelectorAll('.magnetic-btn');

magneticButtons.forEach(btn => {
    btn.addEventListener('mousemove', (e) => {
        const rect = btn.getBoundingClientRect();
        const x = e.clientX - rect.left - rect.width / 2;
        const y = e.clientY - rect.top - rect.height / 2;

        btn.style.transform = `translate(${x * 0.2}px, ${y * 0.2}px)`;
    });

    btn.addEventListener('mouseleave', () => {
        btn.style.transform = '';
    });
});

// ============================================
// Reveal Animations
// ============================================
const revealElements = document.querySelectorAll('.reveal-element, .reveal-text');

const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('revealed');
        }
    });
}, {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
});

revealElements.forEach(el => revealObserver.observe(el));

// Initial reveal for above-fold elements
setTimeout(() => {
    document.querySelectorAll('.header .reveal-element, .header .reveal-text').forEach(el => {
        el.classList.add('revealed');
    });
}, 100);

// ============================================
// Page Loader
// ============================================
window.addEventListener('load', () => {
    const loader = document.getElementById('page-loader');
    setTimeout(() => {
        loader.classList.add('hidden');

        // Trigger initial reveals
        setTimeout(() => {
            revealElements.forEach(el => {
                if (el.getBoundingClientRect().top < window.innerHeight) {
                    el.classList.add('revealed');
                }
            });
        }, 200);
    }, 600);
});

// ============================================
// DOM Elements
// ============================================
const inputText = document.getElementById('input-text');
const outputText = document.getElementById('output-text');
const humanizeBtn = document.getElementById('humanize-btn');
const pasteBtn = document.getElementById('paste-btn');
const clearBtn = document.getElementById('clear-btn');
const copyBtn = document.getElementById('copy-btn');
const intensitySlider = document.getElementById('intensity');
const processingInfo = document.getElementById('processing-info');
const stepsList = document.getElementById('steps-list');
const apiStatus = document.getElementById('api-status');

// Word count elements
const inputWordCount = document.getElementById('input-word-count');
const inputCharCount = document.getElementById('input-char-count');
const outputWordCount = document.getElementById('output-word-count');
const outputCharCount = document.getElementById('output-char-count');

// Toast element
const toast = document.getElementById('toast');

// ============================================
// Toast Notification
// ============================================
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

// ============================================
// Word Count
// ============================================
function countWords(text) {
    return text.trim() ? text.trim().split(/\s+/).length : 0;
}

function updateInputCounts() {
    const text = inputText.value;
    inputWordCount.textContent = countWords(text);
    inputCharCount.textContent = text.length;
}

function updateOutputCounts(text) {
    outputWordCount.textContent = countWords(text);
    outputCharCount.textContent = text.length;
}

// ============================================
// Settings
// ============================================
function getSettings() {
    const modeRadio = document.querySelector('input[name="mode"]:checked');
    const mode = modeRadio ? modeRadio.value : 'balanced';

    const intensityMap = ['light', 'medium', 'heavy'];
    const intensity = intensityMap[parseInt(intensitySlider.value)];

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

// ============================================
// Loading State
// ============================================
function setLoading(isLoading) {
    if (isLoading) {
        humanizeBtn.classList.add('loading');
        humanizeBtn.disabled = true;
        updateApiStatus('Processing...', 'warning');
    } else {
        humanizeBtn.classList.remove('loading');
        humanizeBtn.disabled = false;
        updateApiStatus('Ready', 'success');
    }
}

function updateApiStatus(text, status) {
    if (apiStatus) {
        const statusDot = apiStatus.querySelector('.status-dot');
        const statusText = apiStatus.querySelector('.status-text');

        statusText.textContent = text;
        statusDot.style.background = status === 'success' ? 'var(--success)' :
            status === 'warning' ? 'var(--warning)' :
                'var(--error)';
    }
}

// ============================================
// Processing Steps Display
// ============================================
function displaySteps(steps) {
    stepsList.innerHTML = '';
    steps.forEach(step => {
        const li = document.createElement('li');
        li.textContent = step;
        stepsList.appendChild(li);
    });
    processingInfo.style.display = 'block';
}

// ============================================
// Humanize Function
// ============================================
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
            // Clear placeholder and show text
            outputText.innerHTML = '';
            outputText.textContent = data.humanized;
            updateOutputCounts(data.humanized);

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
        updateApiStatus('Error', 'error');
    } finally {
        setLoading(false);
    }
}

// ============================================
// Clipboard Functions
// ============================================
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

function clearInput() {
    inputText.value = '';
    updateInputCounts();

    // Reset output
    outputText.innerHTML = `
        <div class="placeholder-message">
            <div class="placeholder-visual">
                <span class="placeholder-icon">→</span>
            </div>
            <p>Humanized text appears here</p>
        </div>
    `;
    updateOutputCounts('');
    processingInfo.style.display = 'none';
}

async function copyToClipboard() {
    const text = outputText.textContent;

    if (!text || text.includes('Humanized text appears here')) {
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

// ============================================
// Event Listeners
// ============================================
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

// ============================================
// API Health Check
// ============================================
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        if (response.ok) {
            updateApiStatus('Ready', 'success');
        } else {
            updateApiStatus('Offline', 'error');
        }
    } catch (error) {
        updateApiStatus('Offline', 'error');
        console.warn('API server not available');
    }
}

checkApiHealth();

// ============================================
// Floating Dock Animation
// ============================================
const floatingDock = document.getElementById('floating-dock');

if (floatingDock) {
    let lastScrollY = 0;

    lenis.on('scroll', ({ scroll }) => {
        if (scroll > lastScrollY && scroll > 100) {
            floatingDock.style.transform = 'translateX(-50%) translateY(100px)';
        } else {
            floatingDock.style.transform = 'translateX(-50%) translateY(0)';
        }
        lastScrollY = scroll;
    });
}

// ============================================
// Textarea Auto-Resize (Optional Enhancement)
// ============================================
if (inputText) {
    inputText.addEventListener('focus', () => {
        inputText.closest('.glass-card')?.classList.add('focused');
    });

    inputText.addEventListener('blur', () => {
        inputText.closest('.glass-card')?.classList.remove('focused');
    });
}
