/**
 * Text Humanizer - Premium UI/UX with Clerk Authentication
 * Custom cursor, magnetic effects, smooth scroll, reveal animations, auth
 */

// ============================================
// Configuration
// ============================================
const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const API_BASE = isLocalhost ? 'http://localhost:5000' : '';

// ============================================
// Clerk Authentication State
// ============================================
let clerkLoaded = false;
let isAuthenticated = false;
let sessionToken = null;

// Clerk appearance configuration for dark theme
const clerkAppearance = {
    baseTheme: undefined,
    layout: {
        socialButtonsPlacement: 'top',
        socialButtonsVariant: 'blockButton',
        shimmer: true
    },
    variables: {
        colorPrimary: '#6366f1',
        colorBackground: '#0d0d12',
        colorInputBackground: '#1c1c26',
        colorInputText: '#f5f5f7',
        colorText: '#f5f5f7',
        colorTextSecondary: '#a0a0b0',
        colorTextOnPrimaryBackground: '#ffffff',
        colorDanger: '#f87171',
        colorSuccess: '#34d399',
        colorWarning: '#fbbf24',
        colorNeutral: '#6b6b7b',
        colorShimmer: 'rgba(99, 102, 241, 0.1)',
        borderRadius: '12px',
        fontFamily: 'Outfit, sans-serif',
        fontFamilyButtons: 'Syne, sans-serif',
        fontSmoothing: 'antialiased',
        fontSize: '15px',
        spacingUnit: '16px'
    },
    elements: {
        // Modal backdrop
        modalBackdrop: {
            backgroundColor: 'rgba(8, 8, 12, 0.85)',
            backdropFilter: 'blur(8px)'
        },
        // Root container
        rootBox: {
            boxShadow: '0 25px 80px -12px rgba(0, 0, 0, 0.8)'
        },
        // Main card
        card: {
            backgroundColor: '#0d0d12',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '20px',
            boxShadow: '0 25px 80px -12px rgba(0, 0, 0, 0.8), 0 0 80px rgba(99, 102, 241, 0.15)',
            padding: '32px'
        },
        // Header
        headerTitle: {
            color: '#ffffff',
            fontFamily: 'Syne, sans-serif',
            fontWeight: '700',
            fontSize: '1.5rem'
        },
        headerSubtitle: {
            color: '#a0a0b0',
            fontSize: '0.95rem'
        },
        // Social buttons (Google, GitHub, etc.)
        socialButtonsBlockButton: {
            backgroundColor: '#1c1c26',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            color: '#f5f5f7',
            borderRadius: '12px',
            padding: '14px 20px',
            fontSize: '0.95rem',
            fontWeight: '500',
            transition: 'all 0.2s ease',
            '&:hover': {
                backgroundColor: '#242432',
                borderColor: '#6366f1',
                transform: 'translateY(-1px)',
                boxShadow: '0 4px 12px rgba(99, 102, 241, 0.2)'
            }
        },
        socialButtonsBlockButtonText: {
            color: '#f5f5f7',
            fontWeight: '500'
        },
        // Social button icons
        socialButtonsIconButton: {
            backgroundColor: '#1c1c26',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            borderRadius: '12px',
            '&:hover': {
                backgroundColor: '#242432',
                borderColor: '#6366f1'
            }
        },
        socialButtonsProviderIcon: {
            // White background makes colored icons visible
            backgroundColor: '#ffffff',
            borderRadius: '6px',
            padding: '3px'
        },
        // Divider
        dividerLine: {
            backgroundColor: 'rgba(255, 255, 255, 0.1)'
        },
        dividerText: {
            color: '#6b6b7b',
            fontSize: '0.8rem',
            textTransform: 'uppercase',
            letterSpacing: '0.1em'
        },
        // Form labels
        formFieldLabel: {
            color: '#a0a0b0',
            fontSize: '0.75rem',
            fontWeight: '600',
            textTransform: 'uppercase',
            letterSpacing: '0.12em',
            marginBottom: '8px'
        },
        // Input fields
        formFieldInput: {
            backgroundColor: '#1c1c26',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            color: '#f5f5f7',
            borderRadius: '10px',
            padding: '14px 16px',
            fontSize: '1rem',
            transition: 'all 0.2s ease',
            '&:focus': {
                borderColor: '#6366f1',
                boxShadow: '0 0 0 4px rgba(99, 102, 241, 0.15)',
                backgroundColor: '#242432'
            },
            '&::placeholder': {
                color: '#52525e'
            }
        },
        // Primary button (Continue, Sign in)
        formButtonPrimary: {
            background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
            color: '#ffffff',
            borderRadius: '9999px',
            fontFamily: 'Syne, sans-serif',
            fontWeight: '600',
            fontSize: '1rem',
            padding: '14px 28px',
            border: 'none',
            boxShadow: '0 4px 20px rgba(99, 102, 241, 0.3)',
            transition: 'all 0.2s ease',
            '&:hover': {
                background: 'linear-gradient(135deg, #818cf8 0%, #a78bfa 100%)',
                transform: 'translateY(-2px)',
                boxShadow: '0 8px 30px rgba(99, 102, 241, 0.4)'
            },
            '&:active': {
                transform: 'translateY(0)'
            }
        },
        // Footer links
        footerActionLink: {
            color: '#6366f1',
            fontWeight: '500',
            transition: 'color 0.2s ease',
            '&:hover': {
                color: '#818cf8'
            }
        },
        footerActionText: {
            color: '#6b6b7b'
        },
        // Identity preview
        identityPreviewText: {
            color: '#f5f5f7'
        },
        identityPreviewEditButton: {
            color: '#6366f1',
            '&:hover': {
                color: '#818cf8'
            }
        },
        // Form actions (Forgot password, etc.)
        formFieldAction: {
            color: '#6366f1',
            fontSize: '0.85rem',
            '&:hover': {
                color: '#818cf8'
            }
        },
        // Alerts
        alert: {
            backgroundColor: 'rgba(248, 113, 113, 0.1)',
            border: '1px solid rgba(248, 113, 113, 0.3)',
            borderRadius: '10px'
        },
        alertText: {
            color: '#f5f5f7'
        },
        // Close button
        modalCloseButton: {
            color: '#6b6b7b',
            transition: 'color 0.2s ease',
            '&:hover': {
                color: '#f5f5f7'
            }
        },
        // OTP inputs
        otpCodeFieldInput: {
            backgroundColor: '#1c1c26',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            color: '#f5f5f7',
            borderRadius: '10px',
            '&:focus': {
                borderColor: '#6366f1',
                boxShadow: '0 0 0 4px rgba(99, 102, 241, 0.15)'
            }
        },
        // Avatar
        avatarBox: {
            border: '2px solid #6366f1'
        },
        // Badge
        badge: {
            backgroundColor: 'rgba(99, 102, 241, 0.2)',
            color: '#a5b4fc'
        },
        // User button
        userButtonPopoverCard: {
            backgroundColor: '#0d0d12',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '16px'
        },
        userButtonPopoverActionButton: {
            color: '#f5f5f7',
            '&:hover': {
                backgroundColor: '#1c1c26'
            }
        },
        userButtonPopoverActionButtonIcon: {
            color: '#a0a0b0'
        },
        userButtonPopoverFooter: {
            borderTop: '1px solid rgba(255, 255, 255, 0.08)'
        }
    }
};

// Wait for Clerk to load
async function initClerk() {
    try {
        // Wait for Clerk to be available
        while (!window.Clerk) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        await window.Clerk.load({
            appearance: clerkAppearance
        });
        clerkLoaded = true;

        // Check initial auth state
        updateAuthState();

        // Listen for auth changes
        window.Clerk.addListener(updateAuthState);

        console.log('Clerk initialized with dark theme');
    } catch (error) {
        console.error('Error initializing Clerk:', error);
    }
}

async function updateAuthState() {
    if (!clerkLoaded) return;

    const user = window.Clerk.user;
    isAuthenticated = !!user;

    // Get session token for API calls
    if (isAuthenticated) {
        try {
            const session = window.Clerk.session;
            sessionToken = await session.getToken();
        } catch (e) {
            sessionToken = null;
        }
    } else {
        sessionToken = null;
    }

    // Update UI
    updateAuthUI();
}

function updateAuthUI() {
    const authBtn = document.getElementById('auth-btn');
    const authText = document.getElementById('auth-text');
    const authOverlay = document.getElementById('auth-overlay');

    if (isAuthenticated && window.Clerk.user) {
        // Show user info
        const user = window.Clerk.user;
        const firstName = user.firstName || 'User';

        authBtn.classList.add('authenticated');
        authText.textContent = firstName;

        // Hide auth overlay
        authOverlay.classList.remove('visible');

        // Update status
        updateApiStatus('Ready', 'success');
    } else {
        // Show sign in
        authBtn.classList.remove('authenticated');
        authText.textContent = 'Sign In';

        // Show auth overlay
        authOverlay.classList.add('visible');

        // Update status
        updateApiStatus('Sign In Required', 'warning');
    }
}

function openSignIn() {
    if (clerkLoaded && window.Clerk) {
        window.Clerk.openSignIn({
            appearance: clerkAppearance,
            afterSignInUrl: window.location.href,
            afterSignUpUrl: window.location.href
        });
    }
}

function handleAuthClick() {
    if (!clerkLoaded) return;

    if (isAuthenticated) {
        // Open user profile with dark theme
        window.Clerk.openUserProfile({
            appearance: clerkAppearance
        });
    } else {
        openSignIn();
    }
}

// Initialize Clerk
initClerk();

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
// Disable Lenis for Clerk modal elements
// ============================================
// Lenis should ignore scroll events on Clerk modals
document.addEventListener('wheel', (e) => {
    const clerkElement = e.target.closest('.cl-rootBox, .cl-modalBackdrop, .cl-scrollBox, .cl-pageScrollBox, .cl-userProfile-root');
    if (clerkElement) {
        e.stopPropagation();
    }
}, { passive: false, capture: true });

// Lock body scroll when Clerk modal is open
function checkForClerkModal() {
    const clerkModal = document.querySelector('.cl-modalBackdrop');
    if (clerkModal) {
        document.body.style.overflow = 'hidden';
    } else {
        document.body.style.overflow = '';
    }
}

// Watch for Clerk modal changes
const clerkObserver = new MutationObserver(checkForClerkModal);
clerkObserver.observe(document.body, { childList: true, subtree: false });

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

// Auth elements
const authBtn = document.getElementById('auth-btn');
const authCta = document.getElementById('auth-cta');
const authOverlay = document.getElementById('auth-overlay');

// Word count elements
const inputWordCount = document.getElementById('input-word-count');
const inputCharCount = document.getElementById('input-char-count');
const outputWordCount = document.getElementById('output-word-count');
const outputCharCount = document.getElementById('output-char-count');

// Toast element
const toast = document.getElementById('toast');

// ============================================
// Auth Event Listeners
// ============================================
if (authBtn) {
    authBtn.addEventListener('click', handleAuthClick);
}

if (authCta) {
    authCta.addEventListener('click', openSignIn);
}

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
// Humanize Function (with Auth)
// ============================================
async function humanize() {
    // Check authentication first
    if (!isAuthenticated) {
        showToast('Please sign in to use the humanizer', 'error');
        openSignIn();
        return;
    }

    // Refresh token if needed
    if (window.Clerk && window.Clerk.session) {
        try {
            sessionToken = await window.Clerk.session.getToken();
        } catch (e) {
            showToast('Session expired, please sign in again', 'error');
            openSignIn();
            return;
        }
    }

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
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${sessionToken}`
            },
            body: JSON.stringify({
                text: text,
                mode: settings.mode,
                intensity: settings.intensity,
                options: settings.options
            })
        });

        const data = await response.json();

        if (response.status === 401) {
            // Auth error
            showToast('Authentication required. Please sign in.', 'error');
            isAuthenticated = false;
            updateAuthUI();
            openSignIn();
            return;
        }

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
            if (isAuthenticated) {
                updateApiStatus('Ready', 'success');
            }
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
// Textarea Focus State
// ============================================
if (inputText) {
    inputText.addEventListener('focus', () => {
        inputText.closest('.glass-card')?.classList.add('focused');
    });

    inputText.addEventListener('blur', () => {
        inputText.closest('.glass-card')?.classList.remove('focused');
    });
}
