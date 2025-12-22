/* ui/core.js */
(() => {
    // Flag to track simulated interactions vs real user clicks
    window._isSimulatingClick = false;

    // --- 1. ROBUST XPATH GENERATOR ---
    window._generateSelector = (el) => {
        if (!el || el.nodeType !== Node.ELEMENT_NODE) return '';

        const getElementIndex = (element) => {
            let index = 1;
            let sibling = element;
            while (sibling.previousElementSibling) {
                sibling = sibling.previousElementSibling;
                // Use localName to be safe with SVG/XML casing
                if (sibling.nodeType === Node.ELEMENT_NODE && sibling.localName === element.localName) {
                    index++;
                }
            }
            return index;
        };

        const parts = [];
        let current = el;
        while (current && current.nodeType === Node.ELEMENT_NODE) {
            const tagName = current.localName.toLowerCase();
            if (tagName === 'html') {
                parts.unshift('html[1]');
                break;
            }
            const index = getElementIndex(current);
            parts.unshift(`${tagName}[${index}]`);
            current = current.parentNode;
        }
        return '/' + parts.join('/');
    };

    // --- 2. INITIALIZE UI ---
    if (!document.getElementById('pw-ui')) {
        const ui = document.createElement('div');
        ui.id = 'pw-ui';
        // Includes data-testid for Playwright Testing
        ui.innerHTML = `
            <div id="pw-ui-header" data-testid="pw-ui-header">
                <h2>Category: <span id="pw-category-name" data-testid="pw-category-name">...</span></h2>
                <div style="opacity:0.5">::</div>
            </div>
            <div id="pw-ui-body" data-testid="pw-ui-body">
                <div class="pw-stat-row">
                    <span id="pw-step-counter" data-testid="pw-step-counter">Step 1/1</span>
                    <span id="pw-count-badge" data-testid="pw-count-badge">0 selected</span>
                </div>
                <div class="pw-stat-row" id="pw-predicted-row">
                    <span style="color: #888;">Predicted:</span>
                    <span id="pw-predicted-badge" data-testid="pw-predicted-badge">0 found</span>
                </div>
                <div id="pw-selector-box" data-testid="pw-selector-box">Hover an element...</div>
                <div class="pw-btn-group-top">
                    <button id="pw-btn-select-predicted" class="pw-btn pw-btn-select-predicted" data-testid="pw-btn-select-predicted">Select Predictions</button>
                </div>
                <div class="pw-btn-group">
                    <button id="pw-btn-prev" class="pw-btn pw-btn-secondary" data-testid="pw-btn-prev">Back</button>
                    <button id="pw-btn-next" class="pw-btn pw-btn-primary" data-testid="pw-btn-next">Next Category</button>
                    <button id="pw-btn-done" class="pw-btn pw-btn-success pw-hidden" data-testid="pw-btn-done">Finish & Save</button>
                </div>
                <div class="pw-hint">Right-Click to interact (close modals)<br>Ctrl+Shift+Z to Redo</div>
            </div>
        `;
        document.body.appendChild(ui);

        // Bind UI Actions
        document.getElementById('pw-btn-prev').onclick = () => window._action = 'prev';
        document.getElementById('pw-btn-next').onclick = () => window._action = 'next';
        document.getElementById('pw-btn-done').onclick = () => window._action = 'done';
        document.getElementById('pw-btn-select-predicted').onclick = () => window._action = 'select_predicted';

        // Draggable Header Logic
        const header = document.getElementById('pw-ui-header');
        let isDragging = false;
        let startX, startY, initialX = 0, initialY = 0;
        let xOffset = 0, yOffset = 0;

        header.onmousedown = (e) => {
            if (e.button !== 0) return;
            initialX = e.clientX - xOffset;
            initialY = e.clientY - yOffset;
            isDragging = true;
            header.style.cursor = 'grabbing';
        };

        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            e.preventDefault();
            xOffset = e.clientX - initialX;
            yOffset = e.clientY - initialY;
            ui.style.transform = `translate3d(${xOffset}px, ${yOffset}px, 0)`;
        });

        window.addEventListener('mouseup', () => {
            isDragging = false;
            header.style.cursor = 'grab';
        });
    }

    // --- 3. GLOBAL CLICK HANDLER (CAPTURE PHASE) ---
    // Use capture=true to intercept events before page frameworks
    window._clickedSelector = null;

    document.addEventListener('click', (e) => {
        // A. Ignore clicks inside our UI
        if (e.target.closest('#pw-ui')) return;

        const isLink = e.target.closest('a, button[type="submit"]');

        // B. Handle Simulated Right-Click Interactions
        if (window._isSimulatingClick) {
            // If it's a link, PREVENT navigation, but allow propagation 
            // so the page's JS handlers (like closing a modal) still run.
            if (isLink) {
                e.preventDefault();
            }
            return; // Let it bubble
        }

        // C. Handle Normal Left-Click (Selection)
        if (e.button === 0) {
            // Always prevent default navigation/action on left click
            e.preventDefault();
            e.stopPropagation(); // Stop bubbling (don't trigger page JS)

            window._clickedSelector = window._generateSelector(e.target);
        }
    }, true); // <--- Capture Phase is Critical

    // --- 4. HOVER EFFECTS ---
    document.addEventListener('mouseover', (e) => {
        if (e.target.closest('#pw-ui')) return;
        document.querySelectorAll('.pw-hover').forEach(el => el.classList.remove('pw-hover'));
        e.target.classList.add('pw-hover');

        const box = document.getElementById('pw-selector-box');
        if (box) {
            box.innerText = window._generateSelector(e.target);
        }
    });

    document.addEventListener('mouseout', (e) => {
        if (e.target) e.target.classList.remove('pw-hover');
    });

    // --- 5. RIGHT CLICK (INTERACT) ---
    document.addEventListener('contextmenu', (e) => {
        if (e.target.closest('#pw-ui')) return;

        e.preventDefault(); // Stop Browser Context Menu

        // Trigger a "safe" click that our Capture Listener will handle
        window._isSimulatingClick = true;
        try {
            e.target.click();
        } catch (err) {
            console.log("Interaction failed", err);
        }
        window._isSimulatingClick = false;

    }, true);

    // --- 6. KEYBOARD SHORTCUTS ---
    window._keyAction = null;
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            const k = e.key.toLowerCase();
            if (k === 'z') {
                e.preventDefault();
                window._keyAction = e.shiftKey ? 'redo' : 'undo';
            }
            else if (k === 'y') {
                e.preventDefault();
                window._keyAction = 'redo';
            }
        }
    });
})();