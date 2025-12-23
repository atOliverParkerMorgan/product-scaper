(() => {
    window._isSimulatingClick = false;

    // --- 1. XPATH GENERATOR ---
    window._generateSelector = (el) => {
        if (!el || el.nodeType !== Node.ELEMENT_NODE) return '';
        if (el.closest('#pw-ui')) return '';

        const getElementIndex = (element) => {
            let index = 1;
            let sibling = element;
            while (sibling.previousElementSibling) {
                sibling = sibling.previousElementSibling;
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

    // --- 2. HELPER: GET CLASS SELECTOR ---
    window._getSameClassSelector = (xpath) => {
        try {
            const result = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
            const el = result.singleNodeValue;
            if (!el || !el.className) return null;

            const classes = Array.from(el.classList).filter(c => !c.startsWith('pw-'));
            if (classes.length === 0) return null;

            return classes.join('.');
        } catch (e) { return null; }
    }

    // --- 3. FLICKER-FREE HIGHLIGHTING ---
    window._updateHighlights = (selectedXpaths, predictedXpaths) => {
        const getEl = (xp) => {
            try {
                return document.evaluate(xp, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
            } catch (e) { return null; }
        };

        const selectedSet = new Set();
        selectedXpaths.forEach(xp => {
            const el = getEl(xp);
            if (el) selectedSet.add(el);
        });

        // Update Selected
        document.querySelectorAll('.pw-selected').forEach(el => {
            if (!selectedSet.has(el)) el.classList.remove('pw-selected');
        });
        selectedSet.forEach(el => {
            if (!el.classList.contains('pw-selected')) el.classList.add('pw-selected');
            el.classList.remove('pw-predicted');
        });

        // Update Predicted
        const predictedSet = new Set();
        if (predictedXpaths) {
            predictedXpaths.forEach(xp => {
                const el = getEl(xp);
                if (el && !selectedSet.has(el)) predictedSet.add(el);
            });
        }
        document.querySelectorAll('.pw-predicted').forEach(el => {
            if (!predictedSet.has(el)) el.classList.remove('pw-predicted');
        });
        predictedSet.forEach(el => {
            if (!el.classList.contains('pw-predicted')) el.classList.add('pw-predicted');
        });
    };

    // --- 4. INITIALIZE UI ---
    if (!document.getElementById('pw-ui')) {
        const ui = document.createElement('div');
        ui.id = 'pw-ui';
        ui.innerHTML = `
            <div id="pw-ui-header">
                <h2>Category: <span id="pw-category-name">...</span></h2>
                <div style="opacity:0.5">::</div>
            </div>
            <div id="pw-ui-body">
                <div class="pw-stat-row">
                    <span id="pw-step-counter">Step 1/1</span>
                    <span id="pw-count-badge">0 selected</span>
                </div>
                <div class="pw-stat-row" id="pw-predicted-row">
                    <span style="color: #888;">AI Found:</span>
                    <span id="pw-predicted-badge">0</span>
                </div>
                <div id="pw-selector-box">Hover an element...</div>
                
                <div class="pw-btn-group-top">
                    <button id="pw-btn-toggle-ai" class="pw-btn pw-btn-ai" title="Select/Unselect AI predicted elements">Select AI</button>
                    <button id="pw-btn-toggle-class" class="pw-btn pw-btn-smart" title="Select/Unselect elements with the same CSS class">Class Select</button>
                </div>

                <div class="pw-btn-group">
                    <button id="pw-btn-prev" class="pw-btn pw-btn-secondary" title="Go back to the previous category">Back</button>
                    <button id="pw-btn-next" class="pw-btn pw-btn-primary" title="Confirm selection and go to next category">Next</button>
                    <button id="pw-btn-done" class="pw-btn pw-btn-success pw-hidden" title="Save training data and finish">Finish</button>
                </div>
                <div class="pw-hint">Right Click: Interact | Left Click: Select</div>
            </div>
        `;
        document.body.appendChild(ui);

        // Bind Actions
        document.getElementById('pw-btn-prev').onclick = () => window._action = 'prev';
        document.getElementById('pw-btn-next').onclick = () => window._action = 'next';
        document.getElementById('pw-btn-done').onclick = () => window._action = 'done';
        document.getElementById('pw-btn-toggle-ai').onclick = () => window._action = 'toggle_ai';
        document.getElementById('pw-btn-toggle-class').onclick = () => window._action = 'toggle_class';

        // Dragging Logic
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

    // --- 5. INTERACTIONS ---
    document.addEventListener('mouseover', (e) => {
        if (e.target.closest('#pw-ui')) return;
        document.querySelectorAll('.pw-hover').forEach(el => el.classList.remove('pw-hover'));
        e.target.classList.add('pw-hover');
        const box = document.getElementById('pw-selector-box');
        const sel = window._generateSelector(e.target);
        if (sel) box.innerText = sel;
    });

    document.addEventListener('mouseout', (e) => {
        if (e.target) e.target.classList.remove('pw-hover');
    });

    window._clickedSelector = null;
    document.addEventListener('click', (e) => {
        if (e.target.closest('#pw-ui')) return;

        if (window._isSimulatingClick) {
            const isLink = e.target.closest('a, button[type="submit"]');
            if (isLink) e.preventDefault();
            return;
        }

        if (e.button === 0) {
            e.preventDefault();
            e.stopImmediatePropagation();
            window._clickedSelector = window._generateSelector(e.target);
        }
    }, true);

    document.addEventListener('contextmenu', (e) => {
        if (e.target.closest('#pw-ui')) return;
        e.preventDefault();
        window._isSimulatingClick = true;
        try { e.target.click(); } catch (err) { }
        window._isSimulatingClick = false;
    }, true);

    window._keyAction = null;
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            const k = e.key.toLowerCase();
            if (k === 'z') {
                e.preventDefault();
                window._keyAction = e.shiftKey ? 'redo' : 'undo';
            } else if (k === 'y') {
                e.preventDefault();
                window._keyAction = 'redo';
            }
        }
    });
})();