(() => {
    // Flag to avoid selection loop during simulated clicks
    window._isSimulatingClick = false;

    // --- 1. PREVENT NAVIGATION ---
    // Capture clicks on 'a' tags and prevent navigation while allowing JS handlers to run
    document.addEventListener('click', (e) => {
        const link = e.target.closest('a');
        if (link) {
            // Prevent the browser from following the href
            e.preventDefault();
        }
    }, false);

    // --- 2. XPATH GENERATION (FULL PATH) ---
    window._generateSelector = (el) => {
        if (!el || el.nodeType !== Node.ELEMENT_NODE) return '';
        
        const getElementIndex = (element) => {
            let index = 1;
            let sibling = element;
            // Count backwards to find position from start
            while (sibling.previousElementSibling) {
                sibling = sibling.previousElementSibling;
                if (sibling.nodeType === Node.ELEMENT_NODE && sibling.nodeName === element.nodeName) {
                    index++;
                }
            }
            // Now count forward from the first element
            sibling = element.parentNode ? element.parentNode.firstElementChild : null;
            let position = 0;
            while (sibling) {
                if (sibling.nodeName === element.nodeName) {
                    position++;
                    if (sibling === element) {
                        return position;
                    }
                }
                sibling = sibling.nextElementSibling;
            }
            return 1;
        };

        const parts = [];
        let current = el;
        while (current && current.nodeType === Node.ELEMENT_NODE) {
            const tagName = current.nodeName.toLowerCase();
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

    // --- 3. INITIALIZE UI (ONE TIME RENDER) ---
    if (!document.getElementById('pw-ui')) {
        const ui = document.createElement('div');
        ui.id = 'pw-ui';
        ui.innerHTML = `
            <div id="pw-ui-header" data-testid="pw-ui-header">
                <h2>Select Category: <span id="pw-category-name" data-testid="pw-category-name">...</span></h2>
                <div style="opacity:0.5">::</div>
            </div>
            <div id="pw-ui-body" data-testid="pw-ui-body">
                <div class="pw-stat-row">
                    <span id="pw-step-counter" data-testid="pw-step-counter">Step 1/1</span>
                    <span id="pw-count-badge" data-testid="pw-count-badge">0 selected</span>
                </div>
                <div class="pw-stat-row" id="pw-predicted-row" data-testid="pw-predicted-row">
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
                <div class="pw-hint" data-testid="pw-hint">Right-Click to interact (close modals)<br>Ctrl+Shift+Z to Redo</div>
            </div>
        `;
        document.body.appendChild(ui);

        document.getElementById('pw-btn-prev').onclick = () => window._action = 'prev';
        document.getElementById('pw-btn-next').onclick = () => window._action = 'next';
        document.getElementById('pw-btn-done').onclick = () => window._action = 'done';
        document.getElementById('pw-btn-select-predicted').onclick = () => window._action = 'select_predicted';
        
        // --- DRAGGABLE LOGIC ---
        const header = document.getElementById('pw-ui-header');
        let isDragging = false;
        let startX, startY, initialX = 0, initialY = 0;
        let xOffset = 0, yOffset = 0;

        header.onmousedown = (e) => {
            if(e.button !== 0) return;
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

    // --- 4. EVENT LISTENERS ---
    
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

    // --- LEFT CLICK (SELECT) ---
    window._clickedSelector = null;
    document.addEventListener('click', (e) => {
        if (e.target.closest('#pw-ui')) return;

        // Ignore simulated interaction clicks
        if (window._isSimulatingClick) return;

        // Standard Left Click -> SELECT
        if (e.button === 0) { 
            e.preventDefault();
            e.stopPropagation();
            window._clickedSelector = window._generateSelector(e.target);
        }
    }, true);

    // --- RIGHT CLICK (INTERACT / CLICK-OUT) ---
    document.addEventListener('contextmenu', (e) => {
        if (e.target.closest('#pw-ui')) return;

        e.preventDefault(); // Stop Context Menu
        
        // Simulate a "Real" click on the element to close modals/click buttons
        window._isSimulatingClick = true;
        try {
            // Dispatch a manual click to trigger element handlers.
            // The top-level navigation-prevention listener will still prevent URL changes for links.
            e.target.click(); 
        } catch(err) {
            console.log("Interaction failed", err);
        }
        // Reset flag
        window._isSimulatingClick = false;
        
    }, true);

    // --- KEYBOARD (UNDO/REDO) ---
    window._keyAction = null;
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            const k = e.key.toLowerCase();
            
            if (k === 'z') {
                e.preventDefault();
                // Check Shift explicitly for Redo
                window._keyAction = e.shiftKey ? 'redo' : 'undo';
            }
            // Fallback for browsers using Ctrl+Y for Redo
            else if (k === 'y') {
                e.preventDefault();
                window._keyAction = 'redo';
            }
        }
    });
})();