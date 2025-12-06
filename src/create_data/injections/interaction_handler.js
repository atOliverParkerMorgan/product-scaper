// Store state to allow multiple selections
window._selectionState = window._selectionState || {
    waitingForAction: false,
    resolveAction: null,
    clickHandler: null
};

// Use optimal-select library for better CSS selector generation
window._generateSelector = window._generateSelector || ((elementRef) => {
    const el = elementRef;
    
    // Validate element
    if (!el || !el.tagName || el.nodeType !== Node.ELEMENT_NODE) {
        console.error('Invalid element provided to selector generator');
        return null;
    }
    
    try {
        // Use optimal-select library if available
        if (typeof OptimalSelect !== 'undefined' && OptimalSelect.select) {
            return OptimalSelect.select(el);
        } else if (typeof window.onselect !== 'undefined') {
            return window.onselect(el);
        }
        
        // Fallback: Try ID first
        if (el.id && !/\s/.test(el.id)) {
            const idSel = '#' + CSS.escape(el.id);
            if (document.querySelectorAll(idSel).length === 1) {
                return idSel;
            }
        }
        
        // Simple fallback selector
        let selector = el.tagName.toLowerCase();
        if (el.className && typeof el.className === 'string') {
            const classes = el.className.trim().split(/\s+/)
                .filter(c => c && /^[a-zA-Z_-]/.test(c) && !c.startsWith('pw-'))
                .slice(0, 2);
            if (classes.length > 0) {
                selector += '.' + classes.map(c => CSS.escape(c)).join('.');
            }
        }
        return selector;
    } catch (e) {
        console.error('Error generating selector:', e);
        return null;
    }
});

() => {
    return new Promise((resolve) => {
        window._selectionState.waitingForAction = true;
        window._selectionState.resolveAction = resolve;
        
        // Click handler for elements - DON'T remove it after each click
        if (!window._selectionState.clickHandler) {
            window._selectionState.clickHandler = (e) => {
                // Only intercept left clicks (button 0), allow right-clicks through for closing modals
                if (e.button !== 0) return;
                
                // Ignore clicks on UI
                if (e.target.closest('#pw-ui')) return;
                
                // Only process if we're waiting for an action
                if (!window._selectionState.waitingForAction) return;
                
                e.preventDefault();
                e.stopPropagation();
                
                // Generate selector immediately and return it (not the element)
                if (window._selectionState.resolveAction) {
                    window._selectionState.waitingForAction = false;
                    const resolver = window._selectionState.resolveAction;
                    window._selectionState.resolveAction = null;
                    
                    const selector = window._generateSelector(e.target);
                    resolver({ type: 'element', selector: selector });
                }
            };
            
            document.addEventListener('click', window._selectionState.clickHandler, true);
        }
        
        // Button handlers
        const prev = document.getElementById('pw-prev');
        const next = document.getElementById('pw-next');
        const done = document.getElementById('pw-done');
        
        if (prev) prev.onclick = () => {
            if (!window._selectionState.waitingForAction) return;
            window._selectionState.waitingForAction = false;
            const resolver = window._selectionState.resolveAction;
            window._selectionState.resolveAction = null;
            resolver({ type: 'prev' });
        };
        if (next) next.onclick = () => {
            if (!window._selectionState.waitingForAction) return;
            window._selectionState.waitingForAction = false;
            const resolver = window._selectionState.resolveAction;
            window._selectionState.resolveAction = null;
            resolver({ type: 'next' });
        };
        if (done) done.onclick = () => {
            if (!window._selectionState.waitingForAction) return;
            window._selectionState.waitingForAction = false;
            const resolver = window._selectionState.resolveAction;
            window._selectionState.resolveAction = null;
            // Clean up the click handler when done
            if (window._selectionState.clickHandler) {
                document.removeEventListener('click', window._selectionState.clickHandler, true);
                window._selectionState.clickHandler = null;
            }
            resolver({ type: 'done' });
        };
    });
}
