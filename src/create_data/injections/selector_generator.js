(elementRef) => {
    const el = elementRef;
    
    // Validate element
    if (!el || !el.tagName || el.nodeType !== Node.ELEMENT_NODE) {
        console.error('Invalid element provided to selector generator');
        return null;
    }
    
    // Try ID first
    if (el.id && !/\s/.test(el.id)) {
        const idSel = '#' + CSS.escape(el.id);
        if (document.querySelectorAll(idSel).length === 1) {
            return idSel;
        }
    }
    
    // Build path-based selector
    function getPath(element) {
        if (!element || !element.tagName || element.nodeType !== Node.ELEMENT_NODE) return '';
        if (element.tagName === 'HTML') return '';
        if (element.tagName === 'BODY') return 'body';
        
        let path = element.tagName.toLowerCase();
        
        // Add classes (excluding our own)
        if (element.className && typeof element.className === 'string') {
            const classes = element.className.trim().split(/\s+/)
                .filter(c => c && /^[a-zA-Z_-]/.test(c) && !c.startsWith('pw-'))
                .map(c => CSS.escape(c));
            if (classes.length > 0 && classes.length <= 3) {
                path += '.' + classes.slice(0, 2).join('.');
            }
        }
        
        // Add nth-of-type for specificity
        const parent = element.parentElement;
        if (parent) {
            const siblings = Array.from(parent.children)
                .filter(child => child.tagName === element.tagName);
            if (siblings.length > 1) {
                const index = siblings.indexOf(element) + 1;
                path += `:nth-of-type(${index})`;
            }
        }
        
        return path;
    }
    
    // Build selector from element to root
    const parts = [];
    let current = el;
    let depth = 0;
    
    while (current && current.nodeType === Node.ELEMENT_NODE && current.tagName !== 'HTML' && depth < 6) {
        const part = getPath(current);
        if (part) parts.unshift(part);
        
        // Test if current path is unique
        const testSel = parts.join(' > ');
        try {
            if (document.querySelectorAll(testSel).length === 1) {
                return testSel;
            }
        } catch(e) {}
        
        current = current.parentElement;
        depth++;
    }
    
    return parts.join(' > ') || null;
}
