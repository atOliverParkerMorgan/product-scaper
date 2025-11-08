(function() {
    // --- State Variables ---
    let currentHighlight = null;
    let isSelectorActive = false;
    let currentCategory = null;
    let currentSelectedSelectors = new Set(); 

    // --- Modal HTML and CSS ---
    const modalStyle = `
        #selector-modal {
            position: fixed;
            bottom: 20px; /* Changed from top */
            left: 50%;
            transform: translateX(-50%);
            background: white;
            border: 1px solid #ccc;
            border-radius: 8px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
            padding: 16px 24px;
            z-index: 999999;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 16px;
            line-height: 1.5;
            color: #333;
            display: block; /* Always visible */
            max-width: 90%;
            text-align: center;
        }
        #selector-modal p {
            margin: 0 0 12px 0;
        }
        #selector-modal button {
            padding: 8px 14px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            margin: 0 8px;
            background-color: #007aff;
            color: white;
        }
        #selector-modal button#prev-cat-btn {
             background-color: #f0f0f0;
             color: #333;
        }
    `;

    const highlightStyle = `
        /* Single style for all selected items */
        [data-selector-highlight="selected"] {
            outline: 3px solid #34c759 !important; /* Green */
            box-shadow: 0 0 10px #34c759;
            background-color: rgba(52, 199, 89, 0.1);
        }
    `;

    const modalHTML = `
        <div id="selector-modal">
            <p id="modal-message">Loading...</p>
            <div id="modal-buttons"></div>
        </div>
    `;

    /**
     * Injects the modal and its styles into the page.
     */
    function injectModal() {
        if (document.getElementById('selector-modal')) return;

        const styleSheet = document.createElement("style");
        styleSheet.type = "text/css";
        styleSheet.innerText = modalStyle + '\n' + highlightStyle;
        document.head.appendChild(styleSheet);

        document.body.insertAdjacentHTML('beforeend', modalHTML);
    }

    /**
     * Updates the (now-persistent) modal with a message and buttons.
     */
    function updateModal(message, buttonsHtml) {
        const modal = document.getElementById('selector-modal');
        if (!modal) {
            console.error("Selector modal not found in DOM!");
            return;
        }
        document.getElementById('modal-message').innerHTML = message;
        document.getElementById('modal-buttons').innerHTML = buttonsHtml;
    }

    /**
     * Removes all highlights
     */
    function removeAllHighlights() {
        if (currentHighlight) {
            currentHighlight.style.outline = ''; 
            currentHighlight = null;
        }
        document.querySelectorAll('[data-selector-highlight="selected"]').forEach(el => {
            el.removeAttribute('data-selector-highlight');
        });
    }

    // --- Communication with Python ---

    window.initApp = function() {
        console.log("Selector.js initialized.");
        injectModal();
        // Start the workflow
        window.pywebview.api.start_workflow();
    }

    /**
     * Called by Python to ask the user to select an element for a category.
     */
    window.promptForSelection = function(category, existingSelectorsArray) {
        console.log(`Prompting for ${category}. Found ${existingSelectorsArray.length} existing.`);
        
        currentCategory = category;
        currentSelectedSelectors = new Set(existingSelectorsArray);
        isSelectorActive = true;
        
        removeAllHighlights();
        // Highlight all selectors provided by Python
        existingSelectorsArray.forEach(selector => {
            try {
                document.querySelectorAll(selector).forEach(el => {
                    el.setAttribute('data-selector-highlight', 'selected');
                });
            } catch (e) {
                console.warn("Could not highlight invalid selector:", selector, e);
            }
        });

        // Update the persistent modal
        let predictSelectors = '<button >'
        let message = `Selecting: <strong>${category}</strong> <small>(${existingSelectorsArray.length} selected)</small>`;
        let buttonsHtml = `
            <button id="prev-cat-btn">&larr; Previous</button>
            <button id="next-cat-btn">Next &rarr;</button>
        `;
        updateModal(message, buttonsHtml);

        // Add event listeners for new buttons
        document.getElementById('prev-cat-btn').onclick = () => {
            isSelectorActive = false; // Disable clicks during transition
            updateModal('Loading...', '');
            window.pywebview.api.user_clicked_previous_category();
        };
        
        document.getElementById('next-cat-btn').onclick = () => {
            isSelectorActive = false; // Disable clicks during transition
            updateModal('Loading...', '');
            window.pywebview.api.user_clicked_next_category();
        };
    }

    // --- Event Listeners ---

    document.addEventListener('mouseover', function(e) {
        if (!isSelectorActive) return;
        // Don't highlight the modal itself
        if (e.target.closest && e.target.closest('#selector-modal')) {
            if (currentHighlight) {
                currentHighlight.style.outline = '';
                currentHighlight = null;
            }
            return;
        }
        // Remove old highlight
        if (currentHighlight) {
            currentHighlight.style.outline = '';
        }
        // Add new highlight
        currentHighlight = e.target;
        currentHighlight.style.outline = '2px dashed red';
    });

    document.addEventListener('click', function(e) {
        // Ignore clicks on the modal
        if (e.target.closest && e.target.closest('#selector-modal')) {
            console.log("Click on modal, ignoring.");
            return;
        }

        // disable links
        e.preventDefault();
        e.stopPropagation();
        
        // Only run if selection is active
        if (!isSelectorActive) {
            return;
        }
            
        
        const selector = getCssSelector(e.target);
        if (!selector) return; // Not a valid element
        
        console.log("Element selected:", e.target, "Selector:", selector);
        
        // Disable clicking until Python calls promptForSelection again
        isSelectorActive = false; 
        
        if (currentHighlight) {
            currentHighlight.style.outline = '';
        }

        // Show loading state
        updateModal(`Working on <strong>${currentCategory}</strong>...`, '');

        // Check if we are selecting or unselecting
        if (currentSelectedSelectors.has(selector)) {
            // UNSELECT
            console.log("-> Unselecting element.");
            window.pywebview.api.unselect_selector(currentCategory, selector);
        } else {
            // NEW SELECT
            console.log("-> Selecting new element.");
            window.pywebview.api.accept_selection(currentCategory, selector);
        }

    }, true); // Use capture phase to catch clicks first


    /**
     * Helper function to calculate a unique CSS selector for an element.
     * (Unchanged from your original)
     */
    function getCssSelector(el) {
        if (!(el instanceof Element)) return;
        const path = [];
        while (el.nodeType === Node.ELEMENT_NODE) {
            if (el.nodeName.toLowerCase() === 'body') {
                path.unshift('body');
                break;
            }

            let selector = el.nodeName.toLowerCase();
            if (el.id) {
                // Use a more robust ID selector
                selector = `[id="${el.id.trim()}"]`;
                path.unshift(selector);
                break; // ID is unique, stop here
            } else {
                if (el.className && typeof el.className === 'string') {
                    const classes = el.className.trim().split(/\s+/).filter(Boolean).join('.');
                    if(classes) {
                        selector += '.' + classes;
                    }
                }

                let tagNth = 1;
                let sib = el;
                while (sib = sib.previousElementSibling) {
                    if (sib.nodeName.toLowerCase() === el.nodeName.toLowerCase()) {
                        tagNth++;
                    }
                }
                
                if (tagNth > 1) {
                    selector += `:nth-of-type(${tagNth})`;
                } else {
                    // Check if it's the *only* one. If not, add :nth-of-type(1)
                    sib = el;
                    while (sib = sib.nextElementSibling) {
                         if (sib.nodeName.toLowerCase() === el.nodeName.toLowerCase()) {
                            selector += `:nth-of-type(1)`;
                            break;
                         }
                    }
                }
            }
            path.unshift(selector);
            el = el.parentNode;
        }
        return path.join(" > ").replace(/\s+/g, ' ');
    }
})();