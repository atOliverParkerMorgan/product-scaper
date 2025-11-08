(function() {
    // --- State Variables ---
    let currentHighlight = null;
    let isSelectorActive = false;
    let currentCategory = null;

    // --- Modal HTML and CSS ---
    // We inject this into the page to show non-blocking messages.
    const modalStyle = `
        #selector-modal {
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: white;
            border: 1px solid #ccc;
            border-radius: 8px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
            padding: 24px;
            z-index: 999999;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 16px;
            line-height: 1.5;
            color: #333;
            display: none; /* Hidden by default */
            max-width: 90%;
            text-align: center;
        }
        #selector-modal p {
            margin: 0 0 16px 0;
        }
        #selector-modal button {
            padding: 10px 16px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            margin: 0 8px;
        }
        #modal-ok-btn {
            background-color: #007aff;
            color: white;
        }
        #modal-cancel-btn {
            background-color: #f0f0f0;
            color: #333;
        }
    `;

    const modalHTML = `
        <div id="selector-modal">
            <p id="modal-message"></p>
            <div id="modal-buttons"></div>
        </div>
    `;

    /**
     * Injects the modal and its styles into the page.
     */
    function injectModal() {
        if (document.getElementById('selector-modal')) return; // Already injected

        const styleSheet = document.createElement("style");
        styleSheet.type = "text/css";
        styleSheet.innerText = modalStyle;
        document.head.appendChild(styleSheet);

        document.body.insertAdjacentHTML('beforeend', modalHTML);
    }

    /**
     * Shows the modal with a message and optional buttons.
     * @param {string} message - The HTML content for the modal message.
     * @param {boolean} showConfirmButtons - Show OK/Redo buttons.
     */
    function showModal(message, showConfirmButtons = false) {
        const modal = document.getElementById('selector-modal');
        document.getElementById('modal-message').innerHTML = message;
        const buttonContainer = document.getElementById('modal-buttons');

        if (showConfirmButtons) {
            buttonContainer.innerHTML = `
                <button id="modal-ok-btn">Looks Good (OK)</button>
                <button id="modal-cancel-btn">Redo</button>
            `;
        } else {
            buttonContainer.innerHTML = '';
        }
        modal.style.display = 'block';
        return modal;
    }

    function hideModal() {
        const modal = document.getElementById('selector-modal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    /**
     * Removes all highlights (both red and blue) from the page.
     */
    function removeAllHighlights() {
        // Remove red hover highlight
        if (currentHighlight) {
            currentHighlight.style.outline = '';
            currentHighlight = null;
        }
        // Remove blue prediction highlights
        document.querySelectorAll('[data-selector-highlight="true"]').forEach(el => {
            el.style.outline = '';
            el.removeAttribute('data-selector-highlight');
        });
    }

    // --- Communication with Python ---

    /**
     * Called by Python to initialize the app.
     */
    window.initApp = function() {
        console.log("Selector.js initialized.");
        injectModal();
        // Tell Python we are ready to start the workflow
        window.pywebview.api.start_workflow();
    }

    /**
     * Called by Python to ask the user to select an element for a category.
     * @param {string} category
     */
    window.promptForSelection = function(category) {
        console.log(`Prompting for ${category}...`);
        removeAllHighlights();
        currentCategory = category;
        isSelectorActive = true;
        showModal(`Please select the <strong>${category}</strong>.<br><small>(Mouse over to highlight, click to select)</small>`);
    }

    /**
     * Called by Python to highlight "predicted" elements and ask for confirmation.
     * @param {string} selector - The simplified selector to highlight.
     * @param {string} category - The category being confirmed.
     */
    window.highlightAndConfirm = function(selector, category) {
        const elements = document.querySelectorAll(selector);
        console.log(`Found ${elements.length} similar elements for ${category}.`);

        if (elements.length === 0) {
            // No matches, something is wrong with the selector.
            window.pywebview.api.prediction_confirmed(category, false); // Auto-fail
            return;
        }

        elements.forEach(el => {
            el.style.outline = '3px dashed blue';
            el.setAttribute('data-selector-highlight', 'true');
        });

        const modal = showModal(
            `Found <strong>${elements.length}</strong> similar elements for <strong>${category}</strong>.
             <br><small>Do they look correct?</small>`,
            true
        );

        document.getElementById('modal-ok-btn').onclick = () => {
            window.pywebview.api.prediction_confirmed(category, true);
            removeAllHighlights();
            hideModal();
        };

        document.getElementById('modal-cancel-btn').onclick = () => {
            window.pywebview.api.prediction_confirmed(category, false);
            removeAllHighlights();
            hideModal();
        };
    }

    // --- Event Listeners ---

    document.addEventListener('mouseover', function(e) {
        if (!isSelectorActive) return;

        // Remove old highlight
        if (currentHighlight) {
            currentHighlight.style.outline = '';
        }
        
        // Add new highlight
        currentHighlight = e.target;
        // Don't highlight our own modal
        if (currentHighlight && currentHighlight.closest && currentHighlight.closest('#selector-modal')) {
            currentHighlight = null;
            return;
        }
        currentHighlight.style.outline = '2px dashed red';
    });

    document.addEventListener('click', function(e) {
        // ALWAYS stop the click from doing anything (like following a link)
        // This disables all redirects.
        e.preventDefault();
        e.stopPropagation();

        // Only run selector logic if we are in selection mode
        if (!isSelectorActive) return;
        
        const selector = getCssSelector(e.target);
        
        // Deactivate selector
        isSelectorActive = false;
        hideModal();
        if (currentHighlight) {
            currentHighlight.style.outline = '';
        }

        // Send the selector to Python
        window.pywebview.api.save_selector(currentCategory, selector);

    }, true); // Use 'true' to capture the event before it bubbles up


    /**
     * Helper function to calculate a unique CSS selector for an element.
     */
    function getCssSelector(el) {
        if (!(el instanceof Element)) return;
        const path = [];
        while (el.nodeType === Node.ELEMENT_NODE) {
            let selector = el.nodeName.toLowerCase();
            if (el.id) {
                // Use attribute selector [id="..."]
                // This is the most robust way to handle IDs that
                // start with numbers or contain special characters.
                selector = `[id="${el.id.trim()}"]`;
                path.unshift(selector);
                break; // ID is unique, no need to go further
            } else {
                // Add classes
                if (el.className) {
                    const classes = el.className.trim().split(/\s+/).join('.');
                    if(classes) {
                        selector += '.' + classes;
                    }
                }

                let sib = el, nth = 1;
                while (sib = sib.previousElementSibling) {
                    if (sib.nodeName.toLowerCase() == selector)
                        nth++;
                }
                if (nth != 1)
                    selector += ":nth-of-type(" + nth + ")";
            }
            path.unshift(selector);
            el = el.parentNode;
        }
        // Join and clean up extra spaces
        return path.join(" > ").replace(/\s+/g, ' ');
    }
})();