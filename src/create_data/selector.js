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
            margin: 0 5px; /* Added spacing */
            background-color: #34c759;
            color: white;
        }
        #selector-modal button#prev-cat-btn {
             background-color: #f0f0f0;
             color: #333;
        }
        /* Style for the prediction button */
        #selector-modal button#predict-selectors-btn {
            background-color: #007aff; /* Blue */
            color: white;
        }
        #selector-modal button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
    `;

    const highlightStyle = `
        /* Single style for all selected items */
        [data-selector-highlight="selected"] {
            outline: 3px solid #34c759 !important; /* Green */
            box-shadow: 0 0 10px #34c759;
            background-color: rgba(52, 199, 89, 0.1);
        }

        [data-selector-highlight="predicted"] {
            outline: 3px solid #007aff !important; /* Blue */
            box-shadow: 0 0 10px #007aff;
            background-color: rgba(0, 122, 255, 0.1);
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
        // --- FIX ---
        // Query for all highlighted elements and remove the attribute
        document.querySelectorAll('[data-selector-highlight]').forEach(el => {
            el.removeAttribute('data-selector-highlight');
            // Remove inline styles too, just in case
            el.style.outline = ''; 
            el.style.boxShadow = '';
            el.style.backgroundColor = '';
        });
    }

    // --- Communication with Python ---

    window.initApp = function() {
        injectModal();
        // Start the workflow
        window.pywebview.api.start_workflow();
    }

    /**
     * Called by Python to ask the user to select an element for a category.
     */
    window.promptForSelection = function(category, existingSelectorsArray, selectorType) {
        
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


        let predictSelectorsBtn = '';
        // Check if there are any existing selectors to base a prediction on
        if (existingSelectorsArray.length > 0) {
            // Get the *last* selector added as the basis for prediction
            const lastSpecificSelector = existingSelectorsArray[existingSelectorsArray.length - 1];
            let lastElement = null;

            try {
                lastElement = document.querySelector(lastSpecificSelector);
            } catch (e) {
                console.warn("Last selector was invalid, cannot predict:", lastSpecificSelector, e);
            }

            // Only proceed if we found the last element
            if (lastElement) {
                // Generate a *generalized* selector based on tag + class (per user request)
                // This is better for prediction than a unique ID or a hyper-specific path
                let generalizedSelector = lastElement.tagName.toLowerCase();
                if (lastElement.className && typeof lastElement.className === 'string') {
                    const classes = lastElement.className.trim().split(/\s+/).filter(Boolean).join('.');
                    if (classes) {
                        generalizedSelector += '.' + classes;
                    }
                }
                
                let predictedElements = [];
                let numPredicted = 0;

                try {
                    // We check for the tag + class to find similar elements
                    predictedElements = Array.from(document.querySelectorAll(generalizedSelector));
                    numPredicted = predictedElements.length;
                } catch (e) {
                    console.warn("Invalid prediction selector:", generalizedSelector, e);
                }

                if (numPredicted > 0) {
                    let newElementsFound = 0;
                    predictedElements.forEach(el => {
                        // Highlight predicted elements, but only if not already selected
                        if (el.getAttribute('data-selector-highlight') !== 'selected') {
                            el.setAttribute('data-selector-highlight', 'predicted');
                            newElementsFound++;
                        }
                    });

                    // Only show the button if it adds new elements
                    if (newElementsFound > 0) {
                         predictSelectorsBtn = `<button id="predict-selectors-btn" data-selector='${generalizedSelector}'>Add ${newElementsFound} Similar</button>`;
                    } else {
                    }
                }
            }
        }
        // --- END OF FIX ---

        let message = `Selecting: <strong>${category}</strong> <small style="color:#34c759">(${existingSelectorsArray.length} selected)</small>`;
        let buttonsHtml = `
            <button id="prev-cat-btn">&larr; Previous</button>
            ${predictSelectorsBtn}
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

        // --- PREDICTION BUTTON HANDLER ---
        const predictBtn = document.getElementById('predict-selectors-btn');
        if (predictBtn) {
            predictBtn.onclick = (e) => {
                // Disable button to prevent double click
                e.target.disabled = true; 
                e.target.innerText = 'Adding...';

                const generalizedSelector = predictBtn.getAttribute('data-selector');
                isSelectorActive = false;

                const selectorsToAdd = new Set();
                try {
                    const elements = document.querySelectorAll(generalizedSelector);
                    elements.forEach(el => {
                        const specificSelector = getCssSelector(el);
                        // Add if it's valid and NOT already selected
                        if (specificSelector && !currentSelectedSelectors.has(specificSelector)) {
                            selectorsToAdd.add(specificSelector);
                        }
                    });
                } catch (e) {
                    console.warn("Prediction selector was invalid:", generalizedSelector, e);
                }

                if (selectorsToAdd.size === 0) {
                    window.pywebview.api.refresh_prompt();
                } else {
                    // Send all selectors to Python.
                    // Pass `false` to prevent re-running prediction on these.
                    const selectorsArray = Array.from(selectorsToAdd);
                    selectorsArray.forEach(selector => {
                        window.pywebview.api.accept_selection(currentCategory, selector, false);
                    });
                }
            };
        }
    }

    // --- Event Listeners ---

    document.addEventListener('mouseover', function(e) {
        if (!isSelectorActive) return;
        
        const target = e.target;
        // Don't highlight the modal itself
        if (target.closest && target.closest('#selector-modal')) {
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

        // Add new highlight (if not already selected or predicted)
        if (target.getAttribute('data-selector-highlight')) {
             currentHighlight = null; // Don't highlight already-selected/predicted items
        } else {
            currentHighlight = target;
            currentHighlight.style.outline = '2px dashed red';
        }
    });

    document.addEventListener('click', function(e) {
        // Ignore clicks on the modal
        if (e.target.closest && e.target.closest('#selector-modal')) {
            return;
        }

        // disable links
        e.preventDefault();
        e.stopPropagation();
        
        if (!isSelectorActive) {
            return;
        }
            
        const selector = getCssSelector(e.target);
        if (!selector) return; // Not a valid element
        
        
        isSelectorActive = false; 
        
        if (currentHighlight) {
            currentHighlight.style.outline = '';
        }

        updateModal(`Working on <strong>${currentCategory}</strong>...`, '');

        if (currentSelectedSelectors.has(selector)) {
            // UNSELECT
            window.pywebview.api.unselect_selector(currentCategory, selector);
        } else {
            // NEW SELECT - pass `true` to run prediction
            window.pywebview.api.accept_selection(currentCategory, selector, true);
        }

    }, true); // Use capture phase to catch clicks first


    /**
     * Helper function to calculate a unique CSS selector for an element.
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
                    let needsNth1 = false;
                    while (sib = sib.nextElementSibling) {
                         if (sib.nodeName.toLowerCase() === el.nodeName.toLowerCase()) {
                            needsNth1 = true;
                            break;
                         }
                    }
                    if (needsNth1) {
                         selector += `:nth-of-type(1)`;
                    }
                }
            }
            path.unshift(selector);
            el = el.parentNode;
        }
        return path.join(" > ").replace(/\s+/g, ' ');
    }
    // TODO: Implement this
    function getXpathSelector(el) {
        if (el.id) {
            return `//*[@id="${el.id.trim()}"]`;
        }
        const parts = [];
        while (el && el.nodeType === Node.ELEMENT_NODE) {
            let nbOfPreviousSiblings = 0;
            let hasNextSiblings = false;
            let sibling = el.previousSibling;
            while (sibling) {
                if (sibling.nodeType !== Node.DOCUMENT_TYPE_NODE && sibling.nodeName.toLowerCase() === el.nodeName.toLowerCase()) {
                    nbOfPreviousSiblings++;
                }
                sibling = sibling.previousSibling;
            }
            sibling = el.nextSibling;
            while (sibling) {
                if (sibling.nodeType !== Node.DOCUMENT_TYPE_NODE && sibling.nodeName.toLowerCase() === el.nodeName.toLowerCase()) {
                    hasNextSiblings = true;
                    break;
                }
                sibling = sibling.nextSibling;
            }
            const prefix = el.prefix ? el.prefix + ":" : "";
            const nth = nbOfPreviousSiblings || hasNextSiblings ? `[${nbOfPreviousSiblings + 1}]` : "";
            parts.push(prefix + el.localName + nth);
            el = el.parentNode;
        }
        return "/" + parts.reverse().join("/");
    }
})();