(function() {
    // --- State Variables ---
    let currentHighlight = null;
    let isSelectorActive = false;
    let currentCategory = null;
    let currentSelectedSelectors = new Set();
    let undoStack = [];
    let redoStack = []; 

    // --- Modal HTML and CSS ---
    const modalStyle = `
        #selector-modal {
            position: fixed;
            bottom: 20px;
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
            display: block;
            max-width: 90%;
            text-align: center;
            cursor: move;
            user-select: none;
        }
        #selector-modal.dragging {
            cursor: grabbing;
        }
        #selector-modal p {
            margin: 0 0 12px 0;
            font-size: 18px;
            font-weight: 600;
        }
        #selector-modal p strong {
            color: #007aff;
            font-size: 20px;
            text-transform: uppercase;
            text-shadow: 0 0 10px rgba(0, 122, 255, 0.3);
        }
        #selector-modal button {
            padding: 8px 14px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            margin: 0 5px;
            background-color: #34c759;
            color: white;
            transition: opacity 0.2s ease;
        }
        #selector-modal button:hover:not(:disabled) {
            opacity: 0.85;
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
            color: #666;
            cursor: not-allowed;
            opacity: 0.6;
        }
    `;

    const highlightStyle = `
        /* Single style for all selected items */
        [data-selector-highlight="selected"] {
            outline: 4px solid #34c759 !important; /* Green */
            box-shadow: 0 0 15px #34c759;
            background-color: rgba(52, 199, 89, 0.15);
        }

        [data-selector-highlight="predicted"] {
            outline: 3px dashed #007aff !important; /* Blue dashed */
            box-shadow: 0 0 10px #007aff;
            background-color: rgba(0, 122, 255, 0.1);
        }
        
        [data-selector-highlight="model-predicted"] {
            outline: 3px dashed #ff9500 !important; /* Orange dashed */
            box-shadow: 0 0 10px #ff9500;
            background-color: rgba(255, 149, 0, 0.1);
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
        
        // Make modal draggable
        makeDraggable(document.getElementById('selector-modal'));
    }
    
    /**
     * Makes an element draggable
     */
    function makeDraggable(element) {
        let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
        
        element.onmousedown = dragMouseDown;
        
        function dragMouseDown(e) {
            // Don't drag if clicking on buttons
            if (e.target.tagName === 'BUTTON') return;
            
            e.preventDefault();
            pos3 = e.clientX;
            pos4 = e.clientY;
            document.onmouseup = closeDragElement;
            document.onmousemove = elementDrag;
            element.classList.add('dragging');
        }
        
        function elementDrag(e) {
            e.preventDefault();
            pos1 = pos3 - e.clientX;
            pos2 = pos4 - e.clientY;
            pos3 = e.clientX;
            pos4 = e.clientY;
            
            // Update position
            const newTop = element.offsetTop - pos2;
            const newLeft = element.offsetLeft - pos1;
            
            element.style.top = newTop + "px";
            element.style.left = newLeft + "px";
            element.style.bottom = "auto";
            element.style.transform = "none";
        }
        
        function closeDragElement() {
            document.onmouseup = null;
            document.onmousemove = null;
            element.classList.remove('dragging');
        }
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
     * Removes all highlights from elements
     */
    function removeAllHighlights() {
        if (currentHighlight) {
            currentHighlight.style.outline = ''; 
            currentHighlight = null;
        }
        // Query for all highlighted elements and remove the attribute
        document.querySelectorAll('[data-selector-highlight]').forEach(el => {
            el.removeAttribute('data-selector-highlight');
            // Remove inline styles as well
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
        
        // Setup keyboard shortcuts for undo/redo
        document.addEventListener('keydown', handleKeyboardShortcuts);
    }
    
    /**
     * Handle keyboard shortcuts for undo/redo
     */
    function handleKeyboardShortcuts(e) {
        // Ctrl+Z for undo
        if (e.ctrlKey && !e.shiftKey && e.key === 'z') {
            e.preventDefault();
            performUndo();
        }
        // Ctrl+Shift+Z for redo
        else if (e.ctrlKey && e.shiftKey && e.key === 'Z') {
            e.preventDefault();
            performRedo();
        }
    }
    
    /**
     * Undo the last selection action
     */
    function performUndo() {
        if (undoStack.length === 0) return;
        
        const lastAction = undoStack.pop();
        redoStack.push(lastAction);
        
        if (lastAction.type === 'add') {
            // Undo an add by removing
            window.pywebview.api.unselect_selector(lastAction.category, lastAction.selector);
        } else if (lastAction.type === 'remove') {
            // Undo a remove by adding back
            window.pywebview.api.accept_selection(lastAction.category, lastAction.selector, false);
        }
    }
    
    /**
     * Redo the last undone action
     */
    function performRedo() {
        if (redoStack.length === 0) return;
        
        const action = redoStack.pop();
        undoStack.push(action);
        
        if (action.type === 'add') {
            // Redo an add
            window.pywebview.api.accept_selection(action.category, action.selector, false);
        } else if (action.type === 'remove') {
            // Redo a remove
            window.pywebview.api.unselect_selector(action.category, action.selector);
        }
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
                    }
                }
            }
        }
        // --- END OF FIX ---

        // Request model predictions
        let modelPredictBtn = '';
        window.pywebview.api.get_model_predictions(category).then(predictions => {
            if (predictions && predictions.length > 0) {
                // Count new predictions
                let newPredictions = predictions.filter(sel => !currentSelectedSelectors.has(sel)).length;
                if (newPredictions > 0) {
                    // Highlight model predictions
                    predictions.forEach(selector => {
                        try {
                            document.querySelectorAll(selector).forEach(el => {
                                if (el.getAttribute('data-selector-highlight') !== 'selected') {
                                    el.setAttribute('data-selector-highlight', 'model-predicted');
                                }
                            });
                        } catch (e) {
                            console.warn("Could not highlight model prediction:", selector, e);
                        }
                    });
                    
                    // Add button for accepting model predictions
                    const modelBtn = document.createElement('button');
                    modelBtn.id = 'model-predict-btn';
                    modelBtn.style.backgroundColor = '#ff9500';
                    modelBtn.textContent = `🤖 Add ${newPredictions} AI Predictions`;
                    modelBtn.setAttribute('data-predictions', JSON.stringify(predictions));
                    modelBtn.onclick = acceptModelPredictions;
                    
                    const buttonsDiv = document.getElementById('modal-buttons');
                    const nextBtn = document.getElementById('next-cat-btn');
                    buttonsDiv.insertBefore(modelBtn, nextBtn);
                }
            }
        }).catch(err => {
            console.warn("Could not get model predictions:", err);
        });
        
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

        // Prediction button handler
        const predictBtn = document.getElementById('predict-selectors-btn');
        if (predictBtn) {
            predictBtn.onclick = (e) => {
                // Disable button to prevent double-click
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
                    // No new selectors to add, refresh the prompt
                    window.pywebview.api.refresh_prompt();
                } else {
                    // Send all selectors to Python
                    // Pass false to prevent re-running prediction on these
                    const selectorsArray = Array.from(selectorsToAdd);
                    selectorsArray.forEach(selector => {
                        window.pywebview.api.accept_selection(currentCategory, selector, false);
                    });
                }
            };
        }
    }
    
    /**
     * Accept all model predictions
     */
    function acceptModelPredictions(e) {
        e.target.disabled = true;
        e.target.innerText = 'Adding AI predictions...';
        
        const predictions = JSON.parse(e.target.getAttribute('data-predictions'));
        const selectorsToAdd = predictions.filter(sel => !currentSelectedSelectors.has(sel));
        
        if (selectorsToAdd.length === 0) {
            window.pywebview.api.refresh_prompt();
            return;
        }
        
        // Add all predictions without triggering further predictions
        selectorsToAdd.forEach(selector => {
            window.pywebview.api.accept_selection(currentCategory, selector, false);
        });
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

        // Add new highlight only if element is not already selected or predicted
        if (target.getAttribute('data-selector-highlight')) {
            currentHighlight = null;
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

        // Prevent default link behavior and event bubbling
        e.preventDefault();
        e.stopPropagation();
        
        if (!isSelectorActive) {
            return;
        }
            
        const selector = getCssSelector(e.target);
        if (!selector) {
            return; // Could not generate a valid selector for this element
        }
        
        isSelectorActive = false; 
        
        if (currentHighlight) {
            currentHighlight.style.outline = '';
        }

        updateModal(`Working on <strong>${currentCategory}</strong>...`, '');

        if (currentSelectedSelectors.has(selector)) {
            // UNSELECT - add to undo stack
            undoStack.push({type: 'remove', category: currentCategory, selector: selector});
            redoStack = []; // Clear redo stack on new action
            window.pywebview.api.unselect_selector(currentCategory, selector);
        } else {
            // NEW SELECT - add to undo stack, pass `true` to run prediction
            undoStack.push({type: 'add', category: currentCategory, selector: selector});
            redoStack = []; // Clear redo stack on new action
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
                // Use attribute selector for IDs to handle special characters
                selector = `[id="${el.id.trim()}"]`;
                path.unshift(selector);
                break; // ID is unique, stop here
            } else {
                if (el.className && typeof el.className === 'string') {
                    const classes = el.className.trim().split(/\s+/).filter(Boolean).join('.');
                    if (classes) {
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
                    // Check if there are siblings with the same tag name after this element
                    let needsNth1 = false;
                    sib = el;
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
        return path.join(' > ').replace(/\s+/g, ' ');
    }

    /**
     * Generates an XPath selector for an element (not currently used)
     * TODO: Implement proper XPath generation
     */
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