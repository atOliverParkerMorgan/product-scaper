/* ui/update.js */
(data) => {
    const el = document.getElementById('pw-ui');
    if (!el) return;

    // Text Updates
    document.getElementById('pw-category-name').innerText = data.category;
    document.getElementById('pw-step-counter').innerText = `Step ${data.idx + 1}/${data.total}`;
    document.getElementById('pw-count-badge').innerText = `${data.count} selected`;

    // Buttons
    const btnPrev = document.getElementById('pw-btn-prev');
    const btnNext = document.getElementById('pw-btn-next');
    const btnDone = document.getElementById('pw-btn-done');

    // Toggle Buttons
    const btnAi = document.getElementById('pw-btn-toggle-ai');
    const btnClass = document.getElementById('pw-btn-toggle-class');
    const predictedBadge = document.getElementById('pw-predicted-badge');

    // Nav Visibility
    btnPrev.classList.toggle('pw-hidden', data.idx === 0);
    if (data.idx === data.total - 1) {
        btnNext.classList.add('pw-hidden');
        btnDone.classList.remove('pw-hidden');
    } else {
        btnNext.classList.remove('pw-hidden');
        btnDone.classList.add('pw-hidden');
    }

    // AI Button State & Tooltip
    predictedBadge.innerText = `${data.totalPredictions} found`;
    if (data.totalPredictions === 0) {
        btnAi.innerText = "No AI Found";
        btnAi.disabled = true;
        btnAi.title = "No AI predictions available for this category";
        btnAi.classList.remove('is-unselect');
    } else if (data.allAiSelected) {
        btnAi.innerText = "Unselect AI";
        btnAi.disabled = false;
        btnAi.title = "Deselect all AI predicted elements";
        btnAi.classList.add('is-unselect');
    } else {
        btnAi.innerText = "Select AI";
        btnAi.disabled = false;
        btnAi.title = "Select all AI predicted elements";
        btnAi.classList.remove('is-unselect');
    }

    // Class Button State & Tooltip
    if (data.count === 0) {
        btnClass.innerText = "Class Select";
        btnClass.disabled = true;
        btnClass.title = "Select an item manually first to use Class Select";
        btnClass.classList.remove('is-unselect');
    } else if (data.allClassSelected) {
        btnClass.innerText = "Class Unselect";
        btnClass.disabled = false;
        btnClass.title = "Deselect all elements with the same class as the last selected item";
        btnClass.classList.add('is-unselect');
    } else {
        btnClass.innerText = "Class Select";
        btnClass.disabled = false;
        btnClass.title = "Select all elements with the same class as the last selected item";
        btnClass.classList.remove('is-unselect');
    }
}