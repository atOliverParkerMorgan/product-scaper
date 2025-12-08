(data) => {
    const el = document.getElementById('pw-ui');
    if(!el) return;

    document.getElementById('pw-category-name').innerText = data.category;
    document.getElementById('pw-step-counter').innerText = `Step ${data.idx + 1}/${data.total}`;
    document.getElementById('pw-count-badge').innerText = `${data.count} selected`;
    
    const btnPrev = document.getElementById('pw-btn-prev');
    const btnNext = document.getElementById('pw-btn-next');
    const btnDone = document.getElementById('pw-btn-done');
    const btnSelectPredicted = document.getElementById('pw-btn-select-predicted');
    const predictedRow = document.getElementById('pw-predicted-row');
    const predictedBadge = document.getElementById('pw-predicted-badge');
    
    if (data.idx === 0) btnPrev.classList.add('pw-hidden');
    else btnPrev.classList.remove('pw-hidden');
    
    if (data.idx === data.total - 1) {
        btnNext.classList.add('pw-hidden');
        btnDone.classList.remove('pw-hidden');
    } else {
        btnNext.classList.remove('pw-hidden');
        btnDone.classList.add('pw-hidden');
    }
    
    // Show/hide select predicted button and count based on whether there are predicted elements
    const predictedCount = document.querySelectorAll('.pw-predicted').length;
    if (predictedCount > 0) {
        btnSelectPredicted.classList.remove('pw-hidden');
        predictedRow.style.display = 'flex';
        predictedBadge.innerText = `${predictedCount} found`;
    } else {
        btnSelectPredicted.classList.add('pw-hidden');
        predictedRow.style.display = 'none';
    }
}