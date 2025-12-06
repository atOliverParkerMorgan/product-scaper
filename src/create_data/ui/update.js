(data) => {
    const el = document.getElementById('pw-ui');
    if(!el) return;

    document.getElementById('pw-category-name').innerText = data.category;
    document.getElementById('pw-step-counter').innerText = `Step ${data.idx + 1}/${data.total}`;
    document.getElementById('pw-count-badge').innerText = `${data.count} selected`;
    
    const btnPrev = document.getElementById('pw-btn-prev');
    const btnNext = document.getElementById('pw-btn-next');
    const btnDone = document.getElementById('pw-btn-done');
    
    if (data.idx === 0) btnPrev.classList.add('pw-hidden');
    else btnPrev.classList.remove('pw-hidden');
    
    if (data.idx === data.total - 1) {
        btnNext.classList.add('pw-hidden');
        btnDone.classList.remove('pw-hidden');
    } else {
        btnNext.classList.remove('pw-hidden');
        btnDone.classList.add('pw-hidden');
    }
}