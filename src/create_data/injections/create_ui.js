(params) => {
    let ui = document.getElementById('pw-ui');
    if (!ui) {
        ui = document.createElement('div');
        ui.id = 'pw-ui';
        document.body.appendChild(ui);
    }
    
    ui.innerHTML = `
        <h2>Select: ${params.category}</h2>
        <div class="info">
            Category <strong>${params.categoryIdx + 1}/${params.total}</strong><br>
            Selected: <strong>${params.selectedCount}</strong> elements
        </div>
        <div class="instructions">
            💡 Hover over elements and click to select/deselect them for this category
        </div>
        <div class="buttons">
            <button class="btn-prev" id="pw-prev">← Previous</button>
            <button class="btn-next" id="pw-next">Next →</button>
            <button class="btn-done" id="pw-done">✓ Done & Save</button>
        </div>
    `;
}
