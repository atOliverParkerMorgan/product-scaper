() => {
    let currentHover = null;
    
    document.addEventListener('mouseover', (e) => {
        // Ignore UI elements
        if (e.target.closest('#pw-ui')) return;
        // Remove previous hover
        if (currentHover && !currentHover.classList.contains('pw-selected')) {
            currentHover.classList.remove('pw-hover');
        }
        
        // Add new hover
        currentHover = e.target;
        if (!currentHover.classList.contains('pw-selected')) {
            currentHover.classList.add('pw-hover');
        }
    });
    
    document.addEventListener('mouseout', (e) => {
        if (e.target && !e.target.classList.contains('pw-selected')) {
            e.target.classList.remove('pw-hover');
        }
    });

    document.addEventListener('mousedown', (e) => {
        console.log('Clicked element:', e.target);
        console.log('Element tag:', e.target.tagName);
        console.log('Element classes:', e.target.className);
        console.log('Element id:', e.target.id);
    });
}
