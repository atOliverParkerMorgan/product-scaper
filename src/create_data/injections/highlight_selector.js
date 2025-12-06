(selector) => {
    try {
        document.querySelectorAll(selector).forEach(el => {
            el.classList.add('pw-selected');
        });
        return true;
    } catch (e) {
        console.warn('Could not highlight selector:', selector, e);
        return false;
    }
}
