import lxml.html

from utils.utils import count_unique_tags, generate_selector_for_element, get_unique_xpath, normalize_tag


def test_normalize_tag():
    assert normalize_tag('DIV') == 'div'
    assert normalize_tag(None) == 'unknown'
    assert normalize_tag(123) == 'unknown'

def test_get_unique_xpath():
    html = '<div><span>Test</span></div>'
    elem = lxml.html.fromstring(html)
    span = elem.xpath('.//span')[0]
    xpath = get_unique_xpath(span)
    assert xpath.endswith('/span')

def test_count_unique_tags():
    assert count_unique_tags(['a', 'b', 'a']) == 2
    assert count_unique_tags([]) == 0
    assert count_unique_tags(None) == 0

def test_generate_selector_for_element():
    html = '<div id="main"><span class="foo">Test</span></div>'
    elem = lxml.html.fromstring(html)
    span = elem.xpath('.//span')[0]
    sel = generate_selector_for_element(span)
    assert 'span' in sel
    div = elem
    sel_div = generate_selector_for_element(div)
    assert 'div#main' in sel_div or 'div' in sel_div
