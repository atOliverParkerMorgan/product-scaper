import pandas as pd

from train_model.process_data import get_main_html_content_tag, html_to_dataframe


def test_get_main_html_content_tag_basic():
    html = "<html><body><div><h1>Title</h1><p>Some text</p></div></body></html>"
    elem = get_main_html_content_tag(html)
    assert elem is not None
    assert isinstance(elem.tag, str)


def test_get_main_html_content_tag_empty():
    assert get_main_html_content_tag("") is None
    assert get_main_html_content_tag("<html></html>") is not None


def test_html_to_dataframe_basic():
    html = "<html><body><div><h1>Title</h1><p>Some text</p></div></body></html>"
    df = html_to_dataframe(html, selectors={})
    assert isinstance(df, pd.DataFrame)


def test_html_to_dataframe_with_selectors():
    html = "<html><body><div><h1>Title</h1><p>Some text</p></div></body></html>"
    selectors = {"title": ["//h1"]}
    df = html_to_dataframe(html, selectors=selectors)
    assert isinstance(df, pd.DataFrame)
    # Should have at least one row for the h1
    assert (df["Category"] == "title").any()
