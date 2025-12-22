import pandas as pd

from product_scraper.train_model.train_model import build_pipeline, train_model


def test_build_pipeline():
    num_cols = ["num1"]
    cat_cols = ["cat1"]
    text_cols = ["class_str"]
    pipe = build_pipeline(num_cols, cat_cols, text_cols)
    assert hasattr(pipe, "fit")
    assert hasattr(pipe, "predict")


def test_train_model_target_missing(capsys):
    df = pd.DataFrame({"num1": [1, 2], "cat1": ["a", "b"]})
    try:
        train_model(df)
    except KeyError as e:
        assert "Category" in str(e)
    else:
        assert False, "Expected KeyError for missing 'Category' column"
