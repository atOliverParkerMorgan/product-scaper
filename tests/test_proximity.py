""" Tests for proximity score calculation logic. """

from train_model.predict_data import calculate_proximity_score


def test_proximity_logic():
    xpath_anchor = "/html/body/div[1]/div[2]/ul/li[1]"
    xpath_close = "/html/body/div[1]/div[2]/ul/li[2]"  # Same parent, next sibling
    xpath_far = "/html/body/footer/div"               # Different branch

    score_close = calculate_proximity_score(xpath_anchor, xpath_close)
    score_far = calculate_proximity_score(xpath_anchor, xpath_far)

    # Tree distance should be 2 for siblings (1 up to ul, 1 down to li[2])
    assert score_close[0] == 2
    # Index delta should be |1 - 2| = 1
    assert score_close[1] == 1
    # Far score tree distance should be much higher
    assert score_far[0] > score_close[0]
