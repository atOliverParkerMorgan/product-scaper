"""Tests for proximity score calculation logic."""

from product_scraper.train_model.predict_data import calculate_proximity_score


def test_proximity_logic():
    # Sibling case
    xp1 = "/html/body/div[1]/div[2]/ul/li[1]"
    xp2 = "/html/body/div[1]/div[2]/ul/li[2]"
    tree_dist, idx_delta = calculate_proximity_score(xp1, xp2)
    # Siblings: tree_distance=2 (up to ul, down to li[2]), index_delta=1
    assert tree_dist == 2
    assert idx_delta == 1

    # Different branches
    xp3 = "/html/body/footer/div"
    tree_dist_far, idx_far = calculate_proximity_score(xp1, xp3)
    assert tree_dist_far > tree_dist

    # Ancestor/descendant: direct ancestor
    ancestor = "/html/body/div[1]/div[2]/ul"
    descendant = "/html/body/div[1]/div[2]/ul/li[3]"
    tree_dist_anc, idx_anc = calculate_proximity_score(ancestor, descendant)
    # Should be 1 up or down, index_delta=0 (direct ancestor)
    assert tree_dist_anc == 1
    assert idx_anc == 0

    # Identical
    identical = "/html/body/div[1]/div[2]/ul/li[1]"
    tree_dist_id, idx_id = calculate_proximity_score(identical, identical)
    assert tree_dist_id == 0
    assert idx_id == 0

    # Normalization: li vs li[1]
    norm1 = "/html/body/div/ul/li"
    norm2 = "/html/body/div/ul/li[1]"
    tree_dist_norm, idx_norm = calculate_proximity_score(norm1, norm2)
    assert tree_dist_norm == 0
    assert idx_norm == 0

    # Divergent indices: li vs li[3]
    norm3 = "/html/body/div/ul/li"
    norm4 = "/html/body/div/ul/li[3]"
    tree_dist_div, idx_div = calculate_proximity_score(norm3, norm4)
    assert tree_dist_div == 2
    assert idx_div == 2

    # Partial overlap: prefix/suffix
    prefix = "/html/body/div/ul"
    suffix = "/html/body/div/ul/li[2]"
    tree_dist_pref, idx_pref = calculate_proximity_score(prefix, suffix)
    assert tree_dist_pref == 1
    assert idx_pref == 0


def test_proximity_identical():
    from train_model.predict_data import calculate_proximity_score

    xp = "/html/body/div[1]/div[2]/ul/li[1]"
    # Identical xpaths should have distance 0, index delta 0
    dist, idx = calculate_proximity_score(xp, xp)
    assert dist == 0
    assert idx == 0


def test_proximity_different_depths():
    from train_model.predict_data import calculate_proximity_score

    xp1 = "/html/body/div[1]/div[2]/ul/li[1]"
    xp2 = "/html/body/div[1]/div[2]/ul"
    # Parent-child: should have distance 1 (li[1] up to ul)
    dist, idx = calculate_proximity_score(xp1, xp2)
    assert dist == 1
    assert idx == 0


def test_proximity_index_extraction():
    from train_model.predict_data import calculate_proximity_score

    # XPaths with no explicit index
    xp1 = "/html/body/div/ul/li"
    xp2 = "/html/body/div/ul/li[3]"
    dist, idx = calculate_proximity_score(xp1, xp2)
    # Should still parse index correctly (default 1 for no index)
    assert dist == 2
    assert idx == 2


def test_proximity_index_normalization():
    from train_model.predict_data import calculate_proximity_score

    # XPaths with and without explicit indices, but same logical node
    xp1 = "/html/body/div/ul/li"
    xp2 = "/html/body/div/ul/li[1]"
    # Should be treated as identical after normalization
    dist, idx = calculate_proximity_score(xp1, xp2)
    assert dist == 0
    assert idx == 0


def test_proximity_divergent_indices():
    from train_model.predict_data import calculate_proximity_score

    # XPaths diverge at a segment with and without index
    xp1 = "/html/body/div/ul/li"
    xp2 = "/html/body/div/ul/li[5]"
    dist, idx = calculate_proximity_score(xp1, xp2)
    # Should diverge at li, so tree_distance=0, index delta=2
    assert dist == 2
    assert idx == 4


def test_proximity_partial_overlap():
    from train_model.predict_data import calculate_proximity_score

    # XPaths with partial overlap, one is a prefix of the other
    xp1 = "/html/body/div/ul"
    xp2 = "/html/body/div/ul/li[2]"
    dist, idx = calculate_proximity_score(xp1, xp2)
    # Should diverge at li[2], so tree_distance=1, index delta=1
    assert dist == 1
    assert idx == 0
