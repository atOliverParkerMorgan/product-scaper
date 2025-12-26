from pathlib import Path

import pytest
from pylint import lint, reporters


@pytest.fixture(scope="session", name="linter")
def fixture_linter() -> None:
    """Use pylint to test codestyle for src file."""
    python_files = (Path(__file__).parent.parent / "src").rglob("*.py")
    rep = reporters.CollectingReporter()
    # disabled warnings:
    # C0301 line too long
    # C0103 variables name (does not like shorter than 2 chars)
    # W0123 eval used (useful when loading districts)
    for file in python_files:
        r = lint.Run(
            ["--disable=C0301,C0103,W0123", "-sn", str(file)], reporter=rep, exit=False
        )
        return r.linter


@pytest.mark.parametrize("limit", range(3, 11))
def test_codestyle_score(linter: lint.pylinter.PyLinter, limit: int) -> None:
    """Evaluate codestyle for different thresholds."""
    score = linter.stats.global_note
    assert score >= limit, f"Codestyle score {score} is lower than {limit}"
