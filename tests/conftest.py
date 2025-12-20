"""Pytest configuration for test suite."""
import sys
from pathlib import Path
import pytest


# Add src directory to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_html_content():
    """Provides a basic HTML structure for testing."""
    return """
    <html>
        <body>
            <div id="target" class="item">Click Me</div>
            <div class="pw-predicted" id="p1">Predicted</div>
        </body>
    </html>
    """