import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def test_license_link_points_to_dler_repo():
    text = README.read_text(encoding="utf-8")
    assert "https://github.com/nbasyl/DoRA/LICENSE" not in text
    assert re.search(
