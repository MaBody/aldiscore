import importlib.metadata
import re
import shutil
from pathlib import Path

import aldiscore

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_package_data_present():
    assert (aldiscore.ROOT / "configs" / "config.yaml").is_file()
    assert (aldiscore.ROOT / "models" / "v1.0_aa.txt").is_file()
    assert (aldiscore.ROOT / "models" / "v1.0_dna.txt").is_file()


def test_version_matches_pyproject():
    match = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(), re.MULTILINE)
    assert match is not None
    assert importlib.metadata.version("aldiscore") == match.group(1)


def test_console_script_installed():
    assert shutil.which("aldiscore") is not None
