import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session", autouse=True)
def _assets_present():
    """If neither real nor placeholder assets exist, generate the placeholders."""
    from src.inference import ASSETS, asset_path
    if not asset_path("models/aware_fusion.pt").exists() or not asset_path("manifest.json").exists():
        from src import placeholders
        placeholders.main()


@pytest.fixture(scope="session")
def model():
    from src.inference import load_model
    return load_model()


@pytest.fixture(scope="session")
def manifest():
    from src.inference import load_manifest
    return load_manifest()


@pytest.fixture(scope="session")
def chip(manifest):
    from src.inference import load_chip
    return load_chip(manifest[0]["chip_id"])
