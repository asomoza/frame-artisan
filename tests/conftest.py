import os

# Use Qt offscreen platform so tests can run without a display server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Suppress tqdm progress bars (model loading, downloads, etc.)
os.environ.setdefault("TQDM_DISABLE", "1")


import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-skip gpu_light / gpu_heavy tests unless the corresponding env var is set."""
    skip_light = pytest.mark.skip(reason="GPU_LIGHT not set — skipping light GPU test")
    skip_heavy = pytest.mark.skip(reason="GPU_HEAVY not set — skipping heavy GPU test")

    run_light = os.environ.get("GPU_LIGHT", "") == "1"
    run_heavy = os.environ.get("GPU_HEAVY", "") == "1"

    for item in items:
        if "gpu_light" in item.keywords and not run_light:
            item.add_marker(skip_light)
        if "gpu_heavy" in item.keywords and not run_heavy:
            item.add_marker(skip_heavy)


@pytest.fixture(scope="session")
def app_db_path():
    """Read QSettings to find the app database path. Skip if not configured."""
    from PyQt6.QtCore import QSettings

    settings = QSettings("ZCode", "FrameArtisan")
    data_path = settings.value("data_path", "")
    if not data_path:
        pytest.skip("QSettings data_path not configured — cannot locate app.db")

    db_path = os.path.join(data_path, "app.db")
    if not os.path.isfile(db_path):
        pytest.skip(f"app.db not found at {db_path}")

    return db_path


@pytest.fixture(scope="session")
def models_diffusers_path():
    """Read QSettings to find the models_diffusers directory. Skip if not configured."""
    from PyQt6.QtCore import QSettings

    settings = QSettings("ZCode", "FrameArtisan")
    models_path = settings.value("models_diffusers", "")
    if not models_path or not os.path.isdir(models_path):
        pytest.skip("QSettings models_diffusers not configured or missing")

    return models_path


TINY_MODEL_REPO = "OzzyGT/tiny_LTX2"


@pytest.fixture(scope="session")
def tiny_model_path():
    """Download the tiny model once per session and return its local path."""
    from huggingface_hub import snapshot_download

    return snapshot_download(TINY_MODEL_REPO)


@pytest.fixture(scope="session")
def tiny_upsampler_path(tiny_model_path):
    """Return the path to the tiny latent upsampler inside the tiny_LTX2 repo."""
    return os.path.join(tiny_model_path, "latent_upsampler")
