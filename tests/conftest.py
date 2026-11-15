import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # so `api` is importable


@pytest.fixture
def data_with_outliers():
    """200 well-behaved points plus 3 obvious outliers (indices 200-202)."""
    rng = np.random.default_rng(0)
    normal = rng.normal(loc=50, scale=5, size=200)
    return np.concatenate([normal, [150.0, -80.0, 200.0]])


@pytest.fixture
def data_2d():
    rng = np.random.default_rng(1)
    normal = rng.normal(0, 1, size=(300, 3))
    outliers = np.array([[15, 15, 15], [-12, 14, -13]], dtype=float)
    return np.vstack([normal, outliers])


@pytest.fixture
def csv_dataset(tmp_path, monkeypatch):
    """A small CSV inside a sandboxed data dir; returns its file name."""
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "sensor_id": range(300),
        "temp": rng.normal(20, 2, 300),
        "pressure": rng.normal(1000, 4, 300),
    })
    df.loc[[7, 8], ["temp", "pressure"]] = [[80, 900], [-30, 1100]]
    monkeypatch.setenv("ANOMALY_DATA_DIR", str(tmp_path))
    df.to_csv(tmp_path / "sensors.csv", index=False)
    return "sensors.csv"
