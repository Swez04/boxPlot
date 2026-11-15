from .base import BaseDetector
from .dbscan import DBSCANDetector
from .iqr import IQRDetector
from .isolation_forest import IsolationForestDetector

__all__ = ["BaseDetector", "IQRDetector", "IsolationForestDetector", "DBSCANDetector"]

# The AutoEncoder needs PyTorch; keep the other detectors importable without it.
try:
    from .autoencoder import AutoEncoderDetector  # noqa: F401

    __all__.append("AutoEncoderDetector")
except ImportError:  # pragma: no cover
    pass
