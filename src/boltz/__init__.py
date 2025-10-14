"""Boltz - Biomolecular Structure Prediction."""

from importlib.metadata import PackageNotFoundError, version

try:  # noqa: SIM105
    __version__ = version("boltz")
except PackageNotFoundError:
    # package is not installed
    pass

# Standalone inference (no PyTorch Lightning required)
try:
    from boltz.inference import BoltzInference, predict_structure
    from boltz.model.utils import load_model_from_checkpoint, get_checkpoint_info
    from boltz.model.models.boltz1_standalone import Boltz1Standalone
    from boltz.model.models.boltz2_standalone import Boltz2Standalone

    __all__ = [
        "BoltzInference",
        "predict_structure",
        "load_model_from_checkpoint",
        "get_checkpoint_info",
        "Boltz1Standalone",
        "Boltz2Standalone",
    ]
except ImportError:
    # Some dependencies might not be available
    __all__ = []
