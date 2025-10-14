"""Utilities for loading model checkpoints without PyTorch Lightning dependency."""

import torch
from pathlib import Path
from typing import Union, Optional, Dict, Any

from boltz.model.models.boltz1_standalone import Boltz1Standalone
from boltz.model.models.boltz2_standalone import Boltz2Standalone


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model_type: str = "auto",
    map_location: str = "cpu",
    strict: bool = True,
    **kwargs
) -> Union[Boltz1Standalone, Boltz2Standalone]:
    """Load a Boltz model from a PyTorch Lightning checkpoint.
    
    Parameters
    ----------
    checkpoint_path : Union[str, Path]
        Path to the checkpoint file.
    model_type : str
        Type of model to load. Can be "boltz1", "boltz2", or "auto" to detect automatically.
    map_location : str
        Device to load the model on.
    strict : bool
        Whether to strictly enforce that the keys in state_dict match.
    **kwargs
        Additional arguments to override model configuration.
        
    Returns
    -------
    Union[Boltz1Standalone, Boltz2Standalone]
        The loaded model.
        
    Raises
    ------
    ValueError
        If model_type is not recognized or cannot be auto-detected.
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load the checkpoint to inspect it
    # Handle PyTorch 2.6+ security changes for Lightning checkpoints
    try:
        # Try with weights_only=False for Lightning checkpoints (they contain OmegaConf objects)
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except Exception as e:
        # Fallback: try with safe globals if the above fails
        try:
            import omegaconf
            torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        except Exception:
            # Final fallback: use the original error message
            raise RuntimeError(
                f"Failed to load checkpoint {checkpoint_path}. "
                f"This is likely due to PyTorch 2.6+ security changes. "
                f"Original error: {e}"
            ) from e
    
    # Auto-detect model type if needed
    if model_type == "auto":
        model_type = detect_model_type(checkpoint)
    
    # Load the appropriate model
    if model_type == "boltz1":
        return Boltz1Standalone.from_lightning_checkpoint(
            str(checkpoint_path), map_location=map_location, strict=strict, **kwargs
        )
    elif model_type == "boltz2":
        return Boltz2Standalone.from_lightning_checkpoint(
            str(checkpoint_path), map_location=map_location, strict=strict, **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'boltz1', 'boltz2', or 'auto'.")


def detect_model_type(checkpoint: Dict[str, Any]) -> str:
    """Detect the model type from a checkpoint.
    
    Parameters
    ----------
    checkpoint : Dict[str, Any]
        The loaded checkpoint dictionary.
        
    Returns
    -------
    str
        The detected model type ("boltz1" or "boltz2").
        
    Raises
    ------
    ValueError
        If the model type cannot be determined.
    """
    # Check hyperparameters for model-specific indicators
    hparams = checkpoint.get("hyper_parameters", {})
    
    # Look for Boltz2-specific features
    boltz2_indicators = [
        "affinity_prediction",
        "use_templates_v2", 
        "diffusion_conditioning",
        "contact_conditioning",
        "predict_bfactor"
    ]
    
    # Check if any Boltz2-specific parameters are present
    for indicator in boltz2_indicators:
        if indicator in hparams:
            return "boltz2"
    
    # Check state dict for model-specific modules
    state_dict = checkpoint.get("state_dict", {})
    
    # Look for Boltz2-specific modules in state dict
    boltz2_modules = [
        "affinity_module",
        "diffusion_conditioning",
        "contact_conditioning",
        "bfactor_module",
        "template_module"
    ]
    
    for module_name in boltz2_modules:
        if any(key.startswith(module_name) for key in state_dict.keys()):
            return "boltz2"
    
    # Look for Boltz1-specific patterns
    # If we don't find Boltz2 indicators, assume Boltz1
    # This is a reasonable default since Boltz1 is simpler
    
    # Additional check: look at the class name if available
    if "pytorch-lightning_version" in checkpoint:
        # This suggests it's a Lightning checkpoint
        # Check for specific model class indicators
        pass
    
    # Default to Boltz1 if we can't determine otherwise
    # Most legacy checkpoints will be Boltz1
    return "boltz1"


def convert_lightning_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert a PyTorch Lightning state dict to a standalone model state dict.
    
    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
        The Lightning model state dict.
        
    Returns
    -------
    Dict[str, torch.Tensor]
        The converted state dict for standalone model.
    """
    converted_state_dict = {}
    
    # Remove Lightning-specific prefixes and keys
    lightning_prefixes = ["_metrics", "trainer", "logger"]
    
    for key, value in state_dict.items():
        # Skip Lightning-specific keys
        if any(key.startswith(prefix) for prefix in lightning_prefixes):
            continue
            
        # Keep the key as-is for now
        # The standalone models should have the same parameter names
        converted_state_dict[key] = value
    
    return converted_state_dict


def load_checkpoint_weights_only(
    checkpoint_path: Union[str, Path],
    map_location: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """Load only the model weights from a checkpoint.

    Parameters
    ----------
    checkpoint_path : Union[str, Path]
        Path to the checkpoint file.
    map_location : str
        Device to load the weights on.

    Returns
    -------
    Dict[str, torch.Tensor]
        The model state dict.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except Exception:
        try:
            import omegaconf
            torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}") from e

    state_dict = checkpoint.get("state_dict", {})
    return convert_lightning_state_dict(state_dict)


def get_checkpoint_info(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about a checkpoint file.

    Parameters
    ----------
    checkpoint_path : Union[str, Path]
        Path to the checkpoint file.

    Returns
    -------
    Dict[str, Any]
        Information about the checkpoint including model type, hyperparameters, etc.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        try:
            import omegaconf
            torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}") from e
    
    info = {
        "model_type": detect_model_type(checkpoint),
        "hyperparameters": checkpoint.get("hyper_parameters", {}),
        "pytorch_lightning_version": checkpoint.get("pytorch-lightning_version", "unknown"),
        "epoch": checkpoint.get("epoch", "unknown"),
        "global_step": checkpoint.get("global_step", "unknown"),
        "state_dict_keys": list(checkpoint.get("state_dict", {}).keys())[:10],  # First 10 keys
        "has_ema": "ema" in checkpoint,
    }
    
    return info


def validate_checkpoint_compatibility(
    checkpoint_path: Union[str, Path],
    model_type: str
) -> bool:
    """Validate that a checkpoint is compatible with the specified model type.

    Parameters
    ----------
    checkpoint_path : Union[str, Path]
        Path to the checkpoint file.
    model_type : str
        Expected model type ("boltz1" or "boltz2").

    Returns
    -------
    bool
        True if compatible, False otherwise.
    """
    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            import omegaconf
            torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        detected_type = detect_model_type(checkpoint)
        return detected_type == model_type
    except Exception:
        return False
