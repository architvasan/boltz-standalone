# Standalone Inference with Boltz

This document describes how to use Boltz models for inference without PyTorch Lightning dependency, allowing you to run predictions directly from Python code.

## Overview

The standalone inference functionality provides:

1. **No PyTorch Lightning dependency** for inference (Lightning is now optional, only needed for training)
2. **Python function interface** - call Boltz directly from your Python code
3. **Simplified model loading** from Lightning checkpoints
4. **Flexible data processing** without Lightning DataModules

## Installation

For inference only (without training capabilities):
```bash
pip install boltz
```

For training capabilities (includes PyTorch Lightning):
```bash
pip install boltz[training]
```

## Quick Start

### Basic Usage

```python
from boltz.inference import predict_structure

# Run inference with a simple function call
results = predict_structure(
    input_data="path/to/manifest.json",
    checkpoint_path="path/to/model.ckpt", 
    target_dir="path/to/targets",
    msa_dir="path/to/msa",
    output_dir="path/to/output"
)
```

### Class-based Usage

```python
from boltz.inference import BoltzInference

# Initialize inference class
inference = BoltzInference(
    checkpoint_path="path/to/model.ckpt",
    device="cuda"
)

# Run inference
results = inference.predict_batch(
    manifest=manifest,
    target_dir=target_dir,
    msa_dir=msa_dir,
    recycling_steps=3,
    sampling_steps=200
)
```

## API Reference

### High-level Functions

#### `predict_structure()`

The main high-level function for structure prediction.

**Parameters:**
- `input_data`: Path to manifest file, Record object, or Manifest object
- `checkpoint_path`: Path to model checkpoint file
- `target_dir`: Directory containing processed structure files
- `msa_dir`: Directory containing MSA files
- `output_dir`: Output directory (optional, if None results are only returned)
- `model_type`: "boltz1", "boltz2", or "auto" (default: "auto")
- `device`: Device to use ("auto", "cpu", "cuda", etc.)
- `recycling_steps`: Number of recycling steps (default: 3)
- `sampling_steps`: Number of diffusion sampling steps (default: 200)
- `diffusion_samples`: Number of diffusion samples (default: 1)
- `output_format`: "pdb" or "mmcif" (default: "mmcif")

**Returns:**
- List of prediction dictionaries

### Classes

#### `BoltzInference`

Main inference class for more control over the prediction process.

**Methods:**
- `__init__(checkpoint_path, model_type="auto", device="auto", **kwargs)`
- `predict_single(record, target_dir, msa_dir, **kwargs)`: Predict single structure
- `predict_batch(manifest, target_dir, msa_dir, **kwargs)`: Predict multiple structures

### Model Loading Utilities

```python
from boltz.model.utils import (
    load_model_from_checkpoint,
    get_checkpoint_info,
    validate_checkpoint_compatibility
)

# Load model directly
model = load_model_from_checkpoint("path/to/checkpoint.ckpt")

# Get checkpoint information
info = get_checkpoint_info("path/to/checkpoint.ckpt")
print(f"Model type: {info['model_type']}")

# Validate compatibility
is_compatible = validate_checkpoint_compatibility("path/to/checkpoint.ckpt", "boltz2")
```

## Examples

### Single Structure Prediction

```python
from boltz.inference import BoltzInference
from boltz.data.types import Record

# Create or load a record
record = Record(id="my_protein", chains=[...])

# Initialize inference
inference = BoltzInference("path/to/checkpoint.ckpt")

# Predict
result = inference.predict_single(
    record=record,
    target_dir="path/to/targets",
    msa_dir="path/to/msa"
)

if "coords" in result:
    coordinates = result["coords"]
    print(f"Predicted coordinates: {coordinates.shape}")
```

### Batch Prediction with Custom Parameters

```python
from boltz.inference import predict_structure

results = predict_structure(
    input_data="manifest.json",
    checkpoint_path="boltz2_model.ckpt",
    target_dir="targets/",
    msa_dir="msa/",
    output_dir="predictions/",
    
    # Custom parameters
    recycling_steps=5,
    sampling_steps=500,
    diffusion_samples=3,
    run_confidence_sequentially=True,
    
    # Additional data
    constraints_dir="constraints/",
    template_dir="templates/",
    mol_dir="molecules/",
    
    # Output options
    output_format="pdb",
    write_embeddings=True
)
```

### Affinity Prediction

```python
from boltz.inference import BoltzInference

# Load affinity model
inference = BoltzInference(
    checkpoint_path="boltz2_affinity.ckpt",
    model_type="boltz2"
)

# Run affinity prediction
results = inference.predict_batch(
    manifest=manifest,
    target_dir=target_dir,
    msa_dir=msa_dir,
    mol_dir=mol_dir,
    affinity=True,  # Enable affinity mode
    recycling_steps=5,
    sampling_steps=200
)

# Extract affinity values
for result in results:
    if "affinity_pred_value" in result:
        affinity = result["affinity_pred_value"].item()
        print(f"Predicted affinity: {affinity}")
```

## Migration from Command Line

### Before (Command Line)
```bash
boltz predict input.yaml --out_dir ./output --devices 1
```

### After (Python)
```python
from boltz.inference import predict_structure

results = predict_structure(
    input_data="input.yaml",
    checkpoint_path="~/.boltz/boltz2_conf.ckpt",  # Default checkpoint
    target_dir="./processed/targets",
    msa_dir="./processed/msa", 
    output_dir="./output",
    device="cuda"
)
```

## Performance Tips

1. **Use GPU**: Set `device="cuda"` for faster inference
2. **Batch processing**: Use `predict_batch()` for multiple structures
3. **Adjust sampling**: Reduce `sampling_steps` for faster inference
4. **Memory management**: Use `max_parallel_samples` to control memory usage

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you have the correct dependencies installed
2. **CUDA out of memory**: Reduce `diffusion_samples` or `max_parallel_samples`
3. **Model loading errors**: Check that the checkpoint path is correct and the model type matches

### Getting Help

- Check the example file: `examples/standalone_inference.py`
- Use `get_checkpoint_info()` to inspect checkpoint files
- Enable verbose logging for debugging

## Compatibility

- **Checkpoint compatibility**: All existing Boltz Lightning checkpoints are supported
- **Model versions**: Both Boltz1 and Boltz2 models are supported
- **Auto-detection**: Model type is automatically detected from checkpoints
- **Device flexibility**: Supports CPU and GPU inference

## What's Changed

1. **PyTorch Lightning is now optional** - only needed for training
2. **New Python API** - direct function calls instead of command line
3. **Simplified model loading** - automatic checkpoint conversion
4. **Flexible data processing** - no Lightning DataModules required
5. **Better error handling** - graceful handling of memory issues

The command line interface (`boltz` command) still works as before for backward compatibility.
