"""
Example usage of Boltz standalone inference without PyTorch Lightning.

This example demonstrates how to use the new Python function interface
for running Boltz inference directly from Python code.
"""

from pathlib import Path
from boltz.inference import BoltzInference, predict_structure
from boltz.data.types import Manifest, Record


def example_basic_inference():
    """Basic example of running inference with the standalone interface."""
    
    # Paths to your data
    checkpoint_path = "path/to/your/boltz_checkpoint.ckpt"
    target_dir = Path("path/to/processed/targets")
    msa_dir = Path("path/to/processed/msa")
    output_dir = Path("path/to/output")
    
    # Load manifest (or create a single record)
    manifest_path = "path/to/your/manifest.json"
    
    # Option 1: Use the high-level predict_structure function
    results = predict_structure(
        input_data=manifest_path,
        checkpoint_path=checkpoint_path,
        target_dir=target_dir,
        msa_dir=msa_dir,
        output_dir=output_dir,
        model_type="auto",  # Auto-detect model type
        device="auto",      # Auto-detect device (GPU if available)
        recycling_steps=3,
        sampling_steps=200,
        diffusion_samples=1,
        output_format="mmcif",
    )
    
    print(f"Processed {len(results)} structures")
    return results


def example_class_based_inference():
    """Example using the BoltzInference class for more control."""
    
    # Initialize the inference class
    inference = BoltzInference(
        checkpoint_path="path/to/your/boltz_checkpoint.ckpt",
        model_type="boltz2",  # Specify model type explicitly
        device="cuda",       # Use GPU
    )
    
    # Load your data
    manifest = Manifest.load("path/to/your/manifest.json")
    target_dir = Path("path/to/processed/targets")
    msa_dir = Path("path/to/processed/msa")
    
    # Run inference on all records
    results = inference.predict_batch(
        manifest=manifest,
        target_dir=target_dir,
        msa_dir=msa_dir,
        recycling_steps=5,      # More recycling steps for better quality
        sampling_steps=300,     # More sampling steps
        diffusion_samples=3,    # Multiple samples
    )
    
    # Process results
    for i, result in enumerate(results):
        if not result.get("exception", False):
            record = manifest.records[i]
            print(f"Successfully processed {record.id}")
            
            # Access prediction outputs
            if "coords" in result:
                coords = result["coords"]
                print(f"  Predicted coordinates shape: {coords.shape}")
            
            if "plddt" in result:
                plddt = result["plddt"]
                print(f"  Confidence scores available")
                
    return results


def example_single_structure():
    """Example of processing a single structure."""
    
    # Create a single record (you would typically load this from your data)
    record = Record(
        id="my_protein",
        chains=[],  # Your chain information
        # ... other record fields
    )
    
    # Initialize inference
    inference = BoltzInference(
        checkpoint_path="path/to/your/boltz_checkpoint.ckpt",
        device="auto",
    )
    
    # Run inference on single record
    result = inference.predict_single(
        record=record,
        target_dir=Path("path/to/processed/targets"),
        msa_dir=Path("path/to/processed/msa"),
        recycling_steps=3,
        sampling_steps=200,
    )
    
    if not result.get("exception", False):
        print("Prediction successful!")
        return result
    else:
        print("Prediction failed")
        return None


def example_with_constraints():
    """Example with constraints and additional options."""
    
    results = predict_structure(
        input_data="path/to/your/manifest.json",
        checkpoint_path="path/to/your/boltz_checkpoint.ckpt",
        target_dir="path/to/processed/targets",
        msa_dir="path/to/processed/msa",
        output_dir="path/to/output",
        
        # Additional directories
        constraints_dir="path/to/constraints",
        template_dir="path/to/templates",
        mol_dir="path/to/molecules",
        
        # Inference parameters
        recycling_steps=5,
        sampling_steps=500,
        diffusion_samples=5,
        run_confidence_sequentially=True,
        
        # Output options
        output_format="pdb",
        write_embeddings=True,
        
        # Performance
        num_workers=4,
    )
    
    return results


def example_affinity_prediction():
    """Example of running affinity prediction."""
    
    # Load a Boltz2 affinity model
    inference = BoltzInference(
        checkpoint_path="path/to/boltz2_affinity_checkpoint.ckpt",
        model_type="boltz2",
    )
    
    # Load manifest with affinity records
    manifest = Manifest.load("path/to/affinity_manifest.json")
    
    # Run affinity prediction
    results = inference.predict_batch(
        manifest=manifest,
        target_dir=Path("path/to/processed/targets"),
        msa_dir=Path("path/to/processed/msa"),
        mol_dir=Path("path/to/molecules"),
        affinity=True,  # Enable affinity mode
        recycling_steps=5,
        sampling_steps=200,
        diffusion_samples=3,
    )
    
    # Extract affinity predictions
    for i, result in enumerate(results):
        if not result.get("exception", False):
            record = manifest.records[i]
            if "affinity_pred_value" in result:
                affinity = result["affinity_pred_value"].item()
                probability = result["affinity_probability_binary"].item()
                print(f"{record.id}: Affinity = {affinity:.3f}, Probability = {probability:.3f}")
    
    return results


def example_model_info():
    """Example of getting information about a checkpoint."""
    
    from boltz.model.utils import get_checkpoint_info, validate_checkpoint_compatibility
    
    checkpoint_path = "path/to/your/checkpoint.ckpt"
    
    # Get checkpoint information
    info = get_checkpoint_info(checkpoint_path)
    print("Checkpoint Information:")
    print(f"  Model type: {info['model_type']}")
    print(f"  Epoch: {info['epoch']}")
    print(f"  Global step: {info['global_step']}")
    print(f"  Has EMA: {info['has_ema']}")
    print(f"  Lightning version: {info['pytorch_lightning_version']}")
    
    # Validate compatibility
    is_compatible = validate_checkpoint_compatibility(checkpoint_path, "boltz2")
    print(f"  Compatible with Boltz2: {is_compatible}")
    
    return info


if __name__ == "__main__":
    # Run the basic example
    print("Running basic inference example...")
    
    # Note: You'll need to update the paths to point to your actual data
    # example_basic_inference()
    
    print("Example complete! Check the functions above for different usage patterns.")
    print("\nAvailable functions:")
    print("- example_basic_inference(): High-level function usage")
    print("- example_class_based_inference(): Class-based usage with more control")
    print("- example_single_structure(): Process a single structure")
    print("- example_with_constraints(): Usage with constraints and templates")
    print("- example_affinity_prediction(): Affinity prediction example")
    print("- example_model_info(): Get checkpoint information")
