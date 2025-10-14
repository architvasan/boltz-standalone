#!/usr/bin/env python3
"""Test script for YAML input processing."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from boltz.inference.inference import _process_input_data


def test_yaml_processing():
    """Test YAML input processing."""
    
    # Test the YAML processing function directly first
    print("Testing YAML processing...")
    
    yaml_path = Path("examples/prot.yaml")
    if not yaml_path.exists():
        print(f"❌ YAML file not found: {yaml_path}")
        return False
    
    try:
        # Test the input processing function
        manifest = _process_input_data(
            input_data=yaml_path,
            mol_dir=None,  # Will use default
            model_version="boltz2"
        )
        
        print(f"✅ Successfully processed YAML file")
        print(f"   Number of records: {len(manifest.records)}")
        
        for i, record in enumerate(manifest.records):
            print(f"   Record {i}: {record.id}")
            print(f"     Chains: {len(record.chains)}")
            for j, chain in enumerate(record.chains):
                print(f"       Chain {j}: {chain.chain_name} ({chain.num_residues} residues)")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to process YAML: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_inference():
    """Test full inference with YAML input."""
    
    print("\nTesting full inference...")
    
    # Create test directories
    test_dirs = [
        "test_fold/targets",
        "test_fold/msa", 
        "output"
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Check if checkpoint exists
    checkpoint_path = "model_checkpoints/boltz2_conf.ckpt"
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please download the checkpoint first or update the path")
        return False
    
    try:
        from boltz.inference import predict_structure
        
        print("Running inference...")
        results = predict_structure(
            input_data="examples/prot.yaml",
            checkpoint_path=checkpoint_path,
            target_dir="test_fold/targets/",
            msa_dir="test_fold/msa",
            output_dir="output"
        )
        
        print(f"✅ Inference completed successfully!")
        print(f"   Number of results: {len(results)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== Testing YAML Input Processing ===")
    
    # Test YAML processing first
    yaml_success = test_yaml_processing()
    
    if yaml_success:
        # Test full inference if YAML processing works
        inference_success = test_full_inference()
        
        if inference_success:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Inference test failed")
            sys.exit(1)
    else:
        print("\n❌ YAML processing test failed")
        sys.exit(1)
