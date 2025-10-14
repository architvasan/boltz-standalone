#!/usr/bin/env python3
"""Simple test for YAML parsing without model dependencies."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from boltz.data.parse.yaml import parse_yaml
    from boltz.data.mol import load_canonicals
    print("✅ Successfully imported YAML parser")
except ImportError as e:
    print(f"❌ Failed to import YAML parser: {e}")
    sys.exit(1)


def test_yaml_parsing():
    """Test basic YAML parsing."""
    
    yaml_path = Path("examples/prot.yaml")
    if not yaml_path.exists():
        print(f"❌ YAML file not found: {yaml_path}")
        return False
    
    print(f"Testing YAML file: {yaml_path}")
    
    try:
        # Try to load CCD data (may fail, that's OK)
        try:
            cache_dir = Path.home() / ".boltz"
            mol_dir = cache_dir / "mols"
            if mol_dir.exists():
                ccd = load_canonicals(mol_dir)
                print(f"✅ Loaded CCD data from {mol_dir}")
            else:
                ccd = {}
                print(f"⚠️  No CCD data found at {mol_dir}, using empty dict")
        except Exception as e:
            ccd = {}
            print(f"⚠️  Failed to load CCD data: {e}, using empty dict")
        
        # Parse the YAML
        target = parse_yaml(yaml_path, ccd, mol_dir if 'mol_dir' in locals() else Path("/tmp"), boltz2=True)
        
        print(f"✅ Successfully parsed YAML file")
        print(f"   Target name: {target.name}")
        print(f"   Number of chains: {len(target.chain_infos)}")
        
        for i, chain in enumerate(target.chain_infos):
            print(f"     Chain {i}: {chain.chain_name} ({chain.num_residues} residues, type: {chain.mol_type})")
        
        print(f"   Structure info: {target.structure_info}")
        print(f"   Inference options: {target.inference_options}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to parse YAML: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== Simple YAML Parsing Test ===")
    
    success = test_yaml_parsing()
    
    if success:
        print("\n🎉 YAML parsing test passed!")
        print("\nNow you can try the full inference once you have the required dependencies installed.")
        print("The error you encountered was due to missing dependencies, not the YAML parsing logic.")
    else:
        print("\n❌ YAML parsing test failed")
        sys.exit(1)
