#!/usr/bin/env python3
"""Script to download Boltz checkpoint weights."""

import sys
from pathlib import Path

# Add the src directory to the path so we can import boltz
sys.path.insert(0, str(Path(__file__).parent / "src"))

from boltz.main import download_boltz1, download_boltz2


def main():
    """Download all Boltz checkpoints."""
    # Use the default cache directory
    cache_dir = Path.home() / ".boltz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using cache directory: {cache_dir}")
    
    print("\n=== Downloading Boltz1 checkpoints ===")
    try:
        download_boltz1(cache_dir)
        print("✅ Boltz1 download completed successfully")
    except Exception as e:
        print(f"❌ Boltz1 download failed: {e}")
        return 1
    
    print("\n=== Downloading Boltz2 checkpoints ===")
    try:
        download_boltz2(cache_dir)
        print("✅ Boltz2 download completed successfully")
    except Exception as e:
        print(f"❌ Boltz2 download failed: {e}")
        return 1
    
    print("\n=== Download Summary ===")
    files = list(cache_dir.glob("*"))
    for file in sorted(files):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name}: {size_mb:.1f} MB")
        else:
            print(f"  {file.name}/: (directory)")
    
    print(f"\nAll checkpoints downloaded to: {cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
