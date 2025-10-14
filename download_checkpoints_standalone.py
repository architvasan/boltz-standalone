#!/usr/bin/env python3
"""Standalone script to download Boltz checkpoint weights without PyTorch Lightning dependency."""

import urllib.request
import tarfile
from pathlib import Path


# URLs from the original main.py
CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MOL_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"

BOLTZ1_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz1_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt",
]

BOLTZ2_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
]

BOLTZ2_AFFINITY_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_aff.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
]


def download_with_fallback(urls, destination, description):
    """Download a file with fallback URLs."""
    print(f"Downloading {description} to {destination}")
    
    for i, url in enumerate(urls):
        try:
            print(f"  Trying URL {i+1}/{len(urls)}: {url}")
            urllib.request.urlretrieve(url, str(destination))
            print(f"  ✅ Successfully downloaded from {url}")
            return
        except Exception as e:
            print(f"  ❌ Failed to download from {url}: {e}")
            if i == len(urls) - 1:
                raise RuntimeError(f"Failed to download {description} from all URLs. Last error: {e}") from e
            continue


def download_boltz1_standalone(cache: Path) -> None:
    """Download Boltz1 data without PyTorch Lightning dependency."""
    print("=== Downloading Boltz1 ===")
    
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        print(f"Downloading CCD dictionary to {ccd}")
        urllib.request.urlretrieve(CCD_URL, str(ccd))
        print("✅ CCD dictionary downloaded")
    else:
        print("✅ CCD dictionary already exists")

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        download_with_fallback(BOLTZ1_URL_WITH_FALLBACK, model, "Boltz1 model weights")
    else:
        print("✅ Boltz1 model weights already exist")


def download_boltz2_standalone(cache: Path) -> None:
    """Download Boltz2 data without PyTorch Lightning dependency."""
    print("=== Downloading Boltz2 ===")
    
    # Download CCD/molecules
    mols = cache / "mols"
    tar_mols = cache / "mols.tar"
    if not tar_mols.exists():
        print(f"Downloading molecule data to {tar_mols}")
        urllib.request.urlretrieve(MOL_URL, str(tar_mols))
        print("✅ Molecule data downloaded")
    else:
        print("✅ Molecule data already exists")
        
    if not mols.exists():
        print(f"Extracting molecule data to {mols}")
        with tarfile.open(str(tar_mols), "r") as tar:
            tar.extractall(cache)
        print("✅ Molecule data extracted")
    else:
        print("✅ Molecule data already extracted")

    # Download model
    model = cache / "boltz2_conf.ckpt"
    if not model.exists():
        download_with_fallback(BOLTZ2_URL_WITH_FALLBACK, model, "Boltz2 model weights")
    else:
        print("✅ Boltz2 model weights already exist")

    # Download affinity model
    affinity_model = cache / "boltz2_aff.ckpt"
    if not affinity_model.exists():
        download_with_fallback(BOLTZ2_AFFINITY_URL_WITH_FALLBACK, affinity_model, "Boltz2 affinity weights")
    else:
        print("✅ Boltz2 affinity weights already exist")


def main():
    """Download all Boltz checkpoints."""
    # Use the default cache directory
    cache_dir = Path.home() / ".boltz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using cache directory: {cache_dir}")
    
    try:
        download_boltz1_standalone(cache_dir)
        print()
        download_boltz2_standalone(cache_dir)
        
        print("\n=== Download Summary ===")
        files = list(cache_dir.glob("*"))
        total_size = 0
        for file in sorted(files):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                total_size += size_mb
                print(f"  {file.name}: {size_mb:.1f} MB")
            else:
                print(f"  {file.name}/: (directory)")
        
        print(f"\nTotal size: {total_size:.1f} MB")
        print(f"All checkpoints downloaded to: {cache_dir}")
        print("\n✅ All downloads completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
