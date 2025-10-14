"""Standalone inference functions for Boltz models without PyTorch Lightning."""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Literal
from dataclasses import asdict

from boltz.model.utils.checkpoint_loader import load_model_from_checkpoint
from boltz.inference.data_pipeline import create_inference_dataloader
from boltz.data.types import Manifest, Record
from boltz.data.write.writer import BoltzWriter, BoltzAffinityWriter


class BoltzInference:
    """Standalone inference class for Boltz models."""
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        model_type: str = "auto",
        device: str = "auto",
        **model_kwargs
    ):
        """Initialize the inference class.
        
        Parameters
        ----------
        checkpoint_path : Union[str, Path]
            Path to the model checkpoint.
        model_type : str
            Model type ("boltz1", "boltz2", or "auto").
        device : str
            Device to run inference on ("auto", "cpu", "cuda", etc.).
        **model_kwargs
            Additional arguments for model loading.
        """
        self.checkpoint_path = Path(checkpoint_path)
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load model
        self.model = load_model_from_checkpoint(
            self.checkpoint_path,
            model_type=model_type,
            map_location=self.device,
            **model_kwargs
        )
        self.model.eval()
        self.model.to(self.device)
        
        # Determine model version
        self.model_version = "boltz2" if "Boltz2" in self.model.__class__.__name__ else "boltz1"
        
    def predict_single(
        self,
        record: Record,
        target_dir: Path,
        msa_dir: Path,
        mol_dir: Optional[Path] = None,
        constraints_dir: Optional[Path] = None,
        template_dir: Optional[Path] = None,
        extra_mols_dir: Optional[Path] = None,
        override_method: Optional[str] = None,
        affinity: bool = False,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        max_parallel_samples: Optional[int] = None,
        run_confidence_sequentially: bool = False,
    ) -> Dict[str, Any]:
        """Run inference on a single record.
        
        Parameters
        ----------
        record : Record
            The record to process.
        target_dir : Path
            Path to target directory.
        msa_dir : Path
            Path to MSA directory.
        mol_dir : Optional[Path]
            Path to molecule directory.
        constraints_dir : Optional[Path]
            Path to constraints directory.
        template_dir : Optional[Path]
            Path to template directory.
        extra_mols_dir : Optional[Path]
            Path to extra molecules directory.
        override_method : Optional[str]
            Method override.
        affinity : bool
            Whether this is for affinity prediction.
        recycling_steps : int
            Number of recycling steps.
        sampling_steps : int
            Number of sampling steps.
        diffusion_samples : int
            Number of diffusion samples.
        max_parallel_samples : Optional[int]
            Maximum parallel samples.
        run_confidence_sequentially : bool
            Whether to run confidence sequentially.
            
        Returns
        -------
        Dict[str, Any]
            Prediction results.
        """
        # Create manifest with single record
        manifest = Manifest([record])
        
        # Create dataloader
        dataloader = create_inference_dataloader(
            manifest=manifest,
            target_dir=target_dir,
            msa_dir=msa_dir,
            model_version=self.model_version,
            mol_dir=mol_dir,
            constraints_dir=constraints_dir,
            template_dir=template_dir,
            extra_mols_dir=extra_mols_dir,
            override_method=override_method,
            affinity=affinity,
            num_workers=0,
            batch_size=1,
        )
        
        # Get the batch
        batch = next(iter(dataloader))
        
        # Move batch to device
        batch = self._move_batch_to_device(batch)
        
        # Set prediction arguments
        self.model.predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            "run_confidence_sequentially": run_confidence_sequentially,
        }
        
        # Run inference
        with torch.no_grad():
            predictions = self.model.predict_step(batch)
            
        return predictions
    
    def predict_batch(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        mol_dir: Optional[Path] = None,
        constraints_dir: Optional[Path] = None,
        template_dir: Optional[Path] = None,
        extra_mols_dir: Optional[Path] = None,
        override_method: Optional[str] = None,
        affinity: bool = False,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        max_parallel_samples: Optional[int] = None,
        run_confidence_sequentially: bool = False,
        num_workers: int = 0,
    ) -> List[Dict[str, Any]]:
        """Run inference on a batch of records.
        
        Parameters
        ----------
        manifest : Manifest
            The manifest containing records to process.
        target_dir : Path
            Path to target directory.
        msa_dir : Path
            Path to MSA directory.
        mol_dir : Optional[Path]
            Path to molecule directory.
        constraints_dir : Optional[Path]
            Path to constraints directory.
        template_dir : Optional[Path]
            Path to template directory.
        extra_mols_dir : Optional[Path]
            Path to extra molecules directory.
        override_method : Optional[str]
            Method override.
        affinity : bool
            Whether this is for affinity prediction.
        recycling_steps : int
            Number of recycling steps.
        sampling_steps : int
            Number of sampling steps.
        diffusion_samples : int
            Number of diffusion samples.
        max_parallel_samples : Optional[int]
            Maximum parallel samples.
        run_confidence_sequentially : bool
            Whether to run confidence sequentially.
        num_workers : int
            Number of worker processes.
            
        Returns
        -------
        List[Dict[str, Any]]
            List of prediction results.
        """
        # Create dataloader
        dataloader = create_inference_dataloader(
            manifest=manifest,
            target_dir=target_dir,
            msa_dir=msa_dir,
            model_version=self.model_version,
            mol_dir=mol_dir,
            constraints_dir=constraints_dir,
            template_dir=template_dir,
            extra_mols_dir=extra_mols_dir,
            override_method=override_method,
            affinity=affinity,
            num_workers=num_workers,
            batch_size=1,  # Keep batch size 1 for inference
        )
        
        # Set prediction arguments
        self.model.predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            "run_confidence_sequentially": run_confidence_sequentially,
        }
        
        # Run inference on all batches
        results = []
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Run inference
                predictions = self.model.predict_step(batch)
                results.append(predictions)
                
        return results
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to the appropriate device."""
        # Keys that should not be moved to device
        skip_keys = [
            "all_coords",
            "all_resolved_mask", 
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
            "affinity_mw",
            "exception"
        ]
        
        for key, value in batch.items():
            if key not in skip_keys and isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
                
        return batch


def predict_structure(
    input_data: Union[str, Path, Record, Manifest],
    checkpoint_path: Union[str, Path],
    target_dir: Union[str, Path],
    msa_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    model_type: str = "auto",
    device: str = "auto",
    mol_dir: Optional[Union[str, Path]] = None,
    constraints_dir: Optional[Union[str, Path]] = None,
    template_dir: Optional[Union[str, Path]] = None,
    extra_mols_dir: Optional[Union[str, Path]] = None,
    override_method: Optional[str] = None,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    max_parallel_samples: Optional[int] = None,
    run_confidence_sequentially: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    write_embeddings: bool = False,
    num_workers: int = 0,
    **model_kwargs
) -> List[Dict[str, Any]]:
    """High-level function to predict protein structures.
    
    Parameters
    ----------
    input_data : Union[str, Path, Record, Manifest]
        Input data - can be a path to manifest file, Record, or Manifest object.
    checkpoint_path : Union[str, Path]
        Path to the model checkpoint.
    target_dir : Union[str, Path]
        Path to target directory containing structure files.
    msa_dir : Union[str, Path]
        Path to MSA directory.
    output_dir : Optional[Union[str, Path]]
        Output directory for saving results. If None, results are only returned.
    model_type : str
        Model type ("boltz1", "boltz2", or "auto").
    device : str
        Device to run inference on.
    mol_dir : Optional[Union[str, Path]]
        Path to molecule directory.
    constraints_dir : Optional[Union[str, Path]]
        Path to constraints directory.
    template_dir : Optional[Union[str, Path]]
        Path to template directory.
    extra_mols_dir : Optional[Union[str, Path]]
        Path to extra molecules directory.
    override_method : Optional[str]
        Method override.
    recycling_steps : int
        Number of recycling steps.
    sampling_steps : int
        Number of sampling steps.
    diffusion_samples : int
        Number of diffusion samples.
    max_parallel_samples : Optional[int]
        Maximum parallel samples.
    run_confidence_sequentially : bool
        Whether to run confidence sequentially.
    output_format : Literal["pdb", "mmcif"]
        Output file format.
    write_embeddings : bool
        Whether to write embeddings.
    num_workers : int
        Number of worker processes.
    **model_kwargs
        Additional model arguments.
        
    Returns
    -------
    List[Dict[str, Any]]
        Prediction results.
    """
    # Convert paths
    target_dir = Path(target_dir)
    msa_dir = Path(msa_dir)
    mol_dir = Path(mol_dir) if mol_dir else None
    constraints_dir = Path(constraints_dir) if constraints_dir else None
    template_dir = Path(template_dir) if template_dir else None
    extra_mols_dir = Path(extra_mols_dir) if extra_mols_dir else None
    
    # Handle input data
    if isinstance(input_data, (str, Path)):
        manifest = Manifest.load(Path(input_data))
    elif isinstance(input_data, Record):
        manifest = Manifest([input_data])
    elif isinstance(input_data, Manifest):
        manifest = input_data
    else:
        raise ValueError("input_data must be a path, Record, or Manifest")
    
    # Initialize inference
    inference = BoltzInference(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device,
        **model_kwargs
    )
    
    # Run inference
    results = inference.predict_batch(
        manifest=manifest,
        target_dir=target_dir,
        msa_dir=msa_dir,
        mol_dir=mol_dir,
        constraints_dir=constraints_dir,
        template_dir=template_dir,
        extra_mols_dir=extra_mols_dir,
        override_method=override_method,
        recycling_steps=recycling_steps,
        sampling_steps=sampling_steps,
        diffusion_samples=diffusion_samples,
        max_parallel_samples=max_parallel_samples,
        run_confidence_sequentially=run_confidence_sequentially,
        num_workers=num_workers,
    )
    
    # Save results if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the existing writer to save results
        writer = BoltzWriter(
            data_dir=str(target_dir),
            output_dir=str(output_dir),
            output_format=output_format,
            boltz2=inference.model_version == "boltz2",
            write_embeddings=write_embeddings,
        )
        
        # Save each result
        for i, (result, record) in enumerate(zip(results, manifest.records)):
            if not result.get("exception", False):
                # Create a mock batch for the writer
                batch = {"record": [record]}
                writer.write_on_batch_end(
                    trainer=None,
                    pl_module=None,
                    prediction=result,
                    batch_indices=[i],
                    batch=batch,
                    batch_idx=i,
                    dataloader_idx=0,
                )
    
    return results
