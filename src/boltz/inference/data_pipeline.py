"""Data processing pipeline for standalone inference without PyTorch Lightning."""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from torch.utils.data import Dataset, DataLoader

from boltz.data import const
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import (
    MSA,
    Connection,
    Input,
    Manifest,
    Record,
    ResidueConstraints,
    Structure,
    StructureV2,
)
from boltz.data.mol import load_canonicals, load_molecules
from boltz.data.crop.affinity import AffinityCropper
from boltz.data.pad import pad_to_max


def load_input_data(
    record: Record,
    target_dir: Path,
    msa_dir: Path,
    constraints_dir: Optional[Path] = None,
    model_version: str = "boltz1"
) -> Input:
    """Load input data for a single record.
    
    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        Path to the target directory containing structure files.
    msa_dir : Path
        Path to the MSA directory.
    constraints_dir : Optional[Path]
        Path to constraints directory.
    model_version : str
        Model version ("boltz1" or "boltz2").
        
    Returns
    -------
    Input
        The loaded input data.
    """
    # Load the structure
    structure_file = target_dir / f"{record.id}.npz"
    structure_data = np.load(structure_file)
    
    if model_version == "boltz2":
        structure = StructureV2(
            atoms=structure_data["atoms"],
            bonds=structure_data["bonds"],
            residues=structure_data["residues"],
            chains=structure_data["chains"],
            connections=structure_data["connections"].astype(Connection),
            interfaces=structure_data["interfaces"],
            mask=structure_data["mask"],
        )
    else:
        structure = Structure(
            atoms=structure_data["atoms"],
            bonds=structure_data["bonds"],
            residues=structure_data["residues"],
            chains=structure_data["chains"],
            connections=structure_data["connections"].astype(Connection),
            interfaces=structure_data["interfaces"],
            mask=structure_data["mask"],
        )

    # Load MSAs
    msas = {}
    for chain in record.chains:
        msa_id = chain.msa_id
        if msa_id != -1:
            msa_file = msa_dir / f"{msa_id}.npz"
            if msa_file.exists():
                msa_data = np.load(msa_file)
                msas[chain.chain_id] = MSA(**msa_data)

    # Load constraints if available
    residue_constraints = None
    if constraints_dir is not None:
        constraints_file = constraints_dir / f"{record.id}.npz"
        if constraints_file.exists():
            residue_constraints = ResidueConstraints.load(constraints_file)

    return Input(structure, msas, record, residue_constraints)


class StandaloneInferenceDataset(Dataset):
    """Dataset for standalone inference without Lightning."""
    
    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        model_version: str = "boltz2",
        mol_dir: Optional[Path] = None,
        constraints_dir: Optional[Path] = None,
        template_dir: Optional[Path] = None,
        extra_mols_dir: Optional[Path] = None,
        override_method: Optional[str] = None,
        affinity: bool = False,
    ):
        """Initialize the dataset.
        
        Parameters
        ----------
        manifest : Manifest
            The manifest containing records to process.
        target_dir : Path
            Path to target directory.
        msa_dir : Path
            Path to MSA directory.
        model_version : str
            Model version ("boltz1" or "boltz2").
        mol_dir : Optional[Path]
            Path to molecule directory (for Boltz2).
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
        """
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.model_version = model_version
        self.mol_dir = mol_dir
        self.constraints_dir = constraints_dir
        self.template_dir = template_dir
        self.extra_mols_dir = extra_mols_dir
        self.override_method = override_method
        self.affinity = affinity
        
        # Initialize tokenizer and featurizer based on model version
        if model_version == "boltz2":
            self.tokenizer = Boltz2Tokenizer()
            self.featurizer = Boltz2Featurizer()
            if mol_dir:
                self.canonicals = load_canonicals(mol_dir)
            else:
                self.canonicals = None
        else:
            self.tokenizer = BoltzTokenizer()
            self.featurizer = BoltzFeaturizer()
            self.canonicals = None
            
        # Initialize cropper for affinity
        if affinity:
            self.cropper = AffinityCropper()
    
    def __len__(self) -> int:
        return len(self.manifest.records)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an item from the dataset."""
        record = self.manifest.records[idx]
        
        try:
            # Load input data
            input_data = load_input_data(
                record, self.target_dir, self.msa_dir, 
                self.constraints_dir, self.model_version
            )
            
            # Tokenize
            tokenized = self.tokenizer.tokenize(input_data)
            
            # Handle inference options for pocket constraints
            options = record.inference_options
            if options is None or len(options.pocket_constraints) == 0:
                binder, pocket = None, None
            else:
                binder, pocket = (
                    options.pocket_constraints[0][0],
                    options.pocket_constraints[0][1],
                )
            
            # Featurize
            if self.model_version == "boltz2":
                # Load molecules if needed
                molecules = None
                if self.mol_dir and self.canonicals:
                    molecules = load_molecules(
                        self.mol_dir, self.extra_mols_dir, self.canonicals
                    )
                
                features = self.featurizer.process(
                    tokenized,
                    training=False,
                    max_atoms=None,
                    max_tokens=None,
                    max_seqs=const.max_msa_seqs,
                    pad_to_max_seqs=False,
                    symmetries={},
                    compute_symmetries=False,
                    inference_binder=binder,
                    inference_pocket=pocket,
                    compute_constraint_features=True,
                    molecules=molecules,
                    template_dir=self.template_dir,
                    override_method=self.override_method,
                )
                
                # Apply affinity cropping if needed
                if self.affinity:
                    features = self.cropper.crop(features)
                    
            else:
                features = self.featurizer.process(
                    tokenized,
                    training=False,
                    max_atoms=None,
                    max_tokens=None,
                    max_seqs=const.max_msa_seqs,
                    pad_to_max_seqs=False,
                    symmetries={},
                    compute_symmetries=False,
                    inference_binder=binder,
                    inference_pocket=pocket,
                    compute_constraint_features=True,
                )
            
            features["record"] = record
            return features
            
        except Exception as e:
            print(f"Failed to process {record.id}: {e}")
            # Return a dummy item to avoid breaking the batch
            return {"record": record, "exception": True}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for batching."""
    # For inference, we typically use batch size 1
    # So we just return the first (and only) item
    if len(batch) == 1:
        return batch[0]
    
    # If we have multiple items, we need to implement proper batching
    # For now, just return the first item
    return batch[0]


def create_inference_dataloader(
    manifest: Manifest,
    target_dir: Path,
    msa_dir: Path,
    model_version: str = "boltz2",
    mol_dir: Optional[Path] = None,
    constraints_dir: Optional[Path] = None,
    template_dir: Optional[Path] = None,
    extra_mols_dir: Optional[Path] = None,
    override_method: Optional[str] = None,
    affinity: bool = False,
    num_workers: int = 0,
    batch_size: int = 1,
) -> DataLoader:
    """Create a DataLoader for inference.
    
    Parameters
    ----------
    manifest : Manifest
        The manifest containing records to process.
    target_dir : Path
        Path to target directory.
    msa_dir : Path
        Path to MSA directory.
    model_version : str
        Model version ("boltz1" or "boltz2").
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
    num_workers : int
        Number of worker processes.
    batch_size : int
        Batch size (typically 1 for inference).
        
    Returns
    -------
    DataLoader
        The configured DataLoader.
    """
    dataset = StandaloneInferenceDataset(
        manifest=manifest,
        target_dir=target_dir,
        msa_dir=msa_dir,
        model_version=model_version,
        mol_dir=mol_dir,
        constraints_dir=constraints_dir,
        template_dir=template_dir,
        extra_mols_dir=extra_mols_dir,
        override_method=override_method,
        affinity=affinity,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )
