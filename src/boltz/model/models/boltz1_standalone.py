import gc
import random
from typing import Any, Optional

import torch
import torch._dynamo
from torch import Tensor, nn

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.data.feature.symmetry import (
    minimum_lddt_symmetry_coords,
    minimum_symmetry_coords,
)
from boltz.model.loss.confidence import confidence_loss
from boltz.model.loss.distogram import distogram_loss
from boltz.model.loss.validation import (
    compute_pae_mae,
    compute_pde_mae,
    compute_plddt_mae,
    factored_lddt_loss,
    factored_token_lddt_dist_loss,
    weighted_minimum_rmsd,
)
from boltz.model.modules.confidence import ConfidenceModule
from boltz.model.modules.diffusion import AtomDiffusion
from boltz.model.modules.encoders import RelativePositionEncoder
from boltz.model.modules.trunk import (
    DistogramModule,
    InputEmbedder,
    MSAModule,
    PairformerModule,
)
from boltz.model.modules.utils import ExponentialMovingAverage


class Boltz1Standalone(nn.Module):
    """Boltz1 model without PyTorch Lightning dependency."""

    def __init__(  # noqa: PLR0915, C901, PLR0912
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        training_args: dict[str, Any],
        validation_args: dict[str, Any],
        embedder_args: dict[str, Any],
        msa_args: dict[str, Any],
        pairformer_args: dict[str, Any],
        score_model_args: dict[str, Any],
        diffusion_process_args: dict[str, Any],
        diffusion_loss_args: dict[str, Any],
        confidence_model_args: dict[str, Any],
        atom_feature_dim: int = 128,
        confidence_prediction: bool = False,
        confidence_imitate_trunk: bool = False,
        alpha_pae: float = 0.0,
        structure_prediction_training: bool = True,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        compile_pairformer: bool = False,
        compile_structure: bool = False,
        compile_confidence: bool = False,
        nucleotide_rmsd_weight: float = 5.0,
        ligand_rmsd_weight: float = 10.0,
        no_msa: bool = False,
        no_atom_encoder: bool = False,
        ema: bool = False,
        ema_decay: float = 0.999,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        predict_args: Optional[dict[str, Any]] = None,
        steering_args: Optional[dict[str, Any]] = None,
        use_kernels: bool = False,
    ) -> None:
        super().__init__()

        # Store hyperparameters
        self.atom_s = atom_s
        self.atom_z = atom_z
        self.token_s = token_s
        self.token_z = token_z
        self.num_bins = num_bins
        self.training_args = training_args
        self.validation_args = validation_args
        self.embedder_args = embedder_args
        self.msa_args = msa_args
        self.pairformer_args = pairformer_args
        self.score_model_args = score_model_args
        self.diffusion_process_args = diffusion_process_args
        self.diffusion_loss_args = diffusion_loss_args
        self.confidence_model_args = confidence_model_args
        self.atom_feature_dim = atom_feature_dim
        self.confidence_prediction = confidence_prediction
        self.confidence_imitate_trunk = confidence_imitate_trunk
        self.alpha_pae = alpha_pae
        self.structure_prediction_training = structure_prediction_training
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.compile_pairformer = compile_pairformer
        self.compile_structure = compile_structure
        self.compile_confidence = compile_confidence
        self.nucleotide_rmsd_weight = nucleotide_rmsd_weight
        self.ligand_rmsd_weight = ligand_rmsd_weight
        self.no_msa = no_msa
        self.no_atom_encoder = no_atom_encoder
        self.use_ema = ema
        self.ema_decay = ema_decay
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.predict_args = predict_args or {}
        self.steering_args = steering_args or {}
        self.use_kernels = use_kernels

        # Initialize modules
        self._init_modules()

    def _init_modules(self):
        """Initialize all model modules."""
        # Input embedder
        self.input_embedder = InputEmbedder(**self.embedder_args)

        # Initial embeddings
        self.s_init = nn.Linear(self.input_embedder.s_inputs, self.token_s)
        self.z_init_1 = nn.Linear(self.input_embedder.s_inputs, self.token_z)
        self.z_init_2 = nn.Linear(self.input_embedder.s_inputs, self.token_z)

        # MSA module
        if not self.no_msa:
            self.msa_module = MSAModule(
                token_s=self.token_s,
                token_z=self.token_z,
                **self.msa_args,
            )

        # Pairformer module
        self.pairformer_module = PairformerModule(
            token_s=self.token_s,
            token_z=self.token_z,
            **self.pairformer_args,
        )

        # Distogram module
        self.distogram_module = DistogramModule(
            token_z=self.token_z,
            num_bins=self.num_bins,
        )

        # Structure module (diffusion)
        self.structure_module = AtomDiffusion(
            atom_s=self.atom_s,
            atom_z=self.atom_z,
            token_s=self.token_s,
            token_z=self.token_z,
            atom_feature_dim=self.atom_feature_dim,
            atoms_per_window_queries=self.atoms_per_window_queries,
            atoms_per_window_keys=self.atoms_per_window_keys,
            no_atom_encoder=self.no_atom_encoder,
            use_kernels=self.use_kernels,
            **self.score_model_args,
        )

        # Confidence module
        if self.confidence_prediction:
            self.confidence_module = ConfidenceModule(
                token_s=self.token_s,
                token_z=self.token_z,
                **self.confidence_model_args,
            )

        # Relative position encoder
        self.relative_position_encoder = RelativePositionEncoder()

        # Initialize weights
        self.apply(init.init_weights)

        # EMA
        self.ema = None
        if self.use_ema:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )

        # Compile modules if requested
        if self.compile_pairformer:
            self.pairformer_module = torch.compile(self.pairformer_module)
        if self.compile_structure:
            self.structure_module = torch.compile(self.structure_module)
        if self.compile_confidence and self.confidence_prediction:
            self.confidence_module = torch.compile(self.confidence_module)

    @classmethod
    def from_lightning_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: str = "cpu",
        strict: bool = True,
        **kwargs
    ):
        """Load model from a PyTorch Lightning checkpoint.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to the Lightning checkpoint file.
        map_location : str
            Device to load the model on.
        strict : bool
            Whether to strictly enforce that the keys in state_dict match.
        **kwargs
            Additional arguments to override model configuration.
            
        Returns
        -------
        Boltz1Standalone
            The loaded model.
        """
        # Load the checkpoint with PyTorch 2.6+ compatibility
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        except Exception:
            try:
                import omegaconf
                torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
                checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}") from e
        
        # Extract hyperparameters
        hparams = checkpoint.get("hyper_parameters", {})
        
        # Override with any provided kwargs
        hparams.update(kwargs)
        
        # Create the model
        model = cls(**hparams)
        
        # Load the state dict
        state_dict = checkpoint["state_dict"]
        
        # Remove any Lightning-specific keys if present
        filtered_state_dict = {}
        for key, value in state_dict.items():
            # Remove any keys that start with Lightning-specific prefixes
            if not any(key.startswith(prefix) for prefix in ["_metrics", "trainer"]):
                filtered_state_dict[key] = value
        
        model.load_state_dict(filtered_state_dict, strict=strict)
        
        # Load EMA if present
        if model.use_ema and "ema" in checkpoint:
            if model.ema.compatible(checkpoint["ema"]["shadow_params"]):
                model.ema.load_state_dict(checkpoint["ema"], device=torch.device(map_location))
            else:
                print("Warning: EMA state not loaded due to incompatible model parameters.")
        
        return model

    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: int = 1,
        diffusion_samples: int = 1,
        max_parallel_samples: Optional[int] = None,
        run_confidence_sequentially: bool = False,
    ) -> dict[str, Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        feats : dict[str, Tensor]
            Input features.
        recycling_steps : int
            Number of recycling steps.
        num_sampling_steps : Optional[int]
            Number of sampling steps for diffusion.
        multiplicity_diffusion_train : int
            Multiplicity for diffusion training.
        diffusion_samples : int
            Number of diffusion samples.
        max_parallel_samples : Optional[int]
            Maximum parallel samples.
        run_confidence_sequentially : bool
            Whether to run confidence sequentially.

        Returns
        -------
        dict[str, Tensor]
            Model outputs.
        """
        dict_out = {}

        # Compute input embeddings
        with torch.set_grad_enabled(
            self.training and self.structure_prediction_training
        ):
            s_inputs = self.input_embedder(feats)

            # Initialize the sequence and pairwise embeddings
            s_init = self.s_init(s_inputs)
            z_init = (
                self.z_init_1(s_inputs)[:, :, None]
                + self.z_init_2(s_inputs)[:, None, :]
            )

            # Add relative position encoding
            z_init = z_init + self.relative_position_encoder(feats)

            # MSA processing
            if not self.no_msa and "msa" in feats:
                s_msa, z_msa = self.msa_module(feats, s_init, z_init)
                s_trunk = s_msa
                z_trunk = z_msa
            else:
                s_trunk = s_init
                z_trunk = z_init

            # Recycling loop
            for _ in range(recycling_steps + 1):
                # Pairformer
                s_trunk, z_trunk = self.pairformer_module(
                    s_trunk, z_trunk, feats
                )

            # Distogram prediction
            distogram_logits = self.distogram_module(z_trunk)
            dict_out["distogram_logits"] = distogram_logits

            # Structure prediction
            structure_out = self.structure_module(
                s_trunk,
                z_trunk,
                feats,
                num_sampling_steps=num_sampling_steps,
                multiplicity_diffusion_train=multiplicity_diffusion_train,
                diffusion_samples=diffusion_samples,
                max_parallel_samples=max_parallel_samples,
            )
            dict_out.update(structure_out)

            # Confidence prediction
            if self.confidence_prediction:
                confidence_out = self.confidence_module(
                    s_trunk,
                    z_trunk,
                    feats,
                    structure_out.get("coords", None),
                    run_sequentially=run_confidence_sequentially,
                )
                dict_out.update(confidence_out)

        return dict_out

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int = 0) -> dict[str, Tensor]:
        """Prediction step for inference.

        Parameters
        ----------
        batch : dict[str, Tensor]
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        dict[str, Tensor]
            Predictions.
        """
        try:
            # Get prediction arguments
            predict_args = self.predict_args

            # Run forward pass
            out = self(
                batch,
                recycling_steps=predict_args.get("recycling_steps", 3),
                num_sampling_steps=predict_args.get("sampling_steps", 200),
                diffusion_samples=predict_args.get("diffusion_samples", 1),
                max_parallel_samples=predict_args.get("max_parallel_samples", None),
                run_confidence_sequentially=predict_args.get("run_confidence_sequentially", False),
            )

            return out

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            else:
                raise
