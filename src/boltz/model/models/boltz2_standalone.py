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
from boltz.model.loss.distogramv2 import distogram_loss
from boltz.model.modules.affinity import AffinityModule
from boltz.model.modules.confidencev2 import ConfidenceModule
from boltz.model.modules.diffusion_conditioning import DiffusionConditioning
from boltz.model.modules.diffusionv2 import AtomDiffusion
from boltz.model.modules.encodersv2 import RelativePositionEncoder
from boltz.model.layers.pairformer import PairformerModule
from boltz.model.modules.trunkv2 import (
    BFactorModule,
    ContactConditioning,
    DistogramModule,
    InputEmbedder,
    MSAModule,
    TemplateModule,
    TemplateV2Module,
)
from boltz.model.optim.ema import EMA


class Boltz2Standalone(nn.Module):
    """Boltz2 model without PyTorch Lightning dependency."""

    def __init__(
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
        confidence_model_args: Optional[dict[str, Any]] = None,
        affinity_model_args: Optional[dict[str, Any]] = None,
        affinity_model_args1: Optional[dict[str, Any]] = None,
        affinity_model_args2: Optional[dict[str, Any]] = None,
        validators: Any = None,
        num_val_datasets: int = 1,
        atom_feature_dim: int = 128,
        template_args: Optional[dict] = None,
        confidence_prediction: bool = True,
        affinity_prediction: bool = False,
        structure_prediction_training: bool = True,
        confidence_imitate_trunk: bool = False,
        alpha_pae: float = 0.0,
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
        steering_args: Optional[dict] = None,
        use_templates: bool = False,
        compile_templates: bool = False,
        predict_bfactor: bool = False,
        log_loss_every_steps: int = 50,
        checkpoint_diffusion_conditioning: bool = False,
        use_templates_v2: bool = False,
        use_kernels: bool = False,
        affinity_mw_correction: Optional[bool] = False,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
        fix_sym_check: bool = False,
        cyclic_pos_enc: bool = False,
        bond_type_feature: bool = False,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        **kwargs  # Accept any additional arguments
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

        # Store additional parameters needed for InputEmbedder
        self.atom_feature_dim = atom_feature_dim
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys

        # Store other parameters that might be needed
        self.confidence_prediction = confidence_prediction
        self.affinity_prediction = affinity_prediction
        self.structure_prediction_training = structure_prediction_training
        self.use_templates = use_templates
        self.use_templates_v2 = use_templates_v2
        self.predict_bfactor = predict_bfactor
        self.affinity_mw_correction = affinity_mw_correction
        self.use_no_atom_char = use_no_atom_char
        self.use_atom_backbone_feat = use_atom_backbone_feat
        self.use_residue_feats_atoms = use_residue_feats_atoms
        self.fix_sym_check = fix_sym_check
        self.cyclic_pos_enc = cyclic_pos_enc
        self.bond_type_feature = bond_type_feature
        self.conditioning_cutoff_min = conditioning_cutoff_min
        self.conditioning_cutoff_max = conditioning_cutoff_max
        self.msa_args = msa_args
        self.pairformer_args = pairformer_args
        self.score_model_args = score_model_args
        self.diffusion_process_args = diffusion_process_args
        self.diffusion_loss_args = diffusion_loss_args
        self.confidence_model_args = confidence_model_args or {}
        self.affinity_model_args = affinity_model_args or {}
        self.affinity_model_args1 = affinity_model_args1 or {}
        self.affinity_model_args2 = affinity_model_args2 or {}
        self.validators = validators
        self.num_val_datasets = num_val_datasets
        self.atom_feature_dim = atom_feature_dim
        self.template_args = template_args or {}
        self.confidence_prediction = confidence_prediction
        self.affinity_prediction = affinity_prediction
        self.structure_prediction_training = structure_prediction_training
        self.confidence_imitate_trunk = confidence_imitate_trunk
        self.alpha_pae = alpha_pae
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
        self.use_templates = use_templates
        self.compile_templates = compile_templates
        self.predict_bfactor = predict_bfactor
        self.log_loss_every_steps = log_loss_every_steps
        self.checkpoint_diffusion_conditioning = checkpoint_diffusion_conditioning
        self.use_templates_v2 = use_templates_v2
        self.use_kernels = use_kernels
        self.affinity_mw_correction = affinity_mw_correction

        # Initialize modules
        self._init_modules()

    def _init_modules(self):
        """Initialize all model modules."""
        # Input embedder - need to pass required parameters
        full_embedder_args = {
            "atom_s": self.atom_s,
            "atom_z": self.atom_z,
            "token_s": self.token_s,
            "token_z": self.token_z,
            "atoms_per_window_queries": self.atoms_per_window_queries,
            "atoms_per_window_keys": self.atoms_per_window_keys,
            "atom_feature_dim": self.atom_feature_dim,
            "use_no_atom_char": self.use_no_atom_char,
            "use_atom_backbone_feat": self.use_atom_backbone_feat,
            "use_residue_feats_atoms": self.use_residue_feats_atoms,
            **self.embedder_args,
        }
        self.input_embedder = InputEmbedder(**full_embedder_args)

        # Initial embeddings (Boltz2 uses token_s directly, not input_embedder.s_inputs)
        self.s_init = nn.Linear(self.token_s, self.token_s, bias=False)
        self.z_init_1 = nn.Linear(self.token_s, self.token_z, bias=False)
        self.z_init_2 = nn.Linear(self.token_s, self.token_z, bias=False)

        # Relative position encoder
        self.rel_pos = RelativePositionEncoder(
            self.token_z,
            fix_sym_check=self.fix_sym_check,
            cyclic_pos_enc=self.cyclic_pos_enc
        )

        # Token bonds
        self.token_bonds = nn.Linear(1, self.token_z, bias=False)
        if self.bond_type_feature:
            from boltz.data import const
            self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, self.token_z)

        # Contact conditioning
        self.contact_conditioning = ContactConditioning(
            token_z=self.token_z,
            cutoff_min=self.conditioning_cutoff_min,
            cutoff_max=self.conditioning_cutoff_max,
        )

        # Normalization layers
        self.s_norm = nn.LayerNorm(self.token_s)
        self.z_norm = nn.LayerNorm(self.token_z)

        # Recycling projections
        self.s_recycle = nn.Linear(self.token_s, self.token_s, bias=False)
        self.z_recycle = nn.Linear(self.token_z, self.token_z, bias=False)

        # Initialize recycling weights
        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

        # MSA module
        self.msa_module = MSAModule(
            token_z=self.token_z,
            token_s=self.token_s,
            **self.msa_args
        )

        # Pairformer module
        self.pairformer_module = PairformerModule(
            self.token_s,
            self.token_z,
            **self.pairformer_args
        )

        # Diffusion conditioning
        self.diffusion_conditioning = DiffusionConditioning(
            token_s=self.token_s,
            token_z=self.token_z,
            atom_s=self.atom_s,
            atom_z=self.atom_z,
            atoms_per_window_queries=self.atoms_per_window_queries,
            atoms_per_window_keys=self.atoms_per_window_keys,
            atom_encoder_depth=self.score_model_args["atom_encoder_depth"],
            atom_encoder_heads=self.score_model_args["atom_encoder_heads"],
        )

        # Diffusion module
        self.diffusion_module = AtomDiffusion(**self.diffusion_process_args)

        # Distogram module
        self.distogram_module = DistogramModule(**self.distogram_args)

        # Confidence module
        if self.confidence_prediction:
            self.confidence_module = ConfidenceModule(**self.confidence_model_args)

        # Affinity module
        if self.affinity_prediction:
            self.affinity_module = AffinityModule(**self.affinity_args)

        # B-factor module
        if self.predict_bfactor:
            self.bfactor_module = BFactorModule(**self.bfactor_args)

        # Template modules
        if self.use_templates:
            if self.use_templates_v2:
                self.template_module = TemplateV2Module(**self.template_args)
            else:
                self.template_module = TemplateModule(**self.template_args)

        # MSA module
        if not self.no_msa:
            self.msa_module = MSAModule(
                token_s=self.token_s,
                token_z=self.token_z,
                **self.msa_args,
            )

        # Template module
        if self.use_templates:
            if self.use_templates_v2:
                self.template_module = TemplateV2Module(
                    token_s=self.token_s,
                    token_z=self.token_z,
                    **self.template_args,
                )
            else:
                self.template_module = TemplateModule(
                    token_s=self.token_s,
                    token_z=self.token_z,
                    **self.template_args,
                )

        # Pairformer module (trunk)
        from boltz.model.modules.trunkv2 import PairformerModule
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

        # Contact conditioning
        self.contact_conditioning = ContactConditioning(
            token_z=self.token_z,
        )

        # Diffusion conditioning
        if self.checkpoint_diffusion_conditioning:
            self.diffusion_conditioning = torch.utils.checkpoint.checkpoint_wrapper(
                DiffusionConditioning(
                    token_s=self.token_s,
                    token_z=self.token_z,
                )
            )
        else:
            self.diffusion_conditioning = DiffusionConditioning(
                token_s=self.token_s,
                token_z=self.token_z,
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

        # Affinity module
        if self.affinity_prediction:
            self.affinity_module = AffinityModule(
                token_s=self.token_s,
                token_z=self.token_z,
                **self.affinity_model_args,
            )

        # B-factor module
        if self.predict_bfactor:
            self.bfactor_module = BFactorModule(
                token_s=self.token_s,
            )

        # Relative position encoder
        self.relative_position_encoder = RelativePositionEncoder()

        # Initialize weights
        self.apply(init.init_weights)

        # EMA
        self.ema = None
        if self.use_ema:
            self.ema = EMA(
                parameters=self.parameters(), decay=self.ema_decay
            )

        # Compile modules if requested
        if self.compile_pairformer:
            self.pairformer_module = torch.compile(self.pairformer_module)
        if self.compile_structure:
            self.structure_module = torch.compile(self.structure_module)
        if self.compile_confidence and self.confidence_prediction:
            self.confidence_module = torch.compile(self.confidence_module)
        if self.compile_templates and self.use_templates:
            self.template_module = torch.compile(self.template_module)

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
        Boltz2Standalone
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
            if hasattr(model.ema, 'compatible') and model.ema.compatible(checkpoint["ema"]["shadow_params"]):
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

            # Template processing
            if self.use_templates and "template" in feats:
                s_template, z_template = self.template_module(feats, s_trunk, z_trunk)
                s_trunk = s_template
                z_trunk = z_template

            # Recycling loop
            for _ in range(recycling_steps + 1):
                # Pairformer
                s_trunk, z_trunk = self.pairformer_module(
                    s_trunk, z_trunk, feats
                )

            # Distogram prediction
            distogram_logits = self.distogram_module(z_trunk)
            dict_out["distogram_logits"] = distogram_logits

            # Contact conditioning
            z_conditioned = self.contact_conditioning(z_trunk, feats)

            # Diffusion conditioning
            s_conditioned, z_conditioned = self.diffusion_conditioning(
                s_trunk, z_conditioned, feats
            )

            # Structure prediction
            structure_out = self.structure_module(
                s_conditioned,
                z_conditioned,
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

            # Affinity prediction
            if self.affinity_prediction:
                affinity_out = self.affinity_module(
                    s_trunk,
                    z_trunk,
                    feats,
                    structure_out.get("coords", None),
                )
                dict_out.update(affinity_out)

            # B-factor prediction
            if self.predict_bfactor:
                bfactor_out = self.bfactor_module(s_trunk)
                dict_out.update(bfactor_out)

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
