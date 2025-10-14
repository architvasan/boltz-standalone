"""Tests for standalone inference functionality."""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from boltz.model.models.boltz1_standalone import Boltz1Standalone
from boltz.model.models.boltz2_standalone import Boltz2Standalone
from boltz.model.utils.checkpoint_loader import (
    detect_model_type,
    convert_lightning_state_dict,
    get_checkpoint_info,
)
from boltz.inference import BoltzInference


class TestStandaloneModels:
    """Test standalone model classes."""
    
    def test_boltz1_standalone_init(self):
        """Test Boltz1Standalone initialization."""
        # Mock hyperparameters
        hparams = {
            "atom_s": 128,
            "atom_z": 128,
            "token_s": 384,
            "token_z": 128,
            "num_bins": 64,
            "training_args": {},
            "validation_args": {},
            "embedder_args": {"atom_feature_dim": 128},
            "msa_args": {},
            "pairformer_args": {"num_blocks": 4},
            "score_model_args": {},
            "diffusion_process_args": {},
            "diffusion_loss_args": {},
            "confidence_model_args": {},
        }
        
        # This should not raise an error
        model = Boltz1Standalone(**hparams)
        assert isinstance(model, torch.nn.Module)
        assert model.atom_s == 128
        assert model.token_s == 384
    
    def test_boltz2_standalone_init(self):
        """Test Boltz2Standalone initialization."""
        # Mock hyperparameters
        hparams = {
            "atom_s": 128,
            "atom_z": 128,
            "token_s": 384,
            "token_z": 128,
            "num_bins": 64,
            "training_args": {},
            "validation_args": {},
            "embedder_args": {"atom_feature_dim": 128},
            "msa_args": {},
            "pairformer_args": {"num_blocks": 4},
            "score_model_args": {},
            "diffusion_process_args": {},
            "diffusion_loss_args": {},
        }
        
        # This should not raise an error
        model = Boltz2Standalone(**hparams)
        assert isinstance(model, torch.nn.Module)
        assert model.atom_s == 128
        assert model.token_s == 384


class TestCheckpointLoader:
    """Test checkpoint loading utilities."""
    
    def test_detect_model_type_boltz1(self):
        """Test model type detection for Boltz1."""
        # Mock checkpoint for Boltz1
        checkpoint = {
            "hyper_parameters": {
                "atom_s": 128,
                "token_s": 384,
                # No Boltz2-specific parameters
            },
            "state_dict": {
                "input_embedder.weight": torch.randn(10, 10),
                "pairformer_module.weight": torch.randn(10, 10),
            }
        }
        
        model_type = detect_model_type(checkpoint)
        assert model_type == "boltz1"
    
    def test_detect_model_type_boltz2(self):
        """Test model type detection for Boltz2."""
        # Mock checkpoint for Boltz2
        checkpoint = {
            "hyper_parameters": {
                "atom_s": 128,
                "token_s": 384,
                "affinity_prediction": True,  # Boltz2-specific
            },
            "state_dict": {
                "input_embedder.weight": torch.randn(10, 10),
                "diffusion_conditioning.weight": torch.randn(10, 10),
            }
        }
        
        model_type = detect_model_type(checkpoint)
        assert model_type == "boltz2"
    
    def test_convert_lightning_state_dict(self):
        """Test Lightning state dict conversion."""
        # Mock Lightning state dict
        lightning_state_dict = {
            "input_embedder.weight": torch.randn(10, 10),
            "pairformer_module.bias": torch.randn(10),
            "_metrics.train_loss": torch.tensor(0.5),
            "trainer.global_step": torch.tensor(1000),
        }
        
        converted = convert_lightning_state_dict(lightning_state_dict)
        
        # Should keep model parameters
        assert "input_embedder.weight" in converted
        assert "pairformer_module.bias" in converted
        
        # Should remove Lightning-specific keys
        assert "_metrics.train_loss" not in converted
        assert "trainer.global_step" not in converted
    
    def test_get_checkpoint_info(self):
        """Test checkpoint info extraction."""
        # Create a temporary checkpoint file
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            checkpoint = {
                "hyper_parameters": {"atom_s": 128},
                "epoch": 10,
                "global_step": 1000,
                "state_dict": {"model.weight": torch.randn(10, 10)},
                "pytorch-lightning_version": "2.5.0",
            }
            torch.save(checkpoint, f.name)
            
            info = get_checkpoint_info(f.name)
            
            assert info["model_type"] == "boltz1"  # Default detection
            assert info["epoch"] == 10
            assert info["global_step"] == 1000
            assert info["pytorch_lightning_version"] == "2.5.0"
            assert info["has_ema"] is False
            
            # Clean up
            Path(f.name).unlink()


class TestBoltzInference:
    """Test BoltzInference class."""
    
    @patch('boltz.model.utils.checkpoint_loader.load_model_from_checkpoint')
    def test_boltz_inference_init(self, mock_load_model):
        """Test BoltzInference initialization."""
        # Mock model
        mock_model = Mock()
        mock_model.__class__.__name__ = "Boltz2Standalone"
        mock_model.eval.return_value = None
        mock_model.to.return_value = None
        mock_load_model.return_value = mock_model
        
        # Initialize inference
        inference = BoltzInference(
            checkpoint_path="dummy_path.ckpt",
            device="cpu"
        )
        
        assert inference.device == "cpu"
        assert inference.model_version == "boltz2"
        mock_load_model.assert_called_once()
        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once_with("cpu")
    
    def test_device_auto_detection(self):
        """Test automatic device detection."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('boltz.model.utils.checkpoint_loader.load_model_from_checkpoint') as mock_load:
                mock_model = Mock()
                mock_model.__class__.__name__ = "Boltz1Standalone"
                mock_model.eval.return_value = None
                mock_model.to.return_value = None
                mock_load.return_value = mock_model
                
                inference = BoltzInference(
                    checkpoint_path="dummy_path.ckpt",
                    device="auto"
                )
                
                assert inference.device == "cuda"
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('boltz.model.utils.checkpoint_loader.load_model_from_checkpoint') as mock_load:
                mock_model = Mock()
                mock_model.__class__.__name__ = "Boltz1Standalone"
                mock_model.eval.return_value = None
                mock_model.to.return_value = None
                mock_load.return_value = mock_model
                
                inference = BoltzInference(
                    checkpoint_path="dummy_path.ckpt",
                    device="auto"
                )
                
                assert inference.device == "cpu"


class TestDataPipeline:
    """Test data pipeline functionality."""
    
    def test_move_batch_to_device(self):
        """Test batch device movement."""
        with patch('boltz.model.utils.checkpoint_loader.load_model_from_checkpoint') as mock_load:
            mock_model = Mock()
            mock_model.__class__.__name__ = "Boltz1Standalone"
            mock_model.eval.return_value = None
            mock_model.to.return_value = None
            mock_load.return_value = mock_model
            
            inference = BoltzInference(
                checkpoint_path="dummy_path.ckpt",
                device="cpu"
            )
            
            # Create mock batch
            batch = {
                "atom_coords": torch.randn(1, 100, 3),
                "atom_mask": torch.ones(1, 100),
                "record": Mock(),  # Should not be moved
                "all_coords": torch.randn(1, 100, 3),  # Should not be moved
                "exception": False,  # Should not be moved
            }
            
            moved_batch = inference._move_batch_to_device(batch)
            
            # Tensors should be moved (in this case, they're already on CPU)
            assert moved_batch["atom_coords"].device.type == "cpu"
            assert moved_batch["atom_mask"].device.type == "cpu"
            
            # These should not be moved
            assert "record" in moved_batch
            assert "all_coords" in moved_batch
            assert "exception" in moved_batch


if __name__ == "__main__":
    pytest.main([__file__])
