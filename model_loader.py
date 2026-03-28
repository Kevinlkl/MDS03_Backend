"""
Model loading utilities for 3D latent diffusion and Stable Diffusion models.
Abstracts away path management and checkpoint loading.
"""

from pathlib import Path
from typing import Optional, Tuple
import warnings

import torch

from config import Config


def _resolve_device(requested_device: Optional[str]) -> torch.device:
    """Return a safe device; fallback to CPU when CUDA cannot be used."""
    wanted = (requested_device or Config.DEVICE or "cpu").lower().strip()
    if wanted == "cuda":
        if not torch.cuda.is_available():
            warnings.warn(
                "CUDA requested but not available. Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        try:
            # Probe allocation catches driver/runtime mismatch early.
            _probe = torch.zeros((1,), device="cuda")
            del _probe
            return torch.device("cuda")
        except Exception as e:
            warnings.warn(
                f"CUDA requested but failed to initialize ({e}). Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
    return torch.device("cpu")


class LatentDiffusion3DLoader:
    """Load 3D latent diffusion model components from checkpoints."""

    @staticmethod
    def load_autoencoder(
        device: str = None,
        checkpoint_path: Optional[Path] = None,
    ) -> Tuple[object, dict]:
        """
        Load the 3D AutoencoderKL.

        Args:
            device: torch device ("cuda" or "cpu")
            checkpoint_path: override default checkpoint path

        Returns:
            (autoencoder model, config dict)
        """
        runtime_device = _resolve_device(device)
        checkpoint_path = Path(checkpoint_path or Config.AE_CHECKPOINT)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Autoencoder checkpoint not found: {checkpoint_path}\n"
                f"Expected to find it at: {Config.AE_CHECKPOINT}\n"
                f"See SETUP.md for how to organize your model files."
            )

        try:
            from generative.networks.nets import AutoencoderKL
        except ImportError as exc:
            raise ImportError(
                "monai-generative not installed. "
                "Run: pip install monai-generative"
            ) from exc

        ckpt = torch.load(checkpoint_path, map_location="cpu")

        config = ckpt.get("config", {})
        target_shape = config.get("target_shape", Config.TARGET_SHAPE)
        target_pixdim = config.get("target_pixdim", (1.0, 1.0, 1.0))

        autoencoder = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=(64, 128, 256),
            latent_channels=8,
            num_res_blocks=2,
            norm_num_groups=16,
            attention_levels=(False, False, True),
        ).to(runtime_device)

        autoencoder.load_state_dict(ckpt.get("autoencoder_state_dict", ckpt))
        autoencoder.eval()

        for p in autoencoder.parameters():
            p.requires_grad = False

        return autoencoder, {
            "checkpoint_path": str(checkpoint_path),
            "target_shape": target_shape,
            "target_pixdim": target_pixdim,
            "device": str(runtime_device),
        }

    @staticmethod
    def load_diffusion_unet(
        device: str = None,
        checkpoint_path: Optional[Path] = None,
        use_ema: bool = True,
    ) -> Tuple[object, dict]:
        """
        Load the diffusion UNet (with optional EMA).

        Args:
            device: torch device
            checkpoint_path: override default checkpoint path
            use_ema: if True, load EMA weights (recommended for inference)

        Returns:
            (unet model, config dict)
        """
        runtime_device = _resolve_device(device)
        checkpoint_path = Path(checkpoint_path or Config.DIFFUSION_CHECKPOINT)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Diffusion checkpoint not found: {checkpoint_path}\n"
                f"Expected to find it at: {Config.DIFFUSION_CHECKPOINT}\n"
                f"See SETUP.md for how to organize your model files."
            )

        try:
            from generative.networks.nets import DiffusionModelUNet
        except ImportError as exc:
            raise ImportError(
                "monai-generative not installed. "
                "Run: pip install monai-generative"
            ) from exc

        ckpt = torch.load(checkpoint_path, map_location="cpu")

        latent_channels = ckpt.get("latent_channels", 8)
        unet = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=latent_channels,
            out_channels=latent_channels,
            num_res_blocks=2,
            num_channels=(128, 256, 256),
            attention_levels=(False, True, True),
            num_head_channels=(0, 64, 64),
        ).to(runtime_device)

        state_dict_key = "ema_unet_state_dict" if use_ema else "unet_state_dict"
        if state_dict_key in ckpt:
            unet.load_state_dict(ckpt[state_dict_key])
        else:
            unet.load_state_dict(ckpt.get("unet_state_dict", ckpt))

        unet.eval()
        for p in unet.parameters():
            p.requires_grad = False

        scale_factor = ckpt.get("scale_factor", 1.0)

        return unet, {
            "checkpoint_path": str(checkpoint_path),
            "latent_channels": latent_channels,
            "scale_factor": scale_factor,
            "device": str(runtime_device),
            "use_ema": use_ema,
        }

    @staticmethod
    def load_noise_scheduler(checkpoint_path: Optional[Path] = None) -> object:
        """Load the DDPM noise scheduler."""
        try:
            from generative.networks.schedulers import DDPMScheduler
        except ImportError as exc:
            raise ImportError(
                "monai-generative not installed. "
                "Run: pip install monai-generative"
            ) from exc

        checkpoint_path = Path(checkpoint_path or Config.DIFFUSION_CHECKPOINT)
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            config = ckpt.get("config", {})
            num_timesteps = config.get("num_diffusion_timesteps", 1000)
        else:
            num_timesteps = 1000

        scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            schedule="scaled_linear_beta",
            beta_start=0.00085,
            beta_end=0.012,
        )
        return scheduler


class StableDiffusionLoRALoader:
    """Load Stable Diffusion with LoRA weights."""

    @staticmethod
    def load_pipeline(
        device: str = None,
        lora_checkpoint_path: Optional[Path] = None,
        base_model: str = "runwayml/stable-diffusion-v1-5",
    ) -> object:
        """
        Load StableDiffusionPipeline with LoRA weights.

        Args:
            device: torch device
            lora_checkpoint_path: path to LoRA weights
            base_model: base model identifier

        Returns:
            StableDiffusionPipeline with LoRA loaded
        """
        runtime_device = _resolve_device(device)
        lora_checkpoint_path = Path(lora_checkpoint_path or Config.SD_LORA_CHECKPOINT)

        try:
            from diffusers import StableDiffusionPipeline
        except ImportError as exc:
            raise ImportError(
                "diffusers not installed. "
                "Run: pip install diffusers"
            ) from exc

        pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if runtime_device.type == "cuda" else torch.float32,
            safety_checker=None,
        )

        if lora_checkpoint_path.exists():
            try:
                pipe.load_lora_weights(str(lora_checkpoint_path))
                print(f"Loaded LoRA from: {lora_checkpoint_path}")
            except Exception as e:
                print(f"Warning: Could not load LoRA from {lora_checkpoint_path}: {e}")

        pipe = pipe.to(runtime_device)
        return pipe


def get_model_status() -> dict:
    """Get availability and size info for all models."""
    return Config.check_model_availability()


def print_model_status() -> None:
    """Print readable model status."""
    status = get_model_status()
    print("\n" + "=" * 60)
    print("MODEL STATUS")
    print("=" * 60)
    for model_name, exists in status["status"].items():
        status_str = "READY" if exists else "MISSING"
        size_str = f" ({status['sizes'][model_name]:.2f} GB)" if model_name in status["sizes"] else ""
        print(f"{model_name:20s} {status_str}{size_str}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    Config.ensure_model_dirs_exist()
    Config.print_config()
    print_model_status()
