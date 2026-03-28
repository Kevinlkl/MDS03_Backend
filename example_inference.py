"""
Example: Using trained models in your backend for inference.
Shows how to load and use the 3D latent diffusion model.
"""

from config import Config
from model_loader import (
    LatentDiffusion3DLoader,
    StableDiffusionLoRALoader,
    print_model_status,
)


def example_load_models() -> None:
    """Example 1: Load all models and check status."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Load Models and Check Status")
    print("=" * 70)

    Config.ensure_model_dirs_exist()
    Config.print_config()
    print_model_status()


def example_3d_latent_diffusion() -> None:
    """Example 2: Load and use 3D latent diffusion for T1 synthesis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: 3D Latent Diffusion Inference")
    print("=" * 70)

    try:
        print("Loading autoencoder...")
        _, ae_cfg = LatentDiffusion3DLoader.load_autoencoder()
        print(f"Loaded from: {ae_cfg['checkpoint_path']}")

        print("Loading diffusion UNet (EMA)...")
        _, unet_cfg = LatentDiffusion3DLoader.load_diffusion_unet(use_ema=True)
        print(f"Loaded from: {unet_cfg['checkpoint_path']}")
        print(f"Scale factor: {unet_cfg['scale_factor']}")

        print("Loading noise scheduler...")
        _ = LatentDiffusion3DLoader.load_noise_scheduler()
        print("Loaded scheduler")

        print("\n" + "-" * 70)
        print("Ready to generate synthetic T1 volumes")
        print("-" * 70)
        print(
            f"Input shape: {unet_cfg['latent_channels']} channels, "
            f"spatial shape inferred from latent encoding"
        )
        print(f"Device in use: {unet_cfg['device']}")
        print(f"Using EMA: {unet_cfg['use_ema']}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you copied model files from Colab and updated .env paths.")
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        print("\nIf this is CUDA-related, set DEVICE=cpu in .env and rerun.")


def example_stable_diffusion_lora() -> None:
    """Example 3: Load and use Stable Diffusion with LoRA."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Stable Diffusion with LoRA Inference")
    print("=" * 70)

    try:
        print("Loading Stable Diffusion pipeline...")
        _ = StableDiffusionLoRALoader.load_pipeline(
            base_model="runwayml/stable-diffusion-v1-5"
        )
        print("Pipeline loaded with LoRA weights")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure models/sd_t1_mri/lora exists and contains LoRA weights.")
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        print("\nIf this is CUDA-related, set DEVICE=cpu in .env and rerun.")


def main() -> None:
    print("\n" + "=" * 70)
    print("MODEL LOADING EXAMPLES")
    print("=" * 70)

    example_load_models()
    example_3d_latent_diffusion()
    example_stable_diffusion_lora()

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Keep training/generation on Colab GPU.")
    print("2. Copy trained checkpoints to local models folder.")
    print("3. Run local backend for inference with CPU fallback.")


if __name__ == "__main__":
    main()
