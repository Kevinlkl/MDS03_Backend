"""
Configuration management for model paths and settings.
Supports both environment variables and local .env files.
"""

import os
from pathlib import Path


def _load_env_fallback(env_file: Path) -> None:
    """Minimal .env loader used when python-dotenv is unavailable."""
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


env_file = Path(__file__).parent / ".env"
try:
    from dotenv import load_dotenv

    if env_file.exists():
        load_dotenv(env_file, override=True)
except ImportError:
    _load_env_fallback(env_file)


class Config:
    """Centralized configuration for model paths and settings."""

    PROJECT_ROOT = Path(__file__).parent.resolve()

    MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models")).expanduser().resolve()
    OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", PROJECT_ROOT / "outputs")).expanduser().resolve()

    LATENT_DIFFUSION_DIR = Path(
        os.getenv(
            "LATENT_DIFFUSION_DIR",
            MODELS_DIR / "synthetic_t1_3d_ldm",
        )
    ).expanduser().resolve()

    SD_LORA_DIR = Path(
        os.getenv(
            "SD_LORA_DIR",
            MODELS_DIR / "sd_t1_mri",
        )
    ).expanduser().resolve()

    AE_CHECKPOINT = Path(
        os.getenv(
            "AE_CHECKPOINT",
            LATENT_DIFFUSION_DIR / "checkpoints" / "autoencoder_best.pth",
        )
    ).expanduser().resolve()

    DIFFUSION_CHECKPOINT = Path(
        os.getenv(
            "DIFFUSION_CHECKPOINT",
            LATENT_DIFFUSION_DIR / "checkpoints" / "latent_diffusion_best.pth",
        )
    ).expanduser().resolve()

    SD_LORA_CHECKPOINT = Path(
        os.getenv(
            "SD_LORA_CHECKPOINT",
            SD_LORA_DIR / "lora",
        )
    ).expanduser().resolve()

    DEVICE = os.getenv("DEVICE", "cuda")
    USE_AMP = os.getenv("USE_AMP", "true").lower() == "true"

    TARGET_SHAPE = tuple(map(int, os.getenv("TARGET_SHAPE", "96,96,64").split(",")))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))

    @classmethod
    def ensure_model_dirs_exist(cls) -> None:
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LATENT_DIFFUSION_DIR.mkdir(parents=True, exist_ok=True)
        cls.SD_LORA_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def check_model_availability(cls) -> dict:
        status = {
            "autoencoder": cls.AE_CHECKPOINT.exists(),
            "diffusion": cls.DIFFUSION_CHECKPOINT.exists(),
            "sd_lora": cls.SD_LORA_CHECKPOINT.exists(),
        }

        sizes = {}
        if status["autoencoder"]:
            sizes["autoencoder"] = cls.AE_CHECKPOINT.stat().st_size / (1024**3)
        if status["diffusion"]:
            sizes["diffusion"] = cls.DIFFUSION_CHECKPOINT.stat().st_size / (1024**3)
        if status["sd_lora"]:
            lora_size = sum(
                p.stat().st_size for p in cls.SD_LORA_CHECKPOINT.rglob("*") if p.is_file()
            )
            sizes["sd_lora"] = lora_size / (1024**3)

        return {"status": status, "sizes": sizes}

    @classmethod
    def print_config(cls) -> None:
        print("=" * 60)
        print("MODEL CONFIGURATION")
        print("=" * 60)
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"Models Dir: {cls.MODELS_DIR}")
        print(f"Outputs Dir: {cls.OUTPUTS_DIR}")
        print("\nModel Checkpoints:")
        print(f"  AE: {cls.AE_CHECKPOINT}")
        print(f"     Exists: {cls.AE_CHECKPOINT.exists()}")
        print(f"  Diffusion: {cls.DIFFUSION_CHECKPOINT}")
        print(f"     Exists: {cls.DIFFUSION_CHECKPOINT.exists()}")
        print(f"  SD LoRA: {cls.SD_LORA_CHECKPOINT}")
        print(f"     Exists: {cls.SD_LORA_CHECKPOINT.exists()}")
        print(f"\nDevice: {cls.DEVICE}")
        print(f"Use AMP: {cls.USE_AMP}")
        print(f"Target Shape: {cls.TARGET_SHAPE}")
        print("=" * 60)


if __name__ == "__main__":
    Config.ensure_model_dirs_exist()
    Config.print_config()
    availability = Config.check_model_availability()
    print("\nModel Availability:")
    for model_name, exists in availability["status"].items():
        status_str = "Available" if exists else "Missing"
        print(f"  {model_name}: {status_str}")
        if model_name in availability["sizes"]:
            print(f"    Size: {availability['sizes'][model_name]:.2f} GB")
