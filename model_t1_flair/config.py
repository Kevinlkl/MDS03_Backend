# config.py

from pathlib import Path
import torch


BASE_DIR = Path(__file__).resolve().parent

CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
GT_DIR = BASE_DIR / "samples"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)


class Config:
    # -------------------------------
    # Device
    # -------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # paths
    BASE_DIR = BASE_DIR
    CHECKPOINTS_DIR = CHECKPOINTS_DIR
    UPLOADS_DIR = UPLOADS_DIR
    OUTPUTS_DIR = OUTPUTS_DIR
    GT_DIR = GT_DIR
    # -------------------------------
    # Checkpoints
    # -------------------------------
    AUTOENCODER_CKPT = CHECKPOINTS_DIR / "autoencoder_best_flair.pth"
    LATENT_DIFFUSION_CKPT = CHECKPOINTS_DIR / "latent_diffusion_best_flair.pth"

    # -------------------------------
    # Preprocessing
    # -------------------------------
    SPATIAL_SIZE = (96, 96, 96)
    PIXDIM = (1.5, 1.5, 1.5)

    INTENSITY_LOWER = 0.0
    INTENSITY_UPPER = 99.5
    B_MIN = -1.0
    B_MAX = 1.0

    # -------------------------------
    # Autoencoder settings
    # -------------------------------
    LATENT_CHANNELS = 8
    AUTOENCODER_CHANNELS = (64, 128, 256)

    # -------------------------------
    # Diffusion settings
    # -------------------------------
    DIFFUSION_IN_CHANNELS = LATENT_CHANNELS * 2
    DIFFUSION_OUT_CHANNELS = LATENT_CHANNELS

    # -------------------------------
    # Scheduler
    # -------------------------------
    NUM_TRAIN_TIMESTEPS = 1000
    BETA_START = 0.0015
    BETA_END = 0.0195

    # -------------------------------
    # Inference
    # -------------------------------
    NUM_INFERENCE_STEPS = 1000
    FID_RUNTIME_BATCH_SIZE = 16
    FID_RUNTIME_MAX_SLICES = 64