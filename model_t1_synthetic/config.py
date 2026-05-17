# config.py

from pathlib import Path
import torch


BASE_DIR = Path(__file__).resolve().parent

CHECKPOINTS_DIR = BASE_DIR / "T1 Synthetic Path"
OUTPUTS_DIR = BASE_DIR / "outputs"
GENERATED_DIR = OUTPUTS_DIR / "generated"
UPLOADS_DIR = BASE_DIR / "uploads"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class Config:
    # -------------------------------
    # Device
    # -------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # paths
    BASE_DIR = BASE_DIR
    CHECKPOINTS_DIR = CHECKPOINTS_DIR
    OUTPUTS_DIR = OUTPUTS_DIR
    GENERATED_DIR = GENERATED_DIR
    UPLOADS_DIR = UPLOADS_DIR
    SYNTHETIC_PATH_DIR = BASE_DIR / "T1 Synthetic Path"
    REAL_T1_FEATURES_PATH = SYNTHETIC_PATH_DIR / "real_t1_features.npy"
    REAL_T1_MU_PATH = SYNTHETIC_PATH_DIR / "real_t1_mu.npy"
    REAL_T1_SIGMA_PATH = SYNTHETIC_PATH_DIR / "real_t1_sigma.npy"

    # Synthetic T1 metrics
    SYNTHETIC_T1_FID = None
    SYNTHETIC_T1_KID_MEAN = None
    SYNTHETIC_T1_KID_STD = None
    # -------------------------------
    # Checkpoints
    # -------------------------------
    AUTOENCODER_CKPT = CHECKPOINTS_DIR / "autoencoder_best.pth"
    LATENT_DIFFUSION_CKPT = CHECKPOINTS_DIR / "latent_diffusion_best.pth"

    # -------------------------------
    # Input / Output
    # -------------------------------
    INPUT_KEY = "t1"
    OUTPUT_FILENAME = "pred_t2.nii.gz"

    # -------------------------------
    # Preprocessing
    # -------------------------------
    SPATIAL_SIZE = (96, 96, 64)
    PIXDIM = (1.5, 1.5, 1.5)

    INTENSITY_LOWER = 0.0
    INTENSITY_UPPER = 99.5
    B_MIN = -1.0
    B_MAX = 1.0

    # -------------------------------
    # Autoencoder settings
    # -------------------------------
    LATENT_CHANNELS = 8
    AUTOENCODER_CHANNELS = (32, 64, 128)

    # -------------------------------
    # Diffusion settings
    # -------------------------------
    LATENT_CHANNELS = 8
    DIFF_NUM_CHANNELS = (128, 256, 256)
    DIFF_NUM_HEAD_CHANNELS = (0, 64, 64)

    # -------------------------------
    # Scheduler
    # -------------------------------
    NUM_TRAIN_TIMESTEPS = 1000
    BETA_START = 0.0015
    BETA_END = 0.0195

    # ============================================================
    # Precomputed dataset-level evaluation metrics
    # ============================================================

    PRECOMPUTED_FID = 112.5451

    # Replace these with your actual KID result
    PRECOMPUTED_KID_MEAN = 0.084321
    PRECOMPUTED_KID_STD = 0.006512
    # -------------------------------
    # Inference
    # -------------------------------
    NUM_INFERENCE_STEPS = 1000
    