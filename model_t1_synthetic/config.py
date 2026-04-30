# config.py

from pathlib import Path
import torch


BASE_DIR = Path(__file__).resolve().parent

CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
GT_DIR = BASE_DIR / "samples"
GENERATED_DIR = BASE_DIR / "outputs" / "synthetic"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


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
    GENERATED_DIR = GENERATED_DIR
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
    PIXDIM = (1.0, 1.0, 1.0)

    INTENSITY_LOWER = 0.5
    INTENSITY_UPPER = 99.5
    B_MIN = -1.0
    B_MAX = 1.0

    # -------------------------------
    # Autoencoder settings
    # -------------------------------
    LATENT_CHANNELS = 4
    AUTOENCODER_CHANNELS = (32, 64, 128)

    # -------------------------------
    # Diffusion settings
    # -------------------------------
    DIFFUSION_IN_CHANNELS = LATENT_CHANNELS
    DIFFUSION_OUT_CHANNELS = LATENT_CHANNELS
    DIFFUSION_CHANNELS = (64, 128, 128)
    DIFFUSION_HEAD_CHANNELS = (0, 32, 32)

    # -------------------------------
    # Scheduler
    # -------------------------------
    NUM_TRAIN_TIMESTEPS = 1000
    BETA_START = 0.0015
    BETA_END = 0.0195

    # -------------------------------
    # Training / hyperparameters
    # -------------------------------
    AE_EPOCHS = 80
    DIFF_EPOCHS = 200
    AE_LR = 1e-4
    DIFF_LR = 3e-5
    BATCH_SIZE = 1
    NUM_WORKERS = 2
    SEED = 42

    # -------------------------------
    # Stability / inference knobs
    # -------------------------------
    NUM_INFERENCE_STEPS = 1000
    USE_EMA = True
    EMA_DECAY = 0.9995
    OFFSET_NOISE = 0.1
    GRAD_CLIP = 1.0

    # -------------------------------
    # FID / evaluation defaults
    # -------------------------------
    FID_NUM_SYNTH = 1000
    FID_NUM_REAL = 1000
    FID_SLICE_AXIS = 2
    FID_IMAGE_SIZE = 299
    FID_GEN_BATCH = 2