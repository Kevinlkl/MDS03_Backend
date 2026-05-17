"""
Pure batch FID score calculation between real and synthetic T1 volumes.

Corrected version:
- Real volumes use the same preprocessing style as notebook:
  Orientation -> Spacing -> Resize -> Percentile intensity scaling to [-1, 1]
- Synthetic volumes are assumed already generated in [-1, 1]
- Both real and synthetic are compared at the same TARGET_SHAPE
- Same matched slice percentages are used for both groups
- TorchMetrics FID is used
"""

import sys
import subprocess
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "torchmetrics", "torch-fidelity"
    ])
    from torchmetrics.image.fid import FrechetInceptionDistance

from monai.transforms import (
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    OrientationD,
    SpacingD,
    ResizeD,
    ScaleIntensityRangePercentilesD,
)


# ============================================================
# Configuration
# ============================================================

TARGET_SHAPE = (128, 128, 96)
PIXDIM = (1.0, 1.0, 1.0)

NORM_LOW_PCT = 0.5
NORM_HIGH_PCT = 99.5

FID_SLICE_AXIS = 2
FID_SLICE_PERCENTAGES = np.array(
    [0.15, 0.325, 0.50, 0.675, 0.85],
    dtype=np.float32
)

FID_IMAGE_SIZE = 299
REAL_VOLUMES_USE_FRACTION = 1
DEFAULT_REAL_VOLUMES_USE_PERCENT = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BACKEND_ROOT = Path(__file__).parent.resolve()
REAL_VOLUMES_DIR = BACKEND_ROOT / "T1"
SYNTHETIC_VOLUMES_DIR = BACKEND_ROOT / "generated"


# ============================================================
# Preprocessing transform for real volumes
# ============================================================

val_transforms = Compose([
    LoadImageD(keys="image", image_only=False),
    EnsureChannelFirstD(keys="image"),
    OrientationD(keys="image", axcodes="RAS"),
    SpacingD(keys="image", pixdim=PIXDIM, mode="bilinear"),
    ResizeD(keys="image", spatial_size=TARGET_SHAPE, mode="trilinear"),
    ScaleIntensityRangePercentilesD(
        keys="image",
        lower=NORM_LOW_PCT,
        upper=NORM_HIGH_PCT,
        b_min=-1.0,
        b_max=1.0,
        clip=True,
    ),
])


# ============================================================
# Helper functions
# ============================================================

def to_01_from_minus1_plus1(volume):
    volume = np.asarray(volume, dtype=np.float32)
    volume = np.clip(volume, -1.0, 1.0)
    return (volume + 1.0) / 2.0


def preprocess_real_volume(path):
    sample = val_transforms({"image": str(path)})
    image = sample["image"] if isinstance(sample, dict) else sample
    image = torch.as_tensor(image).float()
    return image[0].cpu().numpy()


def load_volume(path):
    return nib.load(str(path)).get_fdata().astype(np.float32)


def resize_volume_to_target(volume, target_shape):
    if tuple(volume.shape) == tuple(target_shape):
        return volume.astype(np.float32)

    with torch.no_grad():
        vol_t = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        vol_t = F.interpolate(
            vol_t,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )
        return vol_t[0, 0].cpu().numpy().astype(np.float32)


def get_matched_fid_slice_indices(
    target_shape,
    axis=2,
    percentages=FID_SLICE_PERCENTAGES
):
    depth = int(target_shape[axis])
    indices = np.round(percentages * (depth - 1)).astype(int)
    indices = np.clip(indices, 0, depth - 1)
    return indices


def get_slices_at_indices(volume, indices, axis=2):
    slices = []

    for idx in indices:
        idx = int(idx)

        if axis == 0:
            sl = volume[idx, :, :]
        elif axis == 1:
            sl = volume[:, idx, :]
        else:
            sl = volume[:, :, idx]

        slices.append(sl.astype(np.float32))

    return slices


def slice_to_fid_uint8_tensor(slice_2d, image_size=299):
    img = np.asarray(slice_2d, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)

    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(
        t,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    t = t.repeat(1, 3, 1, 1)
    t = (t * 255.0).round().clamp(0, 255).to(torch.uint8)

    return t


def parse_args():
    parser = argparse.ArgumentParser(description="Compute matched-slice FID for T1 volumes.")
    parser.add_argument(
        "--real-percent",
        type=float,
        default=DEFAULT_REAL_VOLUMES_USE_PERCENT,
        help="Percentage of real T1 volumes to use (0-100). Default: 10",
    )
    return parser.parse_args()


# ============================================================
# Main FID calculation
# ============================================================

def compute_batch_fid(real_volume_percent=DEFAULT_REAL_VOLUMES_USE_PERCENT):
    fid_slice_indices = get_matched_fid_slice_indices(
        TARGET_SHAPE,
        axis=FID_SLICE_AXIS,
        percentages=FID_SLICE_PERCENTAGES,
    )

    print("=" * 70)
    print("LOCAL MATCHED-SLICE FID EVALUATOR")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Target shape: {TARGET_SHAPE}")
    print(f"Voxel spacing: {PIXDIM}")
    print(f"Real intensity scaling: percentile {NORM_LOW_PCT}-{NORM_HIGH_PCT} -> [-1, 1]")
    print(f"Slice axis: {FID_SLICE_AXIS}")
    print(f"Slice percentages: {FID_SLICE_PERCENTAGES.tolist()}")
    print(f"Matched slice indices: {fid_slice_indices.tolist()}")
    print()

    fid_metric = FrechetInceptionDistance(
        feature=2048,
        normalize=False
    ).to(DEVICE)

    if not (0 < real_volume_percent <= 100):
        raise ValueError("real_volume_percent must be in the range (0, 100].")

    real_volumes = (
        sorted(REAL_VOLUMES_DIR.glob("*.nii")) +
        sorted(REAL_VOLUMES_DIR.glob("*.nii.gz"))
    )

    print(f"Reading real volumes from: {REAL_VOLUMES_DIR}")
    print(f"Found {len(real_volumes)} real volumes")

    if len(real_volumes) == 0:
        raise RuntimeError(f"No real NIfTI files found in {REAL_VOLUMES_DIR}")

    num_real_to_use = max(
        1,
        int(round(len(real_volumes) * (real_volume_percent / 100.0)))
    )

    real_volumes = real_volumes[:num_real_to_use]

    print(
        f"Using {num_real_to_use} real volumes "
        f"({real_volume_percent:.1f}%)"
    )

    synthetic_volumes = sorted(
        SYNTHETIC_VOLUMES_DIR.glob("synthetic_t1_*.nii.gz")
    )

    print(f"\nReading synthetic volumes from: {SYNTHETIC_VOLUMES_DIR}")
    print(f"Found {len(synthetic_volumes)} synthetic volumes")

    if len(synthetic_volumes) == 0:
        raise RuntimeError(
            f"No synthetic NIfTI files found in {SYNTHETIC_VOLUMES_DIR}"
        )

    print("\nProcessing real volumes...")
    for path in tqdm(real_volumes, desc="Real", ncols=80):
        vol = preprocess_real_volume(path)
        vol_01 = to_01_from_minus1_plus1(vol)

        slices = get_slices_at_indices(
            vol_01,
            fid_slice_indices,
            axis=FID_SLICE_AXIS
        )

        for sl in slices:
            fid_img = slice_to_fid_uint8_tensor(
                sl,
                FID_IMAGE_SIZE
            ).to(DEVICE)

            fid_metric.update(fid_img, real=True)

    print("\nProcessing synthetic volumes...")
    for path in tqdm(synthetic_volumes, desc="Synthetic", ncols=80):
        vol = load_volume(path)
        vol = resize_volume_to_target(vol, TARGET_SHAPE)
        vol_01 = to_01_from_minus1_plus1(vol)

        slices = get_slices_at_indices(
            vol_01,
            fid_slice_indices,
            axis=FID_SLICE_AXIS
        )

        for sl in slices:
            fid_img = slice_to_fid_uint8_tensor(
                sl,
                FID_IMAGE_SIZE
            ).to(DEVICE)

            fid_metric.update(fid_img, real=False)

    fid_score = float(
        fid_metric.compute()
        .detach()
        .cpu()
        .item()
    )

    print()
    print("=" * 70)
    print(f"TorchMetrics Matched-Slice Inception V3 FID: {fid_score:.4f}")
    print(f"Real volumes used: {num_real_to_use}")
    print(f"Synthetic volumes used: {len(synthetic_volumes)}")
    print(f"Real slices used: {num_real_to_use * len(fid_slice_indices)}")
    print(f"Synthetic slices used: {len(synthetic_volumes) * len(fid_slice_indices)}")
    print("=" * 70)

    return fid_score


if __name__ == "__main__":
    args = parse_args()
    fid = compute_batch_fid(real_volume_percent=args.real_percent)