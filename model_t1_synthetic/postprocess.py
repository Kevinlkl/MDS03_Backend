import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Iterator, Optional
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# -------------------------------
# Safe normalization
# -------------------------------
def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if arr.size == 0:
        return arr

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    if max_val > min_val:
        return (arr - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(arr, dtype=np.float32)


# -------------------------------
# NIfTI slices
# -------------------------------
def _iter_nifti_slices(path: Path) -> Iterator[torch.Tensor]:
    volume = nib.load(str(path)).get_fdata(dtype=np.float32)

    if volume.ndim == 4:
        volume = volume[..., 0]

    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.shape}")

    volume = _safe_normalize(volume)

    for z in range(volume.shape[-1]):
        slice_2d = volume[:, :, z]

        # ❗ skip constant slices
        if np.max(slice_2d) == np.min(slice_2d):
            continue

        yield torch.from_numpy(slice_2d).float().unsqueeze(0)


# -------------------------------
# Image slices
# -------------------------------
def _iter_image_slices(path: Path) -> Iterator[torch.Tensor]:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.float32)

    arr = _safe_normalize(arr)

    if np.max(arr) == np.min(arr):
        return  # skip useless image

    yield torch.from_numpy(arr).float().unsqueeze(0)


# -------------------------------
# Batch volume slices (generated)
# -------------------------------
def _iter_batch_volume_slices(batch_np: np.ndarray) -> Iterator[torch.Tensor]:
    if batch_np.ndim == 5:
        volumes = batch_np[:, 0]
    elif batch_np.ndim == 4:
        volumes = batch_np
    else:
        raise ValueError(f"Expected 4D/5D batch, got {batch_np.shape}")

    for vol in volumes:
        vol = np.nan_to_num(vol.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        vol = np.clip(vol, -1.0, 1.0)
        vol = (vol + 1.0) / 2.0

        # ❗ skip constant volumes
        if np.max(vol) == np.min(vol):
            continue

        for z in range(vol.shape[-1]):
            slice_2d = vol[:, :, z]

            if np.max(slice_2d) == np.min(slice_2d):
                continue

            yield torch.from_numpy(slice_2d).float().unsqueeze(0)


# -------------------------------
# Feature extraction
# -------------------------------
def _extract_features_from_iterator(
    iterator: Iterator[torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:

    features = []
    batch = []

    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)

    for slice_tensor in iterator:

        # ❗ skip constant tensors
        if torch.max(slice_tensor) == torch.min(slice_tensor):
            continue

        batch.append(slice_tensor)

        if len(batch) >= batch_size:
            x = torch.stack(batch).to(device)
            x = x.repeat(1, 3, 1, 1)
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

            # safe normalization
            x = (x - mean) / (std + 1e-8)

            with torch.no_grad():
                f = model(x)

            if isinstance(f, tuple):
                f = f[0]

            if f.ndim == 4:
                f = torch.flatten(F.adaptive_avg_pool2d(f, 1), 1)

            features.append(f.cpu().numpy())
            batch = []

    if batch:
        x = torch.stack(batch).to(device)
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - mean) / (std + 1e-8)

        with torch.no_grad():
            f = model(x)

        if isinstance(f, tuple):
            f = f[0]

        if f.ndim == 4:
            f = torch.flatten(F.adaptive_avg_pool2d(f, 1), 1)

        features.append(f.cpu().numpy())

    if not features:
        raise ValueError("No valid features extracted (all slices were constant).")

    return np.concatenate(features, axis=0)


# -------------------------------
# Gaussian stats (SAFE)
# -------------------------------
def _compute_gaussian_stats(features: np.ndarray):
    if features.shape[0] < 2:
        raise ValueError("Need at least 2 feature vectors")

    # ❗ detect collapsed features
    if np.allclose(features, features[0]):
        raise ValueError("All features identical → invalid FID")

    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)

    # add stability
    sigma += np.eye(sigma.shape[0]) * 1e-6

    return mu, sigma


# -------------------------------
# Matrix sqrt (stable)
# -------------------------------
def _matrix_sqrt_psd(matrix):
    matrix = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(matrix)

    eigvals = np.clip(eigvals, 0, None)

    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


# -------------------------------
# FID
# -------------------------------
def _calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2

    covmean = _matrix_sqrt_psd(sigma1 @ sigma2)

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

    return float(max(fid, 0.0))