import numpy as np
import nibabel as nib
import os
import torch
import torch.nn.functional as F
from typing import Iterator, Optional, cast
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_INCEPTION_MODEL_CACHE: dict[str, torch.nn.Module] = {}
_FID_STATS_CACHE: dict[str, np.ndarray] = {}

# Lower slice count keeps local CPU runtime practical while preserving trend quality.
FID_RUNTIME_BATCH_SIZE = 16
FID_RUNTIME_MAX_SLICES = 16

def tensor_to_numpy(pred: torch.Tensor) -> np.ndarray: 
    pred = pred.detach().cpu() 
    if pred.ndim == 5: 
        pred = pred[0, 0] 
    elif pred.ndim == 4: 
        pred = pred[0] 
    return pred.numpy() 

def save_nifti(pred: torch.Tensor, output_path: str, affine=None): 
    arr = tensor_to_numpy(pred) 
    if affine is None: 
        affine = np.eye(4) 
    nii = nib.Nifti1Image(arr, affine) 
    nib.save(nii, output_path) 
    return output_path
    
# =========================
# Metric computation (CLEAN)
# =========================
def compute_psnr_ssim(pred_np: np.ndarray, gt_np: np.ndarray):
    psnr_list = []
    ssim_list = []

    B = pred_np.shape[0]
    for i in range(B):
        pred_vol = pred_np[i, 0]
        gt_vol = gt_np[i, 0]

        pred_vol = np.clip(pred_vol, -1.0, 1.0)
        gt_vol = np.clip(gt_vol, -1.0, 1.0)

        pred_vol = (pred_vol + 1.0) / 2.0
        gt_vol = (gt_vol + 1.0) / 2.0

        # PSNR (3D)
        psnr_val = psnr(gt_vol, pred_vol, data_range=1.0)

        # SSIM (slice-wise)
        ssim_slices = []
        for z in range(pred_vol.shape[0]):
            ssim_val = ssim(
                gt_vol[z],
                pred_vol[z],
                data_range=1.0
            )
            ssim_slices.append(ssim_val)

        ssim_val = float(np.mean(ssim_slices))
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


# =========================
# FID computation (device auto-selection)
# =========================
def _resolve_fid_device(device: Optional[str] = None) -> torch.device:
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    requested = torch.device(device)
    if requested.type == "cuda":
        if torch.cuda.is_available():
            return requested
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    if requested.type == "mps":
        if mps_available:
            return requested
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return requested


def _build_inception_feature_extractor(device: torch.device) -> torch.nn.Module:
    cache_key = str(device)
    cached = _INCEPTION_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from torchvision.models import Inception_V3_Weights, inception_v3
    except ImportError as exc:
        raise ImportError("torchvision is required for FID feature extraction.") from exc

    model = None
    init_errors: list[str] = []

    for kwargs in (
        {"weights": Inception_V3_Weights.IMAGENET1K_V1, "transform_input": False},
        {"weights": Inception_V3_Weights.IMAGENET1K_V1},
    ):
        try:
            model = inception_v3(**kwargs)
            break
        except (TypeError, ValueError) as exc:
            init_errors.append(str(exc))

    if model is None:
        raise RuntimeError(f"Failed to initialize InceptionV3 for FID: {init_errors}")

    model.fc = torch.nn.Identity()
    model.eval().to(device)
    _INCEPTION_MODEL_CACHE[cache_key] = model
    return model


def _normalize_inception_output(output: object) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError("Unexpected Inception output type.")


def _iter_batch_volume_slices(
    batch_np: np.ndarray,
    max_slices_per_volume: Optional[int] = None,
) -> Iterator[torch.Tensor]:
    if batch_np.ndim == 5:
        volumes = batch_np[:, 0]
    elif batch_np.ndim == 4:
        volumes = batch_np
    else:
        raise ValueError(f"Expected 4D or 5D batch, got shape {batch_np.shape}")

    for vol in volumes:
        vol = np.nan_to_num(vol.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        vol = np.clip(vol, -1.0, 1.0)
        vol = (vol + 1.0) / 2.0

        z_total = vol.shape[-1]
        indices = np.arange(z_total)

        if max_slices_per_volume is not None and max_slices_per_volume > 0 and z_total > max_slices_per_volume:
            indices = np.linspace(0, z_total - 1, num=max_slices_per_volume, dtype=int)

        for z in indices:
            yield torch.from_numpy(vol[:, :, int(z)]).float().unsqueeze(0)


def _extract_features_from_slice_iterator(
    slice_iterator: Iterator[torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    features: list[np.ndarray] = []
    batch: list[torch.Tensor] = []
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)

    def flush_batch() -> None:
        nonlocal batch
        if not batch:
            return

        x = torch.stack(batch, dim=0).to(device)
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - mean) / std

        with torch.no_grad():
            f = _normalize_inception_output(model(x))

        if f.ndim == 4:
            f = torch.flatten(F.adaptive_avg_pool2d(f, output_size=1), start_dim=1)

        features.append(f.detach().cpu().numpy().astype(np.float64))
        batch = []

    for slice_tensor in slice_iterator:
        batch.append(slice_tensor)
        if len(batch) >= batch_size:
            flush_batch()

    flush_batch()

    if not features:
        raise ValueError("No valid slices found for FID feature extraction.")

    return np.concatenate(features, axis=0)


def _compute_gaussian_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {features.shape}")
    if features.shape[0] < 2:
        raise ValueError("Need at least 2 feature vectors to compute covariance for FID.")

    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def _matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    sqrt_diag = np.diag(np.sqrt(eigvals))
    return eigvecs @ sqrt_diag @ eigvecs.T


def _calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError("Mean vectors have different lengths.")
    if sigma1.shape != sigma2.shape:
        raise ValueError("Covariance matrices have different dimensions.")

    diff = mu1 - mu2

    offset = np.eye(sigma1.shape[0], dtype=np.float64) * eps
    sigma1 = sigma1 + offset
    sigma2 = sigma2 + offset

    sqrt_sigma1 = _matrix_sqrt_psd(sigma1)
    middle = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    covmean = _matrix_sqrt_psd(middle)

    fid = float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    if not np.isfinite(fid):
        raise ValueError("FID produced a non-finite value.")
    return max(fid, 0.0)


def _load_fid_stats_cached(path: str) -> np.ndarray:
    abs_path = os.path.abspath(path)
    mtime = os.path.getmtime(abs_path)
    cache_key = f"{abs_path}:{mtime}"

    cached = _FID_STATS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    arr = np.load(abs_path)
    _FID_STATS_CACHE[cache_key] = arr
    return arr


def compute_fid_from_tensors(
    pred_t2: Optional[torch.Tensor] = None,
    gt_t2: Optional[torch.Tensor] = None,
    device: Optional[str] = None,
    batch_size: int = 32,
    max_slices_per_volume: int = 64,
    mu_pred_path: Optional[str] = None,
    sigma_pred_path: Optional[str] = None,
    mu_ref_path: Optional[str] = None,
    sigma_ref_path: Optional[str] = None,
) -> float:
    """
    Compute FID. If mu_pred_path and sigma_pred_path are provided, load predicted stats from .npy files.
    If mu_ref_path and sigma_ref_path are provided, load reference stats from .npy files.
    Otherwise, compute from tensors as before.
    """
    pred_stats_from_file = mu_pred_path is not None and sigma_pred_path is not None
    ref_stats_from_file = mu_ref_path is not None and sigma_ref_path is not None

    need_pred_features = not pred_stats_from_file
    need_ref_features = not ref_stats_from_file

    if need_pred_features and pred_t2 is None:
        raise ValueError("Must provide either pred_t2 tensor or mu_pred/sigma_pred paths.")
    if need_ref_features and gt_t2 is None:
        raise ValueError("Must provide either gt_t2 tensor or mu_ref/sigma_ref paths.")

    model: Optional[torch.nn.Module] = None
    requested_device: Optional[torch.device] = None
    if need_pred_features or need_ref_features:
        requested_device = _resolve_fid_device(device)
        model = _build_inception_feature_extractor(requested_device)

    if pred_stats_from_file:
        if mu_pred_path is None or sigma_pred_path is None:
            raise ValueError("mu_pred_path and sigma_pred_path must both be set.")
        mu_pred = _load_fid_stats_cached(mu_pred_path)
        sigma_pred = _load_fid_stats_cached(sigma_pred_path)
    else:
        if pred_t2 is None or model is None or requested_device is None:
            raise ValueError("Prediction tensor feature extraction was requested without model/device.")
        pred_np = pred_t2.detach().cpu().numpy()
        pred_features = _extract_features_from_slice_iterator(
            _iter_batch_volume_slices(pred_np, max_slices_per_volume=max_slices_per_volume),
            model=cast(torch.nn.Module, model),
            device=cast(torch.device, requested_device),
            batch_size=batch_size,
        )
        mu_pred, sigma_pred = _compute_gaussian_stats(pred_features)

    if ref_stats_from_file:
        if mu_ref_path is None or sigma_ref_path is None:
            raise ValueError("mu_ref_path and sigma_ref_path must both be set.")
        mu_ref = _load_fid_stats_cached(mu_ref_path)
        sigma_ref = _load_fid_stats_cached(sigma_ref_path)
    else:
        if gt_t2 is None or model is None or requested_device is None:
            raise ValueError("Reference tensor feature extraction was requested without model/device.")
        gt_np = gt_t2.detach().cpu().numpy()
        gt_features = _extract_features_from_slice_iterator(
            _iter_batch_volume_slices(gt_np, max_slices_per_volume=max_slices_per_volume),
            model=cast(torch.nn.Module, model),
            device=cast(torch.device, requested_device),
            batch_size=batch_size,
        )
        mu_ref, sigma_ref = _compute_gaussian_stats(gt_features)

    return _calculate_frechet_distance(mu_ref, sigma_ref, mu_pred, sigma_pred)


# =========================
# MAIN evaluation
# =========================
def evaluate_batch(pred_t2: torch.Tensor, gt_t2: Optional[torch.Tensor] = None) -> dict[str, Optional[float]]:
    pred_np = pred_t2.detach().cpu().numpy()

    results: dict[str, Optional[float]] = {
        "psnr": None,
        "ssim": None,
        "fid": None
    }

    if gt_t2 is not None:
        gt_np = gt_t2.detach().cpu().numpy()
        try:
            psnr_val, ssim_val = compute_psnr_ssim(pred_np, gt_np)
            results["psnr"] = float(psnr_val) if np.isfinite(psnr_val) else None
            results["ssim"] = float(ssim_val) if np.isfinite(ssim_val) else None
        except Exception as e:
            print("PSNR/SSIM computation failed:", e)

    # Checkpoints-based FID fast paths (no download/unzip required)
    checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    mu_pred_path = os.path.join(checkpoints_dir, "mu_pred.npy")
    sigma_pred_path = os.path.join(checkpoints_dir, "sigma_pred.npy")
    mu_ref_path = os.path.join(checkpoints_dir, "mu_ref.npy")
    sigma_ref_path = os.path.join(checkpoints_dir, "sigma_ref.npy")

    has_pred_stats = os.path.isfile(mu_pred_path) and os.path.isfile(sigma_pred_path)
    has_ref_stats = os.path.isfile(mu_ref_path) and os.path.isfile(sigma_ref_path)

    # Backward compatibility: if only mu_pred/sigma_pred exist, treat them as fixed reference stats.
    alias_ref_mu = mu_ref_path if has_ref_stats else (mu_pred_path if has_pred_stats else None)
    alias_ref_sigma = sigma_ref_path if has_ref_stats else (sigma_pred_path if has_pred_stats else None)

    try:
        fid_val: Optional[float]
        if has_ref_stats and has_pred_stats:
            # Fastest path: both distributions are precomputed in checkpoints.
            fid_val = compute_fid_from_tensors(
                pred_t2=None,
                mu_pred_path=mu_pred_path,
                sigma_pred_path=sigma_pred_path,
                mu_ref_path=mu_ref_path,
                sigma_ref_path=sigma_ref_path,
                device=None,
                batch_size=FID_RUNTIME_BATCH_SIZE,
                max_slices_per_volume=FID_RUNTIME_MAX_SLICES,
            )
        elif alias_ref_mu is not None and alias_ref_sigma is not None:
            # Fast path: current prediction tensor vs fixed reference stats.
            fid_val = compute_fid_from_tensors(
                pred_t2=pred_t2,
                gt_t2=None,
                mu_ref_path=alias_ref_mu,
                sigma_ref_path=alias_ref_sigma,
                device=None,
                batch_size=FID_RUNTIME_BATCH_SIZE,
                max_slices_per_volume=FID_RUNTIME_MAX_SLICES,
            )
        elif gt_t2 is not None:
            # Slow fallback: tensor-vs-tensor FID (still local, no network download).
            fid_val = compute_fid_from_tensors(
                pred_t2=pred_t2,
                gt_t2=gt_t2,
                device=None,
                batch_size=FID_RUNTIME_BATCH_SIZE,
                max_slices_per_volume=FID_RUNTIME_MAX_SLICES,
            )
        else:
            fid_val = None

        if fid_val is not None and np.isfinite(fid_val):
            results["fid"] = float(fid_val)
        else:
            results["fid"] = None
    except Exception as e:
        print("FID computation failed:", e)

    return results