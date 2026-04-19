import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import urlretrieve
from typing import Iterator, Optional

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_INCEPTION_MODEL_CACHE: dict[str, torch.nn.Module] = {}


def _is_nifti_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _extract_gdrive_file_id(gdrive_link: str) -> str:
    parsed = urlparse(gdrive_link)

    if "drive.google.com" not in parsed.netloc:
        raise ValueError("Expected a Google Drive link.")

    if "/file/d/" in parsed.path:
        # Format: /file/d/<FILE_ID>/view
        return parsed.path.split("/file/d/")[1].split("/")[0]

    query = parse_qs(parsed.query)
    file_ids = query.get("id", [])
    if file_ids:
        return file_ids[0]

    raise ValueError("Could not parse Google Drive file ID from link.")


def _download_gdrive_zip(gdrive_link: str, dst_zip: Path) -> Path:
    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    if dst_zip.exists():
        return dst_zip

    file_id = _extract_gdrive_file_id(gdrive_link)
    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        import gdown

        gdown.download(url=direct_url, output=str(dst_zip), quiet=False)
    except ImportError:
        # Fallback for smaller public files when gdown is not available.
        urlretrieve(direct_url, str(dst_zip))

    if not dst_zip.exists() or dst_zip.stat().st_size == 0:
        raise RuntimeError(f"Failed to download ZIP from Google Drive: {gdrive_link}")

    return dst_zip


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.nan_to_num(volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if volume.size == 0:
        return volume

    lo = float(np.percentile(volume, 1.0))
    hi = float(np.percentile(volume, 99.0))
    if hi > lo:
        volume = np.clip(volume, lo, hi)
        volume = (volume - lo) / (hi - lo)
    else:
        volume = np.zeros_like(volume, dtype=np.float32)
    return volume


def _iter_nifti_slices(path: Path, max_slices_per_volume: Optional[int] = None) -> Iterator[torch.Tensor]:
    volume = nib.load(str(path)).get_fdata(dtype=np.float32)
    if volume.ndim == 4:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, but got shape {volume.shape}")

    volume = _normalize_volume(volume)
    z_total = volume.shape[-1]
    indices = np.arange(z_total)

    if max_slices_per_volume is not None and max_slices_per_volume > 0 and z_total > max_slices_per_volume:
        indices = np.linspace(0, z_total - 1, num=max_slices_per_volume, dtype=int)

    for z in indices:
        slice_2d = volume[:, :, int(z)]
        yield torch.from_numpy(slice_2d).float().unsqueeze(0)


def _iter_image_slices(path: Path) -> Iterator[torch.Tensor]:
    with Image.open(path) as img:
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float32)

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    yield torch.from_numpy(arr).float().unsqueeze(0)


def _iter_input_slices(input_path: Path, max_slices_per_volume: Optional[int] = None) -> Iterator[torch.Tensor]:
    if input_path.is_file():
        if _is_nifti_file(input_path):
            yield from _iter_nifti_slices(input_path, max_slices_per_volume=max_slices_per_volume)
            return
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield from _iter_image_slices(input_path)
            return
        raise ValueError(f"Unsupported input file type: {input_path}")

    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    for file_path in sorted(input_path.rglob("*")):
        if not file_path.is_file():
            continue
        if _is_nifti_file(file_path):
            yield from _iter_nifti_slices(file_path, max_slices_per_volume=max_slices_per_volume)
        elif file_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield from _iter_image_slices(file_path)


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
    last_exc = None

    try:
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    except (TypeError, ValueError) as exc:
        last_exc = exc

    if model is None:
        try:
            model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        except (TypeError, ValueError) as exc:
            last_exc = exc

    if model is None:
        raise RuntimeError(f"Failed to initialize InceptionV3 for FID: {last_exc}")

    model.fc = torch.nn.Identity()
    model.eval().to(device)
    _INCEPTION_MODEL_CACHE[cache_key] = model
    return model


def _normalize_inception_output(output: torch.Tensor) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "logits"):
        return output.logits
    return output


def _extract_features(
    input_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    max_slices_per_volume: Optional[int] = None,
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

    for slice_tensor in _iter_input_slices(input_path, max_slices_per_volume=max_slices_per_volume):
        batch.append(slice_tensor)
        if len(batch) >= batch_size:
            flush_batch()

    flush_batch()

    if not features:
        raise ValueError(f"No valid slices found for input: {input_path}")

    return np.concatenate(features, axis=0)


def _iter_batch_volume_slices(
    batch_np: np.ndarray,
    max_slices_per_volume: Optional[int] = None,
) -> Iterator[torch.Tensor]:
    if batch_np.ndim == 5:
        # [B, C, H, W, D] -> use channel 0
        volumes = batch_np[:, 0]
    elif batch_np.ndim == 4:
        # [B, H, W, D]
        volumes = batch_np
    else:
        raise ValueError(f"Expected 4D/5D batch, got shape {batch_np.shape}")

    for vol in volumes:
        vol = np.nan_to_num(vol.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        vol = np.clip(vol, -1.0, 1.0)
        vol = (vol + 1.0) / 2.0

        z_total = vol.shape[-1]
        indices = np.arange(z_total)

        if max_slices_per_volume is not None and max_slices_per_volume > 0 and z_total > max_slices_per_volume:
            indices = np.linspace(0, z_total - 1, num=max_slices_per_volume, dtype=int)

        for z in indices:
            slice_2d = vol[:, :, int(z)]
            yield torch.from_numpy(slice_2d).float().unsqueeze(0)


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


def compute_fid_from_tensors(
    pred_t2: torch.Tensor,
    gt_t2: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 32,
    max_slices_per_volume: Optional[int] = 64,
) -> float:
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    if requested_device.type == "mps":
        requested_device = torch.device("cpu")

    pred_np = pred_t2.detach().cpu().numpy()
    gt_np = gt_t2.detach().cpu().numpy()

    model = _build_inception_feature_extractor(requested_device)

    gt_features = _extract_features_from_slice_iterator(
        _iter_batch_volume_slices(gt_np, max_slices_per_volume=max_slices_per_volume),
        model=model,
        device=requested_device,
        batch_size=batch_size,
    )
    pred_features = _extract_features_from_slice_iterator(
        _iter_batch_volume_slices(pred_np, max_slices_per_volume=max_slices_per_volume),
        model=model,
        device=requested_device,
        batch_size=batch_size,
    )

    mu_ref, sigma_ref = _compute_gaussian_stats(gt_features)
    mu_pred, sigma_pred = _compute_gaussian_stats(pred_features)
    return _calculate_frechet_distance(mu_ref, sigma_ref, mu_pred, sigma_pred)


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

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(max(fid, 0.0))


def _extract_t2_from_zip(zip_path: Path, out_dir: Path) -> Path:
    extract_root = out_dir / "t2_extracted"
    extract_root.mkdir(parents=True, exist_ok=True)

    selected_files = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            member_parts = Path(member.filename.replace("\\", "/")).parts
            lower_parts = [p.lower() for p in member_parts]
            if "t2" not in lower_parts:
                continue

            t2_idx = lower_parts.index("t2")
            relative_after_t2 = Path(*member_parts[t2_idx + 1 :]) if t2_idx + 1 < len(member_parts) else None
            if relative_after_t2 is None or str(relative_after_t2) in {"", "."}:
                continue

            target_path = extract_root / relative_after_t2
            target_path.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(member, "r") as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            selected_files += 1

    if selected_files == 0:
        raise ValueError("No T2 files found in ZIP. Check folder naming in the archive.")

    return extract_root


def compute_fid_from_gdrive_t2(
    gdrive_zip_link: str,
    generated_input: str,
    device: str = "cuda",
    batch_size: int = 32,
    dims: int = 2048,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    max_slices_per_volume: Optional[int] = 64,
) -> float:
    """
    Compute FID where the reference set is T2 data inside a Google Drive ZIP.

    Args:
        gdrive_zip_link: Google Drive share link to a ZIP containing at least a T2 folder.
        generated_input: Path to user output (NIfTI file or folder of NIfTI/images).
        device: 'cuda', 'cpu', etc.
        batch_size: Batch size for FID feature extraction.
        dims: Reserved for compatibility; inception features are fixed to 2048.
        num_workers: Reserved for compatibility.
        cache_dir: Optional persistent directory for downloaded/extracted data.
        max_slices_per_volume: Optional cap on slices sampled from each volume.

    Returns:
        FID score as float.
    """
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")

    if requested_device.type == "mps":
        requested_device = torch.device("cpu")

    if dims != 2048:
        raise ValueError("This in-memory implementation currently supports dims=2048 only.")

    _ = num_workers

    use_temp_dir = cache_dir is None
    work_dir = Path(tempfile.mkdtemp(prefix="fid_eval_")) if use_temp_dir else Path(cache_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        zip_path = work_dir / "dataset.zip"
        _download_gdrive_zip(gdrive_zip_link, zip_path)

        ref_dir = _extract_t2_from_zip(zip_path, work_dir)
        gen_path = Path(generated_input)

        model = _build_inception_feature_extractor(requested_device)
        ref_features = _extract_features(
            input_path=ref_dir,
            model=model,
            device=requested_device,
            batch_size=batch_size,
            max_slices_per_volume=max_slices_per_volume,
        )
        gen_features = _extract_features(
            input_path=gen_path,
            model=model,
            device=requested_device,
            batch_size=batch_size,
            max_slices_per_volume=max_slices_per_volume,
        )

        mu_ref, sigma_ref = _compute_gaussian_stats(ref_features)
        mu_gen, sigma_gen = _compute_gaussian_stats(gen_features)
        return _calculate_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)
    finally:
        if use_temp_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

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
    print(B)
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
# MAIN evaluation
# =========================
def evaluate_batch(pred_t2: torch.Tensor, gt_t2: Optional[torch.Tensor] = None) -> dict[str, Optional[float]]:
    pred_np = pred_t2.detach().cpu().numpy()

    results: dict[str, Optional[float]] = {
        "psnr": None,
        "ssim": None,
        "fid": None
    }

    if gt_t2 is None:
        return results

    gt_np = gt_t2.detach().cpu().numpy()

    try:
        psnr_val, ssim_val = compute_psnr_ssim(pred_np, gt_np)

        results["psnr"] = float(psnr_val) if np.isfinite(psnr_val) else None
        results["ssim"] = float(ssim_val) if np.isfinite(ssim_val) else None

    except Exception as e:
        print("PSNR/SSIM computation failed:", e)

    try:
        fid_val = compute_fid_from_tensors(
            pred_t2=pred_t2,
            gt_t2=gt_t2,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=16,
            max_slices_per_volume=64,
        )
        results["fid"] = float(fid_val) if np.isfinite(fid_val) else None

    except Exception as e:
        print("FID computation failed:", e)

    return results