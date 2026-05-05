from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import math
import zipfile

from nibabel.loadsave import save as nib_save
from nibabel.nifti1 import Nifti1Image
import numpy as np
import torch
import torch.nn.functional as F

from model_t1_synthetic.config import Config
from model_t1_synthetic.models.autoencoder import load_autoencoder
from model_t1_synthetic.models.diffusion_unet import load_latent_diffusion_unet, build_scheduler


_INCEPTION_MODEL_CACHE: dict[str, torch.nn.Module] = {}


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
) -> list[torch.Tensor]:
    slices = []
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
            slice_2d = vol[:, :, int(z)]
            slices.append(torch.from_numpy(slice_2d).float().unsqueeze(0))

    return slices


def _extract_features_from_slice_iterator(
    slice_iterator: list[torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    features: list[np.ndarray] = []
    batch: list[torch.Tensor] = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

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


@torch.no_grad()
def compute_batch_fid(
    generated_tensors: list[torch.Tensor],
    device: str = "cuda",
    batch_size: int = 32,
    max_slices_per_volume: int = 64,
) -> Optional[float]:
    """
    Compute FID for a generated synthetic T1 batch against
    precomputed real T1 Inception feature statistics.
    """

    if len(generated_tensors) < 2:
        print("FID skipped: at least 2 generated samples are required.")
        return None

    requested_device = torch.device(device)

    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")

    try:
        real_mu = np.load(Config.REAL_T1_MU_PATH)
        real_sigma = np.load(Config.REAL_T1_SIGMA_PATH)
    except Exception as e:
        print(f"Failed to load real T1 stats: {e}")
        return None

    try:
        clean_tensors = []

        for tensor in generated_tensors:
            tensor = tensor.detach().cpu().float()

            # Expected original: [1, 1, D, H, W]
            # Convert to: [1, D, H, W]
            if tensor.ndim == 5:
                tensor = tensor[0]

            # If already [1, D, H, W], keep it
            if tensor.ndim != 4:
                print(f"Unexpected generated tensor shape for FID: {tuple(tensor.shape)}")
                return None

            clean_tensors.append(tensor)

        batch_np = torch.stack(clean_tensors, dim=0).numpy()
        # Final shape: [N, 1, D, H, W] or [N, 1, H, W, D]

    except Exception as e:
        print(f"Failed to prepare generated tensors for FID: {e}")
        return None

    try:
        slices = _iter_batch_volume_slices(
            batch_np,
            max_slices_per_volume=max_slices_per_volume,
        )

        model = _build_inception_feature_extractor(requested_device)

        features = _extract_features_from_slice_iterator(
            slices,
            model,
            requested_device,
            batch_size,
        )

        fake_mu, fake_sigma = _compute_gaussian_stats(features)

        fid = _calculate_frechet_distance(
            real_mu,
            real_sigma,
            fake_mu,
            fake_sigma,
        )

        return float(fid)

    except Exception as e:
        print(f"FID computation failed: {e}")
        return None

def _polynomial_kernel(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef0: float = 1.0,
) -> np.ndarray:
    if gamma is None:
        gamma = 1.0 / x.shape[1]

    return (gamma * (x @ y.T) + coef0) ** degree


def _compute_kid_from_features(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    subset_size: int = 50,
    n_subsets: int = 20,
) -> tuple[float, float]:
    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)

    if real_features.ndim != 2 or fake_features.ndim != 2:
        raise ValueError("KID features must be 2D arrays.")

    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError("Real and fake feature dimensions do not match.")

    if len(real_features) < 2 or len(fake_features) < 2:
        raise ValueError("Need at least 2 real and 2 fake feature vectors for KID.")

    subset_size = min(subset_size, len(real_features), len(fake_features))

    kid_scores = []

    for _ in range(n_subsets):
        real_idx = np.random.choice(len(real_features), subset_size, replace=False)
        fake_idx = np.random.choice(len(fake_features), subset_size, replace=False)

        real_subset = real_features[real_idx]
        fake_subset = fake_features[fake_idx]

        k_rr = _polynomial_kernel(real_subset, real_subset)
        k_ff = _polynomial_kernel(fake_subset, fake_subset)
        k_rf = _polynomial_kernel(real_subset, fake_subset)

        m = subset_size

        mmd = (
            (k_rr.sum() - np.trace(k_rr)) / (m * (m - 1))
            + (k_ff.sum() - np.trace(k_ff)) / (m * (m - 1))
            - 2.0 * k_rf.mean()
        )

        kid_scores.append(mmd)

    return float(np.mean(kid_scores)), float(np.std(kid_scores))


@torch.no_grad()
def compute_batch_kid(
    generated_tensors: list[torch.Tensor],
    device: str = "cuda",
    batch_size: int = 32,
    max_slices_per_volume: int = 64,
    subset_size: int = 50,
    n_subsets: int = 20,
) -> tuple[Optional[float], Optional[float]]:
    """
    Compute KID for a generated synthetic T1 batch against
    precomputed real T1 Inception features.

    Requires:
        Config.REAL_T1_FEATURES_PATH
    """

    if len(generated_tensors) < 2:
        print("KID skipped: at least 2 generated samples are required.")
        return None, None

    requested_device = torch.device(device)

    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")

    try:
        real_features = np.load(Config.REAL_T1_FEATURES_PATH)
    except Exception as e:
        print(f"Failed to load real T1 features: {e}")
        return None, None

    try:
        clean_tensors = []

        for tensor in generated_tensors:
            tensor = tensor.detach().cpu().float()

            # Expected original: [1, 1, D, H, W]
            # Convert to: [1, D, H, W]
            if tensor.ndim == 5:
                tensor = tensor[0]

            if tensor.ndim != 4:
                print(f"Unexpected generated tensor shape for KID: {tuple(tensor.shape)}")
                return None, None

            clean_tensors.append(tensor)

        batch_np = torch.stack(clean_tensors, dim=0).numpy()

    except Exception as e:
        print(f"Failed to prepare generated tensors for KID: {e}")
        return None, None

    try:
        slices = _iter_batch_volume_slices(
            batch_np,
            max_slices_per_volume=max_slices_per_volume,
        )

        model = _build_inception_feature_extractor(requested_device)

        fake_features = _extract_features_from_slice_iterator(
            slices,
            model,
            requested_device,
            batch_size,
        )

        kid_mean, kid_std = _compute_kid_from_features(
            real_features=real_features,
            fake_features=fake_features,
            subset_size=subset_size,
            n_subsets=n_subsets,
        )

        return float(kid_mean), float(kid_std)

    except Exception as e:
        print(f"KID computation failed: {e}")
        return None, None
    
def save_nifti_volume(volume: torch.Tensor, output_path: str, affine: Optional[np.ndarray] = None) -> str:
    arr = volume.detach().cpu().float().numpy()

    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape}")

    if affine is None:
        affine = np.eye(4)

    nii = Nifti1Image(arr, affine)
    nib_save(nii, output_path)
    return output_path


class SyntheticT1GenerationPipeline:
    def __init__(self) -> None:
        self.device = Config.DEVICE

        self.autoencoder = load_autoencoder(
            checkpoint_path=Config.AUTOENCODER_CKPT,
            latent_channels=Config.LATENT_CHANNELS,
            device=self.device,
        )
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.unet, self.diffusion_meta = load_latent_diffusion_unet(
        checkpoint_path=Config.LATENT_DIFFUSION_CKPT,
        device=self.device,
    )
        self.unet.eval()
        for param in self.unet.parameters():
            param.requires_grad = False

        self.scale_factor = float(self.diffusion_meta.get("scale_factor", 1.0))
        self.latent_channels = int(self.diffusion_meta.get("latent_channels", Config.LATENT_CHANNELS))
        self.scheduler = build_scheduler(
            num_train_timesteps=Config.NUM_TRAIN_TIMESTEPS,
            beta_start=Config.BETA_START,
            beta_end=Config.BETA_END,
        )
        self.latent_spatial_shape = self._infer_latent_spatial_shape()

    def _infer_latent_spatial_shape(self) -> tuple[int, ...]:
        dummy = torch.zeros((1, 1, *Config.SPATIAL_SIZE), device=self.device)
        try:
            latent = self.autoencoder.encode_stage_2_inputs(dummy)
            return tuple(int(dim) for dim in latent.shape[2:])
        except Exception:
            return tuple(max(1, int(math.ceil(size / 8))) for size in Config.SPATIAL_SIZE)

    def _resolve_output_path(self, index: int, output_dir: Optional[Path] = None) -> Path:
        base_dir = Path(output_dir) if output_dir is not None else Config.GENERATED_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / f"synthetic_t1_{index:04d}.nii.gz"

    @torch.no_grad()
    def generate_tensor(
        self,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        steps = int(num_inference_steps or Config.NUM_INFERENCE_STEPS)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))

        latent = torch.randn(
            (1, self.latent_channels, *self.latent_spatial_shape),
            device=self.device,
            generator=generator,
        )

        self.scheduler.set_timesteps(steps)

        for timestep in self.scheduler.timesteps:
            if not torch.is_tensor(timestep):
                timestep_tensor = torch.tensor([timestep], device=self.device, dtype=torch.long)
            else:
                timestep_tensor = timestep.reshape(1).to(self.device).long()

            noise_pred = self.unet(latent, timesteps=timestep_tensor)
            if hasattr(noise_pred, "sample"):
                noise_pred = noise_pred.sample

            step_output = self.scheduler.step(noise_pred, timestep, latent)
            if hasattr(step_output, "prev_sample"):
                latent = step_output.prev_sample
            elif isinstance(step_output, tuple):
                latent = step_output[0]
            else:
                latent = step_output

        if self.scale_factor != 0:
            latent = latent / self.scale_factor

        pred_t1 = self.autoencoder.decode_stage_2_outputs(latent)
        return pred_t1

    @torch.no_grad()
    def generate_many(
        self,
        num_samples: int,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ) -> list[dict[str, Any]]:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        results: list[dict[str, Any]] = []
        for index in range(num_samples):
            sample_seed = None if seed is None else int(seed) + index
            sample_tensor = self.generate_tensor(
                num_inference_steps=num_inference_steps,
                seed=sample_seed,
            )
            output_path = self._resolve_output_path(index + 1, output_dir=output_dir)
            save_nifti_volume(sample_tensor, str(output_path))
            results.append(
                {
                    "index": index + 1,
                    "seed": sample_seed,
                    "output_path": str(output_path),
                    "tensor": sample_tensor,
                }
            )

        return results


class SyntheticT1BatchResult(dict):
    pass


def zip_generated_files(file_paths: list[str], zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in file_paths:
            path = Path(file_path)
            archive.write(path, arcname=path.name)
    return zip_path
