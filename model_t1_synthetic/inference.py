from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import math
import zipfile

import nibabel as nib
from nibabel.loadsave import save as nib_save
from nibabel.nifti1 import Nifti1Image
import numpy as np
import torch

from model_t1_synthetic.config import Config
from model_t1_synthetic.models.autoencoder import load_autoencoder
from model_t1_synthetic.models.diffusion_unet import load_latent_diffusion_unet, build_scheduler
from model_t1_t2.postprocess import compute_fid_from_tensors


def _load_generated_batch_from_paths(pred_paths: list[str]) -> torch.Tensor:
    volumes: list[np.ndarray] = []
    for pred_path in pred_paths:
        volume = nib.load(str(pred_path)).get_fdata(dtype=np.float32)
        if volume.ndim == 4:
            volume = volume[..., 0]
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape} from {pred_path}")
        volumes.append(volume)

    if not volumes:
        raise ValueError("pred_paths must contain at least one generated NIfTI volume.")

    batch = np.stack(volumes, axis=0)[:, np.newaxis, ...]
    return torch.from_numpy(batch).float()


def compute_fid_score(
    pred_paths: list[str],
    gdrive_file_id: str = "",
) -> Optional[float]:
    """
    Compute FID score for generated T1 images using precomputed checkpoint stats.

    Args:
        pred_paths: List of paths to generated T1 NIfTI files
        gdrive_file_id: Deprecated. Kept for backward compatibility and ignored.

    Returns:
        FID score (lower is better), or None if computation fails
    """
    _ = gdrive_file_id
    try:
        checkpoints_dir = Config.CHECKPOINTS_DIR
        mu_ref_path = checkpoints_dir / "mu_ref.npy"
        sigma_ref_path = checkpoints_dir / "sigma_ref.npy"

        # Backward compatibility: if only mu_pred/sigma_pred exist, treat them as reference stats.
        if not mu_ref_path.exists() or not sigma_ref_path.exists():
            mu_ref_path = checkpoints_dir / "mu_pred.npy"
            sigma_ref_path = checkpoints_dir / "sigma_pred.npy"

        if not mu_ref_path.exists() or not sigma_ref_path.exists():
            print(
                "FID computation skipped: missing reference stats at "
                f"{mu_ref_path} and {sigma_ref_path}"
            )
            return None

        pred_batch = _load_generated_batch_from_paths(pred_paths)
        fid = compute_fid_from_tensors(
            pred_t2=pred_batch,
            gt_t2=None,
            mu_ref_path=str(mu_ref_path),
            sigma_ref_path=str(sigma_ref_path),
            device=Config.DEVICE,
            batch_size=Config.FID_GEN_BATCH,
            max_slices_per_volume=64,
        )

        return float(fid)
    except Exception as e:
        print(f"FID computation error: {e}")
        return None


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
            in_channels=Config.DIFFUSION_IN_CHANNELS,
            out_channels=Config.DIFFUSION_OUT_CHANNELS,
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

    def compute_fid_with_reference(
        self,
        generated_path: str,
        gt_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute FID between generated and reference data.
        
        Args:
            generated_path: Path to generated NIfTI file or directory
            gt_tensor: Optional ground truth tensor. If provided, computes proper FID.
        
        Returns:
            FID score (lower is better)
        """
        from model_t1_t2.postprocess import compute_fid_from_tensors
        
        # Load generated data
        gen_path = Path(generated_path)
        if gen_path.suffix in {".nii", ".gz"}:
            gen_img = nib.load(str(gen_path))
            gen_data = gen_img.get_fdata(dtype=np.float32)
            gen_tensor = torch.from_numpy(gen_data).float().unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported file type: {gen_path.suffix}")
        
        if gt_tensor is None:
            return float(compute_fid_score([str(gen_path)]) or 0.0)
        
        # Compute proper FID
        return compute_fid_from_tensors(
            pred_t2=gen_tensor,
            gt_t2=gt_tensor,
            device=self.device,
            batch_size=16,
            max_slices_per_volume=64,
        )

class SyntheticT1BatchResult(dict):
    pass


def zip_generated_files(file_paths: list[str], zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in file_paths:
            path = Path(file_path)
            archive.write(path, arcname=path.name)
    return zip_path
