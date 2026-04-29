from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import math
import zipfile

from nibabel.loadsave import save as nib_save
from nibabel.nifti1 import Nifti1Image
import numpy as np
import torch

from model_t1_synthetic.config import Config
from model_t1_synthetic.models.autoencoder import load_autoencoder
from model_t1_synthetic.models.diffusion_unet import load_latent_diffusion_unet, build_scheduler


def compute_fid_score(tensor: torch.Tensor) -> Optional[float]:
	"""
	Compute FID-like quality score based on tensor statistics.
	FID (Fréchet Inception Distance) is approximated using simple image statistics.
	Lower is better (range 0-200).
	
	For a proper FID, use the full FID computation with InceptionV3 features.
	This is a simplified approximation.
	"""
	try:
		arr = tensor.detach().cpu().float().numpy()
		
		# Compute basic statistics as FID approximation
		# In practice, you'd use actual Inception features
		mean = np.mean(arr)
		std = np.std(arr)
		
		# Heuristic FID approximation (0-200 scale)
		# Ideal: mean ~0, std ~1 for normalized medical images
		fid_approx = abs(mean) * 50 + abs(std - 1) * 50
		fid_score = min(200, max(0, float(fid_approx)))
		
		return fid_score
	except Exception:
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


class SyntheticT1BatchResult(dict):
    pass


def zip_generated_files(file_paths: list[str], zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in file_paths:
            path = Path(file_path)
            archive.write(path, arcname=path.name)
    return zip_path
