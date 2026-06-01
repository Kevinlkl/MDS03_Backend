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
from model_t1_synthetic.models.diffusion_unet import (
    load_latent_diffusion_unet,
    build_scheduler,
)


def save_nifti_volume(
    volume: torch.Tensor,
    output_path: str,
    affine: Optional[np.ndarray] = None,
) -> str:
    """
    Function description:
        Save a generated synthetic T1 tensor volume as a NIfTI file.

    Parameters:
        volume (torch.Tensor): Generated tensor volume.
        output_path (str): Destination NIfTI path.
        affine (np.ndarray | None): Optional affine matrix for the NIfTI image.

    Returns:
        str: Destination path where the NIfTI file was saved.
    """
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


def get_precomputed_dataset_metrics() -> dict[str, Any]:
    """
    Function description:
        Return precomputed dataset-level metrics for synthetic T1 generation.

    Parameters:
        None

    Returns:
        dict[str, Any]: FID, KID, and metric note values.
    """
    return {
        "dataset_fid": float(Config.PRECOMPUTED_FID),
        "dataset_kid_mean": float(Config.PRECOMPUTED_KID_MEAN),
        "dataset_kid_std": float(Config.PRECOMPUTED_KID_STD),
        "metrics_note": "FID and KID are precomputed using the full evaluation dataset.",
    }


class SyntheticT1GenerationPipeline:
    """
    Class description:
        End-to-end unconditional synthetic T1 generation pipeline.

    Attributes:
        device (str): Torch device used for generation.
        autoencoder (torch.nn.Module): Trained autoencoder used to decode generated latents.
        unet (torch.nn.Module): Trained latent diffusion denoising model.
        scale_factor (float): Latent scaling value saved with the checkpoint.
        latent_channels (int): Number of channels in the latent representation.
        scheduler (DDPMScheduler): Noise scheduler used during reverse diffusion.
        latent_spatial_shape (tuple[int, ...]): Spatial shape used for latent noise sampling.
    """

    def __init__(self) -> None:
        """
        Function description:
            Load trained synthetic T1 models, scheduler, and latent shape settings.

        Parameters:
            None

        Returns:
            None
        """
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

        self.scale_factor = float(
            self.diffusion_meta.get("scale_factor", 1.0)
        )

        self.latent_channels = int(
            self.diffusion_meta.get(
                "latent_channels",
                Config.LATENT_CHANNELS,
            )
        )

        self.scheduler = build_scheduler(
            num_train_timesteps=Config.NUM_TRAIN_TIMESTEPS,
            beta_start=Config.BETA_START,
            beta_end=Config.BETA_END,
        )

        self.latent_spatial_shape = self._infer_latent_spatial_shape()

    def _infer_latent_spatial_shape(self) -> tuple[int, ...]:
        """
        Function description:
            Infer the latent spatial shape expected by the autoencoder.

        Parameters:
            None

        Returns:
            tuple[int, ...]: Latent spatial shape used when sampling noise.
        """
        dummy = torch.zeros(
            (1, 1, *Config.SPATIAL_SIZE),
            device=self.device,
        )

        try:
            latent = self.autoencoder.encode_stage_2_inputs(dummy)
            return tuple(int(dim) for dim in latent.shape[2:])
        except Exception:
            return tuple(
                max(1, int(math.ceil(size / 8)))
                for size in Config.SPATIAL_SIZE
            )

    def _resolve_output_path(
        self,
        index: int,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Function description:
            Resolve the destination path for one generated synthetic T1 volume.

        Parameters:
            index (int): One-based generated sample index.
            output_dir (Path | None): Optional output directory override.

        Returns:
            Path: Destination path for the generated NIfTI file.
        """
        base_dir = Path(output_dir) if output_dir is not None else Config.GENERATED_DIR
        base_dir.mkdir(parents=True, exist_ok=True)

        return base_dir / f"synthetic_t1_{index:04d}.nii.gz"

    @torch.no_grad()
    def generate_tensor(
        self,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Function description:
            Generate one synthetic T1 tensor by denoising sampled latent noise.

        Parameters:
            num_inference_steps (int | None): Optional number of denoising steps.
            seed (int | None): Optional seed for deterministic noise sampling.

        Returns:
            torch.Tensor: Generated synthetic T1 tensor.
        """
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
                timestep_tensor = torch.tensor(
                    [timestep],
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                timestep_tensor = timestep.reshape(1).to(self.device).long()

            noise_pred = self.unet(
                latent,
                timesteps=timestep_tensor,
            )

            if hasattr(noise_pred, "sample"):
                noise_pred = noise_pred.sample

            step_output = self.scheduler.step(
                noise_pred,
                timestep,
                latent,
            )

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
        """
        Function description:
            Generate multiple synthetic T1 volumes and save each one to disk.

        Parameters:
            num_samples (int): Number of synthetic T1 volumes to generate.
            num_inference_steps (int | None): Optional number of denoising steps per sample.
            seed (int | None): Optional base seed incremented for each sample.
            output_dir (Path | None): Optional directory where files should be saved.

        Returns:
            list[dict[str, Any]]: Generated sample metadata, tensors, paths, and metrics.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        results: list[dict[str, Any]] = []

        for index in range(num_samples):
            sample_seed = None if seed is None else int(seed) + index

            sample_tensor = self.generate_tensor(
                num_inference_steps=num_inference_steps,
                seed=sample_seed,
            )

            output_path = self._resolve_output_path(
                index + 1,
                output_dir=output_dir,
            )

            save_nifti_volume(
                sample_tensor,
                str(output_path),
            )

            results.append(
                {
                    "index": index + 1,
                    "seed": sample_seed,
                    "output_path": str(output_path),
                    "tensor": sample_tensor,
                    "metrics": get_precomputed_dataset_metrics(),
                }
            )

        return results


class SyntheticT1BatchResult(dict):
    """
    Class description:
        Dictionary-like container reserved for synthetic T1 batch results.

    Attributes:
        None
    """

    pass


def zip_generated_files(
    file_paths: list[str],
    zip_path: Path,
) -> Path:
    """
    Function description:
        Package generated NIfTI files into a ZIP archive.

    Parameters:
        file_paths (list[str]): Paths to generated files that should be archived.
        zip_path (Path): Destination ZIP archive path.

    Returns:
        Path: Path to the created ZIP archive.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for file_path in file_paths:
            path = Path(file_path)
            archive.write(path, arcname=path.name)

    return zip_path
