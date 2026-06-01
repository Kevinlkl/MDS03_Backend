from pathlib import Path
from typing import Optional

import torch

from model_t1_flair.config import Config
from model_t1_flair.preprocess import MRIProcessor
from model_t1_t2.postprocess import save_nifti, evaluate_batch
from model_t1_flair.models.autoencoder import load_autoencoder
from model_t1_flair.models.diffusion_unet import (
    load_latent_diffusion_unet,
    build_scheduler,
)


class InferencePipeline:
    """End-to-end T1-to-FLAIR inference pipeline with preprocessing and metrics."""

    def __init__(self):
        """Load preprocessing transforms, trained models, and the noise scheduler."""
        self.device = Config.DEVICE

        self.processor = MRIProcessor(
            spatial_size=Config.SPATIAL_SIZE,
            pixdim=Config.PIXDIM,
            intensity_lower=Config.INTENSITY_LOWER,
            intensity_upper=Config.INTENSITY_UPPER,
            b_min=Config.B_MIN,
            b_max=Config.B_MAX,
        )

        self.autoencoder = load_autoencoder(
            checkpoint_path=Config.AUTOENCODER_CKPT,
            latent_channels=Config.LATENT_CHANNELS,
            device=self.device,
        )
        self.autoencoder.eval()
        for p in self.autoencoder.parameters():
            p.requires_grad = False

        self.unet, self.diffusion_meta = load_latent_diffusion_unet(
            checkpoint_path=Config.LATENT_DIFFUSION_CKPT,
            in_channels=Config.DIFFUSION_IN_CHANNELS,
            out_channels=Config.DIFFUSION_OUT_CHANNELS,
            device=self.device,
        )
        self.unet.eval()

        self.scale_factor = self.diffusion_meta.get("scale_factor", 1.0)

        self.scheduler = build_scheduler(
            num_train_timesteps=Config.NUM_TRAIN_TIMESTEPS,
            beta_start=Config.BETA_START,
            beta_end=Config.BETA_END,
        )

    @staticmethod
    def _make_output_path(input_path: str, output_path: Optional[str]) -> Path:
        """Resolve the output NIfTI path for a generated FLAIR volume."""
        if output_path is not None:
            return Path(output_path)

        p = Path(input_path)
        if p.name.endswith(".nii.gz"):
            input_name = p.name[:-7]
            return Config.OUTPUTS_DIR / f"{input_name}_pred_flair.nii.gz"
        return Config.OUTPUTS_DIR / f"{p.stem}_pred_flair.nii"

    @torch.no_grad()
    def encode_condition(self, t1: torch.Tensor) -> torch.Tensor:
        """Encode the input T1 tensor into the scaled latent conditioning space."""
        z_t1 = self.autoencoder.encode_stage_2_inputs(t1)

        # Checkpoints may store the latent scale as a Python float or tensor.
        if isinstance(self.scale_factor, torch.Tensor):
            scale_factor = self.scale_factor.to(z_t1.device, dtype=z_t1.dtype)
        else:
            scale_factor = torch.tensor(
                self.scale_factor,
                device=z_t1.device,
                dtype=z_t1.dtype,
            )

        return z_t1 * scale_factor

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a predicted latent tensor back into image space."""
        if isinstance(self.scale_factor, torch.Tensor):
            scale_factor = self.scale_factor.to(z.device, dtype=z.dtype)
        else:
            scale_factor = torch.tensor(
                self.scale_factor,
                device=z.device,
                dtype=z.dtype,
            )

        z = z / scale_factor
        pred = self.autoencoder.decode_stage_2_outputs(z)
        return pred

    @torch.no_grad()
    def reverse_diffusion(
        self,
        z_cond: torch.Tensor,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Denoise random latent noise with classifier-free T1 conditioning."""
        if num_inference_steps is None:
            num_inference_steps = Config.NUM_INFERENCE_STEPS

        # Keep this fixed internally.
        # Do not expose this to the demo UI.
        guidance_scale = 4.0

        z_cond = z_cond.to(self.device)

        # Unconditional condition: zero T1 latent
        z_uncond = torch.zeros_like(z_cond, device=self.device)

        # Start from pure noise
        z = torch.randn_like(z_cond, device=self.device)

        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        for t in self.scheduler.timesteps:
            if not torch.is_tensor(t):
                t = torch.tensor(t, device=self.device)

            timesteps = torch.full(
                (z.shape[0],),
                t,
                device=self.device,
                dtype=torch.long,
            )

            # -----------------------------
            # Unconditional prediction
            # -----------------------------
            model_input_uncond = torch.cat([z, z_uncond], dim=1)
            noise_pred_uncond = self.unet(
                model_input_uncond,
                timesteps=timesteps,
            )

            if hasattr(noise_pred_uncond, "sample"):
                noise_pred_uncond = noise_pred_uncond.sample

            # -----------------------------
            # Conditional prediction
            # -----------------------------
            model_input_cond = torch.cat([z, z_cond], dim=1)
            noise_pred_cond = self.unet(
                model_input_cond,
                timesteps=timesteps,
            )

            if hasattr(noise_pred_cond, "sample"):
                noise_pred_cond = noise_pred_cond.sample

            # -----------------------------
            # Classifier-free guidance
            # -----------------------------
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            step_output = self.scheduler.step(noise_pred, t, z)

            if hasattr(step_output, "prev_sample"):
                z = step_output.prev_sample
            elif isinstance(step_output, tuple):
                z = step_output[0]
            else:
                z = step_output

        return z

    @torch.no_grad()
    def infer_tensor(
        self,
        t1: torch.Tensor,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Run model inference for a preprocessed T1 tensor."""
        z_cond = self.encode_condition(t1)

        z_pred = self.reverse_diffusion(
            z_cond=z_cond,
            num_inference_steps=num_inference_steps,
        )

        pred_flair = self.decode_latent(z_pred)
        pred_flair = torch.clamp(pred_flair, -1.0, 1.0)

        return pred_flair
    
    @torch.no_grad()
    def run_and_evaluate(
        self,
        input_path: str,
        gt_path: Optional[str] = None,
        output_path: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
    ) -> dict:
        """Preprocess input files, generate FLAIR output, and attach metrics."""
        if gt_path is not None:
            # Pair preprocessing keeps input and ground-truth transforms aligned.
            item = self.processor.preprocess_pair(
                t1_path=input_path,
                t2_path=gt_path,
                device=self.device,
            )
            t1 = item["t1"]
            gt_flair = item["t2"]
        else:
            item = self.processor.preprocess_input(
                image_path=input_path,
                device=self.device,
            )
            t1 = item["t1"]
            gt_flair = None

        pred_flair = self.infer_tensor(
            t1=t1,
            num_inference_steps=num_inference_steps,
        )

        output_path_obj = self._make_output_path(input_path, output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        save_nifti(pred_flair, str(output_path_obj))

        dataset_metrics = {
            "dataset_mean_psnr": 15.8870,
            "dataset_mean_ssim": 0.3460,
            "dataset_fid": 103.5871,
            "dataset_kid_mean": 0.150175,
            "dataset_kid_std": 0.014372
                            }

        if gt_flair is not None:
            metrics = evaluate_batch(pred_flair, gt_flair)
        else:
            metrics = {
                "psnr": None,
                "ssim": None,
            }

        metrics.update(dataset_metrics)

        return {
            "output_path": str(output_path_obj),
            "pred_flair": pred_flair,
            "gt_flair": gt_flair,
            "t1": t1,
            "metrics": metrics,
            "preprocess_item": item,
        }
