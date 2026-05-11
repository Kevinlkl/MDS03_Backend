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
    def __init__(self):
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
        if output_path is not None:
            return Path(output_path)

        p = Path(input_path)
        if p.name.endswith(".nii.gz"):
            input_name = p.name[:-7]
            return Config.OUTPUTS_DIR / f"{input_name}_pred_flair.nii.gz"
        return Config.OUTPUTS_DIR / f"{p.stem}_pred_flair.nii"

    @torch.no_grad()
    def encode_condition(self, t1: torch.Tensor) -> torch.Tensor:
        z_t1 = self.autoencoder.encode_stage_2_inputs(t1)

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
        if num_inference_steps is None:
            num_inference_steps = Config.NUM_INFERENCE_STEPS

        z = torch.randn_like(z_cond, device=self.device)
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            if not torch.is_tensor(t):
                t_tensor = torch.tensor([t], device=self.device, dtype=torch.long)
            else:
                t_tensor = t.reshape(1).to(self.device).long()

            model_input = torch.cat([z, z_cond], dim=1)
            noise_pred = self.unet(model_input, timesteps=t_tensor)

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
        z_cond = self.encode_condition(t1)
        z_pred = self.reverse_diffusion(
            z_cond=z_cond,
            num_inference_steps=num_inference_steps,
        )
        pred_flair = self.decode_latent(z_pred)
        return pred_flair

    @torch.no_grad()
    def run_and_evaluate(
        self,
        input_path: str,
        gt_path: Optional[str] = None,
        output_path: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
    ) -> dict:
        if gt_path is not None:
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
            "dataset_mean_psnr": None,
            "dataset_mean_ssim": None,
            "dataset_fid": None,
            "dataset_kid_mean": None,
            "dataset_kid_std": None,
            "dataset_metric_scope": "Dataset metrics are not available for T1-FLAIR inference.",
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
