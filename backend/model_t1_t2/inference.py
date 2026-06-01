from pathlib import Path
from typing import Optional
import torch

from model_t1_t2.config import Config
from model_t1_t2.preprocess import MRIProcessor
from model_t1_t2.postprocess import save_nifti, evaluate_batch
from model_t1_t2.models.autoencoder import load_autoencoder
from model_t1_t2.models.diffusion_unet import load_latent_diffusion_unet, build_scheduler



class InferencePipeline:
    """
    Class description:
        End-to-end T1-to-T2 inference pipeline with preprocessing and metrics.

    Attributes:
        device (str): Torch device used for inference.
        processor (MRIProcessor): Preprocessing pipeline for input volumes.
        autoencoder (torch.nn.Module): Trained autoencoder used for latent encoding and decoding.
        unet (torch.nn.Module): Trained latent diffusion denoising model.
        scale_factor (float | torch.Tensor): Latent scaling value saved with the checkpoint.
        scheduler (DDPMScheduler): Noise scheduler used during reverse diffusion.
    """

    def __init__(self):
        """
        Function description:
            Load preprocessing transforms, trained models, and the noise scheduler.

        Parameters:
            None

        Returns:
            None
        """
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
        """
        Function description:
            Resolve the output NIfTI path for a generated T2 volume.

        Parameters:
            input_path (str): Path to the source T1 input file.
            output_path (str | None): Optional explicit output file path.

        Returns:
            Path: Output path for the generated T2 NIfTI file.
        """
        if output_path is not None:
            return Path(output_path)

        p = Path(input_path)
        if p.name.endswith(".nii.gz"):
            input_name = p.name[:-7]
            return Config.OUTPUTS_DIR / f"{input_name}_pred_t2.nii.gz"
        return Config.OUTPUTS_DIR / f"{p.stem}_pred_t2.nii"

    @torch.no_grad()
    def encode_condition(self, t1: torch.Tensor) -> torch.Tensor:
        """
        Function description:
            Encode the input T1 tensor into the scaled latent conditioning space.

        Parameters:
            t1 (torch.Tensor): Preprocessed T1 tensor.

        Returns:
            torch.Tensor: Scaled latent conditioning tensor.
        """
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
        """
        Function description:
            Decode a predicted latent tensor back into image space.

        Parameters:
            z (torch.Tensor): Predicted latent tensor.

        Returns:
            torch.Tensor: Decoded T2 image tensor.
        """
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
        """
        Function description:
            Denoise random latent noise while conditioning on the encoded T1 volume.

        Parameters:
            z_cond (torch.Tensor): Encoded T1 latent conditioning tensor.
            num_inference_steps (int | None): Optional number of denoising steps.

        Returns:
            torch.Tensor: Predicted latent tensor for the T2 output.
        """
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
        """
        Function description:
            Run model inference for a preprocessed T1 tensor.

        Parameters:
            t1 (torch.Tensor): Preprocessed T1 tensor.
            num_inference_steps (int | None): Optional number of denoising steps.

        Returns:
            torch.Tensor: Generated T2 tensor.
        """
        z_cond = self.encode_condition(t1)
        z_pred = self.reverse_diffusion(
            z_cond=z_cond,
            num_inference_steps=num_inference_steps,
        )
        pred_t2 = self.decode_latent(z_pred)
        return pred_t2

    @torch.no_grad()
    def run_and_evaluate(
        self,
        input_path: str,
        gt_path: Optional[str] = None,
        output_path: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
    ) -> dict:
        """
        Function description:
            Preprocess input files, generate T2 output, and attach metrics.

        Parameters:
            input_path (str): Path to the source T1 NIfTI file.
            gt_path (str | None): Optional path to a ground-truth T2 NIfTI file.
            output_path (str | None): Optional output path for the generated NIfTI file.
            num_inference_steps (int | None): Optional number of denoising steps.

        Returns:
            dict: Generated tensors, output path, metrics, and preprocessing details.
        """
        if gt_path is not None:
            # Pair preprocessing keeps input and ground-truth transforms aligned.
            item = self.processor.preprocess_pair(
                t1_path=input_path,
                t2_path=gt_path,
                device=self.device,
            )
            t1 = item["t1"]
            gt_t2 = item["t2"]
        else:
            item = self.processor.preprocess_input(
                image_path=input_path,
                device=self.device,
            )
            t1 = item["t1"]
            gt_t2 = None

        pred_t2 = self.infer_tensor(
            t1=t1,
            num_inference_steps=num_inference_steps,
        )

        output_path_obj = self._make_output_path(input_path, output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        save_nifti(pred_t2, str(output_path_obj))

        # Static dataset-level metrics are shown alongside optional per-case metrics.
        dataset_metrics = {
            "dataset_mean_psnr": 18.6299,
            "dataset_mean_ssim": 0.6465,
            "dataset_fid": 86.9938,
            "dataset_kid_mean": 0.07973,
            "dataset_kid_std": 0.006863,
            "dataset_metric_scope": "Precomputed on test dataset",
        }

        if gt_t2 is not None:
            metrics = evaluate_batch(pred_t2, gt_t2)
        else:
            metrics = {
                "psnr": None,
                "ssim": None,
            }

        metrics.update(dataset_metrics)

        return {
            "output_path": str(output_path_obj),
            "pred_t2": pred_t2,
            "gt_t2": gt_t2,
            "t1": t1,
            "metrics": metrics,
            "preprocess_item": item,
        }
