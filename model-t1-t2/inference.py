# inference_pipeline.py

from pathlib import Path
import torch

from config import Config
from preprocess import MRIProcessor
from postprocess import save_nifti
from models.autoencoder import load_autoencoder
from models.diffusion_unet import load_latent_diffusion_unet, build_scheduler


class InferencePipeline:
    def __init__(self):
        self.device = Config.DEVICE

        # Preprocess
        self.processor = MRIProcessor(
            spatial_size=Config.SPATIAL_SIZE,
            pixdim=Config.PIXDIM,
            source_key=Config.INPUT_KEY,
            intensity_lower=Config.INTENSITY_LOWER,
            intensity_upper=Config.INTENSITY_UPPER,
            b_min=Config.B_MIN,
            b_max=Config.B_MAX,
        )

        # Load autoencoder
        self.autoencoder = load_autoencoder(
            checkpoint_path=Config.AUTOENCODER_CKPT,
            latent_channels=Config.LATENT_CHANNELS,
            device=self.device,
        )

        # Load latent diffusion model + metadata
        self.unet, self.diffusion_meta = load_latent_diffusion_unet(
            checkpoint_path=Config.LATENT_DIFFUSION_CKPT,
            in_channels=Config.DIFFUSION_IN_CHANNELS,
            out_channels=Config.DIFFUSION_OUT_CHANNELS,
            device=self.device,
        )

        # Read scale factor from checkpoint metadata
        self.scale_factor = self.diffusion_meta.get("scale_factor", 1.0)

        # Scheduler
        self.scheduler = build_scheduler(
            num_train_timesteps=Config.NUM_TRAIN_TIMESTEPS,
            beta_start=Config.BETA_START,
            beta_end=Config.BETA_END,
        )

    @torch.no_grad()
    def encode_condition(self, t1: torch.Tensor) -> torch.Tensor:
        """
        Encode T1 into latent space for conditioning.
        """
        z_t1 = self.autoencoder.encode_stage_2_inputs(t1)

        if isinstance(self.scale_factor, torch.Tensor):
            scale_factor = self.scale_factor.to(z_t1.device)
        else:
            scale_factor = torch.tensor(self.scale_factor, device=z_t1.device, dtype=z_t1.dtype)

        z_t1 = z_t1 * scale_factor
        return z_t1

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent back to image space.
        """
        if isinstance(self.scale_factor, torch.Tensor):
            scale_factor = self.scale_factor.to(z.device)
        else:
            scale_factor = torch.tensor(self.scale_factor, device=z.device, dtype=z.dtype)

        z = z / scale_factor
        pred = self.autoencoder.decode_stage_2_outputs(z)
        return pred

    @torch.no_grad()
    def reverse_diffusion(self, z_cond: torch.Tensor) -> torch.Tensor:
        """
        Run reverse diffusion starting from random noise.
        """
        z = torch.randn_like(z_cond).to(self.device)

        self.scheduler.set_timesteps(Config.NUM_INFERENCE_STEPS)

        for t in self.scheduler.timesteps:
            if not torch.is_tensor(t):
                t_tensor = torch.tensor([t], device=self.device).long()
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
    def run(self, input_path: str, output_path: str = None) -> str:
        """
        Full inference pipeline:
        1. preprocess T1
        2. encode T1 to latent
        3. reverse diffusion in latent space
        4. decode to predicted T2
        5. save NIfTI
        """
        item = self.processor.preprocess(input_path, device=self.device)
        t1 = item["image"]

        # encode conditioning latent
        z_cond = self.encode_condition(t1)

        # reverse diffusion
        z_pred = self.reverse_diffusion(z_cond)

        # decode predicted latent to T2
        pred_t2 = self.decode_latent(z_pred)

        # output path
        if output_path is None:
            input_name = Path(input_path).stem
            output_path = Config.OUTPUTS_DIR / f"{input_name}_pred_t2.nii"
        else:
            output_path = Path(output_path)

        save_nifti(pred_t2, str(output_path))

        return str(output_path)