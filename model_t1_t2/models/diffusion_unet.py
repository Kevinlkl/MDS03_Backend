# models/diffusion_unet.py
import torch
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler


def build_latent_diffusion_unet(in_channels, out_channels, device="cpu"):
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        num_res_blocks=2,
        num_channels=(64, 128, 256),
        attention_levels=(False, True, True),
        num_head_channels=(0, 64, 128),
    ).to(device)
    return model


def load_latent_diffusion_unet(checkpoint_path, in_channels, out_channels, device="cpu"):
    model = build_latent_diffusion_unet(
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print("Diffusion checkpoint keys:", checkpoint.keys())

    if "unet_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["unet_state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    metadata = {
        "scale_factor": checkpoint.get("scale_factor", 1.0),
        "latent_channels": checkpoint.get("latent_channels", out_channels),
        "epoch": checkpoint.get("epoch", None),
        "best_val_loss": checkpoint.get("best_val_loss", None),
    }

    return model, metadata


def build_scheduler(num_train_timesteps=1000, beta_start=0.0015, beta_end=0.0195):
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        schedule="scaled_linear_beta",
        beta_start=beta_start,
        beta_end=beta_end,
    )