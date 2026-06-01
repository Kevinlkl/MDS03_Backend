# models/diffusion_unet.py
import torch
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler


def build_latent_diffusion_unet(in_channels, out_channels, device="cpu"):
    """
    Function description:
        Build the latent diffusion UNet for T1-to-FLAIR generation.

    Parameters:
        in_channels (int): Number of input channels accepted by the UNet.
        out_channels (int): Number of output channels produced by the UNet.
        device (str | torch.device): Device where the model should be placed.

    Returns:
        DiffusionModelUNet: UNet model on the selected device.
    """
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        num_res_blocks=3,
        num_channels=(64, 128, 256),
        attention_levels=(False, True, True),
        num_head_channels=(0, 64, 128),
    ).to(device)
    return model


def load_latent_diffusion_unet(checkpoint_path, in_channels, out_channels, device="cpu"):
    """
    Function description:
        Build the latent diffusion UNet and load saved checkpoint weights.

    Parameters:
        checkpoint_path (str | Path): Path to the saved diffusion checkpoint.
        in_channels (int): Number of input channels accepted by the UNet.
        out_channels (int): Number of output channels produced by the UNet.
        device (str | torch.device): Device used for loading and inference.

    Returns:
        tuple[DiffusionModelUNet, dict]: Evaluation-mode UNet and checkpoint metadata.
    """
    model = build_latent_diffusion_unet(
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print("Diffusion checkpoint keys:", checkpoint.keys())

    if "ema_unet_state_dict" in checkpoint:
        print("Loading EMA UNet weights")
        model.load_state_dict(checkpoint["ema_unet_state_dict"])

    elif "model_state_dict" in checkpoint:
        print("Loading model_state_dict")
        model.load_state_dict(checkpoint["model_state_dict"])

    else:
        print("Loading raw checkpoint")
        model.load_state_dict(checkpoint)

    model.eval()

    metadata = {
        "scale_factor": checkpoint.get("scale_factor", 1.0),
        "latent_channels": checkpoint.get("latent_channels", out_channels),
        "epoch": checkpoint.get("epoch", None),
        "best_val_loss": checkpoint.get("best_val_loss", None),
    }

    return model, metadata


def build_scheduler(num_train_timesteps=1000, beta_start=0.0015, beta_end=0.012):
    """
    Function description:
        Build the DDIM scheduler used for T1-to-FLAIR reverse diffusion.

    Parameters:
        num_train_timesteps (int): Number of training timesteps in the diffusion schedule.
        beta_start (float): Initial beta value for the scaled-linear schedule.
        beta_end (float): Final beta value for the scaled-linear schedule.

    Returns:
        DDIMScheduler: Configured diffusion scheduler.
    """
    return DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        schedule="scaled_linear_beta",
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False
    )
