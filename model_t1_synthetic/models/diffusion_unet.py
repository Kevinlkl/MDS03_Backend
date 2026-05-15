import torch


def build_latent_diffusion_unet(
    in_channels,
    out_channels,
    device="cpu",
):
    from generative.networks.nets import DiffusionModelUNet

    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        num_channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_head_channels=(0, 64, 64),
        num_res_blocks=2,
        norm_num_groups=16,
    ).to(device)

    return model

def load_latent_diffusion_unet(checkpoint_path, device="cpu"):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    latent_channels = checkpoint.get("latent_channels", 8)

    print("Checkpoint latent_channels:", latent_channels)

    model = build_latent_diffusion_unet(
        in_channels=latent_channels,
        out_channels=latent_channels,
        device=device,
    )

    state_dict = (
        checkpoint.get("ema_unet_state_dict")
        or checkpoint.get("unet_state_dict")
        or checkpoint.get("model_state_dict")
        or checkpoint.get("state_dict")
    )

    if state_dict is None:
        raise KeyError(
            "No UNet state dict found. Expected one of: "
            "ema_unet_state_dict, unet_state_dict, model_state_dict, state_dict"
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    metadata = {
        "scale_factor": checkpoint.get("scale_factor", 1.0),
        "latent_channels": latent_channels,
        "memory_preset": checkpoint.get("memory_preset", "safe"),
        "epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
    }

    return model, metadata

def build_scheduler(num_train_timesteps=1000, beta_start=0.0005, beta_end=0.012):
	try:
		from generative.networks.schedulers import DDPMScheduler
	except ImportError as exc:
		raise ImportError(
			"monai-generative not installed. Run: pip install monai-generative"
		) from exc

	return DDPMScheduler(
		num_train_timesteps=num_train_timesteps,
		schedule="scaled_linear_beta",
		beta_start=beta_start,
		beta_end=beta_end,
	)
