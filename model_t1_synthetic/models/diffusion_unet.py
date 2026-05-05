import torch


def build_latent_diffusion_unet(in_channels, out_channels, device="cpu"):
	try:
		from generative.networks.nets import DiffusionModelUNet
	except ImportError as exc:
		raise ImportError(
			"monai-generative not installed. Run: pip install monai-generative"
		) from exc

	model = DiffusionModelUNet(
		spatial_dims=3,
		in_channels=in_channels,
		out_channels=out_channels,
		num_res_blocks=2,
		num_channels=(64, 128, 128),
		attention_levels=(False, True, True),
		num_head_channels=(0, 64, 64),
	).to(device)
	return model

def load_latent_diffusion_unet(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print("Checkpoint latent_channels:", checkpoint.get("latent_channels"))

    latent_channels = checkpoint.get("latent_channels", 8)

    model = build_latent_diffusion_unet(
        in_channels=latent_channels,
        out_channels=latent_channels,
        device=device,
    )

    state_dict = checkpoint.get("ema_unet_state_dict", checkpoint.get("unet_state_dict"))

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    metadata = {
        "scale_factor": checkpoint.get("scale_factor", 1.0),
        "latent_channels": latent_channels,
    }

    return model, metadata

def build_scheduler(num_train_timesteps=1000, beta_start=0.0015, beta_end=0.0195):
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
