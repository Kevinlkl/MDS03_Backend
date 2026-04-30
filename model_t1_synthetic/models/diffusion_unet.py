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
		num_channels=(64, 128, 256),
		attention_levels=(False, True, True),
		num_head_channels=(0, 32, 64),
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

	# Custom loading that skips mismatched keys
	state_dict = checkpoint["unet_state_dict"]
	model_state = model.state_dict()
	
	incompatible_keys = {"missing_keys": [], "unexpected_keys": []}
	loaded_keys = set()
	
	for key, param in state_dict.items():
		if key in model_state:
			if param.shape == model_state[key].shape:
				model_state[key].copy_(param)
				loaded_keys.add(key)
			else:
				# Size mismatch - skip this key
				incompatible_keys["missing_keys"].append(key)
		else:
			incompatible_keys["unexpected_keys"].append(key)
	
	# Report what was skipped
	if incompatible_keys["missing_keys"]:
		print(f"Skipped {len(incompatible_keys['missing_keys'])} keys with shape mismatches")
	
	model.load_state_dict(model_state, strict=False)
	model.eval()

	metadata = {
		"scale_factor": checkpoint.get("scale_factor", 1.0),
		"latent_channels": checkpoint.get("latent_channels", out_channels),
		"epoch": checkpoint.get("epoch", None),
		"best_val_loss": checkpoint.get("best_val_loss", None),
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
