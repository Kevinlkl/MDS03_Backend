import torch


def build_autoencoder(latent_channels=4, device="cpu"):
	try:
		from generative.networks.nets import AutoencoderKL
	except ImportError as exc:
		raise ImportError(
			"monai-generative not installed. Run: pip install monai-generative"
		) from exc

	model = AutoencoderKL(
		spatial_dims=3,
		in_channels=1,
		out_channels=1,
		num_channels=(32, 64, 128),
		latent_channels=latent_channels,
		num_res_blocks=2,
		norm_num_groups=16,
		attention_levels=(False, False, True),
	).to(device)
	return model


def load_autoencoder(checkpoint_path, latent_channels=4, device="cpu"):
	model = build_autoencoder(latent_channels=latent_channels, device=device)

	checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

	if isinstance(checkpoint, dict):
		print("Autoencoder checkpoint keys:", checkpoint.keys())

		if "autoencoder_state_dict" in checkpoint:
			model.load_state_dict(checkpoint["autoencoder_state_dict"])
		elif "model_state_dict" in checkpoint:
			model.load_state_dict(checkpoint["model_state_dict"])
		else:
			model.load_state_dict(checkpoint)
	else:
		model.load_state_dict(checkpoint)

	model.eval()
	return model
