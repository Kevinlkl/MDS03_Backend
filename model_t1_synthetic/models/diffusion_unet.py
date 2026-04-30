import torch


def build_latent_diffusion_unet(in_channels, out_channels, device="cpu"):
	try:
		from generative.networks.nets import DiffusionModelUNet
	except ImportError as exc:
		raise ImportError(
			"monai-generative not installed. Run: pip install monai-generative"
		) from exc

	# Import config to get correct channel dimensions
	from model_t1_synthetic.config import Config

	# sanitize/derive head channels: ensure head-channel sizes are sensible relative
	# to the configured `DIFFUSION_CHANNELS`. If user config contains 0 or values
	# larger than the corresponding channel count, derive a safe default.
	raw_heads = getattr(Config, "DIFFUSION_HEAD_CHANNELS", None)
	channels = list(getattr(Config, "DIFFUSION_CHANNELS", ()))
	if not channels:
		num_head_channels = None
	else:
		# Build a per-level head-channel tuple matching length of channels
		num_head_channels_list = []
		for i, c in enumerate(channels):
			# desired from raw config if available
			raw_h = None
			try:
				raw_h = int(raw_heads[i]) if raw_heads is not None else None
			except Exception:
				raw_h = None
			# If invalid or out-of-range, pick sensible default (channels//4, at least 1)
			if raw_h is None or raw_h <= 0 or raw_h > c:
				h = max(1, c // 4)
			else:
				h = raw_h
			num_head_channels_list.append(h)
		num_head_channels = tuple(num_head_channels_list)

	model = DiffusionModelUNet(
		spatial_dims=3,
		in_channels=in_channels,
		out_channels=out_channels,
		num_res_blocks=2,
		num_channels=Config.DIFFUSION_CHANNELS,
		attention_levels=(False, True, True),
		num_head_channels=num_head_channels,
	).to(device)
	return model


def load_latent_diffusion_unet(checkpoint_path, in_channels, out_channels, device="cpu", use_ema=True):
	model = build_latent_diffusion_unet(
		in_channels=in_channels,
		out_channels=out_channels,
		device=device,
	)

	checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

	print("Diffusion checkpoint keys:", checkpoint.keys())

	state_dict_key = "ema_unet_state_dict" if use_ema and "ema_unet_state_dict" in checkpoint else "unet_state_dict"
	if state_dict_key in checkpoint:
		model.load_state_dict(checkpoint[state_dict_key], strict=True)
	elif "model_state_dict" in checkpoint:
		model.load_state_dict(checkpoint["model_state_dict"], strict=True)
	else:
		model.load_state_dict(checkpoint, strict=True)

	model.eval()

	metadata = {
		"scale_factor": checkpoint.get("scale_factor", 1.0),
		"latent_channels": checkpoint.get("latent_channels", out_channels),
		"epoch": checkpoint.get("epoch", None),
		"best_val_loss": checkpoint.get("best_val_loss", None),
		"state_dict_key": state_dict_key,
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
