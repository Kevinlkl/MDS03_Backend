# models/autoencoder.py
import torch
from generative.networks.nets import AutoencoderKL


def build_autoencoder(latent_channels=3, device="cpu"):
    model = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 128, 256),
        latent_channels=8,
        num_res_blocks=2,
        norm_num_groups=16,
        attention_levels=(False, False, True),
    ).to(device)
    return model


def load_autoencoder(checkpoint_path, latent_channels=3, device="cpu"):
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