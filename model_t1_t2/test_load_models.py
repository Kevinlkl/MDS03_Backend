from config import Config
from models.autoencoder import load_autoencoder
from models.diffusion_unet import load_latent_diffusion_unet

autoencoder = load_autoencoder(
    checkpoint_path=Config.AUTOENCODER_CKPT,
    latent_channels=Config.LATENT_CHANNELS,
    device=Config.DEVICE,
)
print("Autoencoder loaded successfully")

unet, diffusion_meta = load_latent_diffusion_unet(
    checkpoint_path=Config.LATENT_DIFFUSION_CKPT,
    in_channels=Config.DIFFUSION_IN_CHANNELS,
    out_channels=Config.DIFFUSION_OUT_CHANNELS,
    device=Config.DEVICE,
)
print("Latent diffusion UNet loaded successfully")
print("Diffusion metadata:", diffusion_meta)