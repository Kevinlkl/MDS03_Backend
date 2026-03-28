# Team Quickstart (Model Setup)

Use this guide for local setup and local runtime only.

If you are editing in VS Code but running the notebook on a Colab runtime, use `SETUP (Colab in VS Code).md` instead.

## What this solves

- Model files are too large for GitHub.
- Everyone can run inference without retraining.
- Training stays on Colab GPU; local machines can use CPU.

## 1) First-time setup (local)

Run in project root:

```powershell
Copy-Item .env.example .env
mkdir models\synthetic_t1_3d_ldm\checkpoints
mkdir models\sd_t1_mri\lora
mkdir outputs
```

## 2) Copy trained files from Colab export

Place these locally:

- `models/synthetic_t1_3d_ldm/checkpoints/autoencoder_best.pth`
- `models/synthetic_t1_3d_ldm/checkpoints/latent_diffusion_best.pth`
- `models/sd_t1_mri/lora/pytorch_lora_weights.safetensors` (if LoRA is used)

## 3) Set runtime device in `.env`

Use CPU for local machines without NVIDIA driver:

```env
LATENT_DIFFUSION_DIR=./models/synthetic_t1_3d_ldm
SD_LORA_DIR=./models/sd_t1_mri
DEVICE=cpu
```

## 4) Verify

```powershell
python config.py
python -c "from model_loader import print_model_status; print_model_status()"
python example_inference.py
```

You are good when:

- checkpoints show `Exists: True`
- status shows `READY`
- no CUDA driver crash

## 5) Backend startup load

Add this pattern in `main.py`:

```python
from model_loader import LatentDiffusion3DLoader

@app.on_event("startup")
async def load_models():
    global autoencoder, diffusion_unet, scheduler
    autoencoder, _ = LatentDiffusion3DLoader.load_autoencoder()
    diffusion_unet, _ = LatentDiffusion3DLoader.load_diffusion_unet(use_ema=True)
    scheduler = LatentDiffusion3DLoader.load_noise_scheduler()
```

## 6) Git rules (critical)

Do not commit:

- `.env`
- `models/`
- `outputs/`
- `*.pth`
- `*.safetensors`

Check before commit:

```powershell
git status
```

## 7) Summary setup:

1. Clone repo.
2. Copy `.env.example` to `.env`.
3. Copy model files into local `models/`.
4. Run the 3 verify commands.

No retraining is required for normal backend inference.

## Related guide

- Colab runtime in VS Code: `SETUP (Colab in VS Code).md`
