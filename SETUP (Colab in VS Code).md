# Team Quickstart (Colab Runtime in VS Code)

Use this guide when your notebook is edited in VS Code but executed on a Colab runtime.

If you are running everything locally, use `SETUP (local).md`.

## What is different from local setup

- File paths are Colab paths (`/content/...`), not Windows paths.
- Local workspace folders are not automatically visible to the Colab runtime.
- Checkpoints must exist in Colab runtime storage (recommended: Google Drive mount).

## 1) Mount Google Drive in Colab runtime

Run this in a notebook cell near the top:

```python
from pathlib import Path

try:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').exists():
        drive.mount('/content/drive')
except Exception as e:
    print(f'Drive mount note: {e}')
```

## 2) Put checkpoints in Drive

Create this structure in Google Drive:

- `/content/drive/MyDrive/models/synthetic_t1_3d_ldm/checkpoints/autoencoder_best.pth`
- `/content/drive/MyDrive/models/synthetic_t1_3d_ldm/checkpoints/latent_diffusion_best.pth`

## 3) Add Colab model path config (Cell 4 helper)

Add this block after your current config setup:

```python
from pathlib import Path

COLAB_MODEL_ROOT = Path('/content/drive/MyDrive/models/synthetic_t1_3d_ldm')
COLAB_CHECKPOINT_DIR = COLAB_MODEL_ROOT / 'checkpoints'

AE_CHECKPOINT_PATH = COLAB_CHECKPOINT_DIR / 'autoencoder_best.pth'
DIFFUSION_CHECKPOINT_PATH = COLAB_CHECKPOINT_DIR / 'latent_diffusion_best.pth'

OUTPUT_ROOT = Path('/content/drive/MyDrive/MDS03_Backend_outputs/synthetic_t1_3d_ldm')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print('AE checkpoint:', AE_CHECKPOINT_PATH, '| exists =', AE_CHECKPOINT_PATH.exists())
print('Diffusion checkpoint:', DIFFUSION_CHECKPOINT_PATH, '| exists =', DIFFUSION_CHECKPOINT_PATH.exists())
print('Output root:', OUTPUT_ROOT)

if not AE_CHECKPOINT_PATH.exists() or not DIFFUSION_CHECKPOINT_PATH.exists():
    raise FileNotFoundError('Missing checkpoints in Drive. Check paths above.')
```

## 4) Inference-only mode (no retraining)

Before running your run-control cell, set:

```python
RUN_AE_TRAIN = False
RUN_DIFF_TRAIN = False
RUN_SAMPLE = True
RUN_DIFF_PSNR_SSIM_EVAL = False

# Optional speed-up
INFERENCE_STEPS = 120
```

## 5) Recommended execution order

1. Run Cell 3 (installs/imports).
2. Run Cell 4 (dataset/config) and the Colab model path helper block.
3. Run Cells 5 to 14.
4. Run inference-only override block.
5. Run Cell 15 to generate sample output.
6. Run Cell 16 for visualization.

## 6) Expected output location

Generated sample is saved under:

- `/content/drive/MyDrive/MDS03_Backend_outputs/synthetic_t1_3d_ldm/generated/sample_synth_t1.nii.gz`

## 7) Common issues

- `FileNotFoundError` on checkpoints:
  - Verify Drive is mounted.
  - Verify exact checkpoint filenames and folders.
- Runtime reset lost data:
  - Keep checkpoints and outputs on Drive, not only `/content`.
- Slow generation:
  - Reduce `INFERENCE_STEPS` (for example, 80 to 120).

## Related guide

- Local setup and local runtime: `SETUP (local).md`
