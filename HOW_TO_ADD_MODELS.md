# How to Add or Replace Models

This guide explains how to hand over new trained models into the backend. There
are two supported approaches:

1. Replace or update an existing model folder.
2. Add a new model folder and expose it through a new API endpoint.

Use option 1 when the new model performs the same task as an existing endpoint.
Use option 2 when the new model is a new task, modality, architecture, or should
remain available beside the current models.

## Current Backend Model Structure

The runnable backend is in:

```text
backend/
```

Current model folders:

| Model folder | Purpose | Current API route file |
|---|---|---|
| `backend/model_t1_t2/` | Generate T2 from T1 | `backend/api/t1_t2_inference.py` |
| `backend/model_t1_flair/` | Generate FLAIR from T1 | `backend/api/t1_flair_inference.py` |
| `backend/model_t1_synthetic/` | Generate synthetic T1 volumes | `backend/api/t1_synthetic_inference.py` |
| `backend/model_classification/` | Brain tumor classification | `backend/api/classification_inference.py` |

Most diffusion model folders follow this layout:

```text
model_name/
  checkpoints/          # trained .pth files, or the configured checkpoint folder
  models/               # model architecture definitions
  config.py             # checkpoint paths, image size, scheduler settings, etc.
  preprocess.py         # input preprocessing, if needed
  postprocess.py        # output saving/evaluation, if needed
  inference.py          # pipeline loaded by the API route
```

Important checkpoint locations used by the current code:

| Model | Checkpoint folder | Expected checkpoint files |
|---|---|---|
| T1 to T2 | `backend/model_t1_t2/checkpoints/` | `autoencoder_best.pth`, `latent_diffusion_best.pth` |
| T1 to FLAIR | `backend/model_t1_flair/checkpoints/` | `autoencoder_best_flair.pth`, `latent_diffusion_best_flair.pth` |
| Synthetic T1 | `backend/model_t1_synthetic/T1_Synthetic_Path/` | `autoencoder_best.pth`, `latent_diffusion_best.pth` |
| Classification | `backend/model_classification/checkpoints/` | `baseline_classifier.pth`, `synthetic_aug_classifier.pth` |

The large `.pth` checkpoint files are not stored in Git. Download them from the
shared storage and place them into the matching checkpoint folder before running
inference.

## Option 1: Replace or Update an Existing Model Folder

Use this when the endpoint stays the same, for example replacing the T1-to-T2
model with a newer T1-to-T2 model.

### Step 1: Choose the model folder

Examples:

```text
backend/model_t1_t2/
backend/model_t1_flair/
backend/model_t1_synthetic/
backend/model_classification/
```

### Step 2: Replace checkpoint files

Place the new `.pth` files into the checkpoint folder expected by that model.
Either keep the same filenames, or update `config.py` / route constants to point
to the new filenames.

For T1 to T2, the default paths are in `backend/model_t1_t2/config.py`:

```python
AUTOENCODER_CKPT = CHECKPOINTS_DIR / "autoencoder_best.pth"
LATENT_DIFFUSION_CKPT = CHECKPOINTS_DIR / "latent_diffusion_best.pth"
```

For T1 to FLAIR, the default paths are in `backend/model_t1_flair/config.py`:

```python
AUTOENCODER_CKPT = CHECKPOINTS_DIR / "autoencoder_best_flair.pth"
LATENT_DIFFUSION_CKPT = CHECKPOINTS_DIR / "latent_diffusion_best_flair.pth"
```

For synthetic T1, the default checkpoint folder is currently:

```python
CHECKPOINTS_DIR = BASE_DIR / "T1 Synthetic Path"
```

For classification, the checkpoint paths are currently set inside
`backend/api/classification_inference.py`:

```python
BASELINE_MODEL_PATH = BASE_DIR / "model_classification" / "checkpoints" / "baseline_classifier.pth"
SYNTHETIC_MODEL_PATH = BASE_DIR / "model_classification" / "checkpoints" / "synthetic_aug_classifier.pth"
```

### Step 3: Update model settings if training changed

Edit the model folder's `config.py` if the new model used different training
settings. Common values to check:

```python
SPATIAL_SIZE
PIXDIM
LATENT_CHANNELS
AUTOENCODER_CHANNELS
DIFFUSION_IN_CHANNELS
DIFFUSION_OUT_CHANNELS
NUM_TRAIN_TIMESTEPS
BETA_START
BETA_END
NUM_INFERENCE_STEPS
```

The values in `config.py` must match the way the checkpoint was trained. If they
do not match, the checkpoint may fail to load or inference output may be poor.

### Step 4: Update architecture files if needed

If the new checkpoint was trained with a changed architecture, update the files
inside the model folder's `models/` directory.

Examples:

```text
backend/model_t1_t2/models/autoencoder.py
backend/model_t1_t2/models/diffusion_unet.py
backend/model_classification/classifier.py
```

If the architecture is unchanged, do not edit these files. Replacing only the
checkpoint files and `config.py` is usually enough.

### Step 5: Update output labels or response keys if the task changed

If the output is still the same modality, no API change is needed.

If the output name changed, update the matching route file in `backend/api/`.
For example, the T1-to-T2 route currently returns:

```python
"mode": "t1-t2"
```

and creates download names like:

```python
download_name = f"{base_name}_pred_t2_{num_inference_steps}steps.nii.gz"
```

Keep response fields stable if the frontend already depends on them.

### Step 6: Restart and test

From the backend folder:

```powershell
cd backend
python -m uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Test the relevant endpoint with a `.nii` or `.nii.gz` file.

## Option 2: Add a New Model Folder

Use this when wanting to add a separate model without overwriting
the existing model folders.

### Step 1: Copy the closest existing model folder

Choose the current folder that is closest to the new model.

Examples:

```text
backend/model_t1_t2/          -> copy for another conditional T1 translation model
backend/model_t1_flair/       -> copy for another guided T1 translation model
backend/model_t1_synthetic/   -> copy for another unconditional generator
backend/model_classification/ -> copy for another classifier
```

Rename the copied folder using a clear name. Example:

```text
backend/model_t1_pd/
```

### Step 2: Update imports inside the copied model folder

Open the copied files and replace imports that still point to the old folder.

Example for a copied `model_t1_pd` folder:

```python
from model_t1_pd.config import Config
from model_t1_pd.preprocess import MRIProcessor
from model_t1_pd.models.autoencoder import load_autoencoder
from model_t1_pd.models.diffusion_unet import load_latent_diffusion_unet
```

If a copied file imports shared utilities from another model folder, only keep
that import if it is intentionally shared. For example, the FLAIR model currently
uses:

```python
from model_t1_t2.postprocess import save_nifti, evaluate_batch
```

This is acceptable if the same postprocessing is still correct.

### Step 3: Add the new checkpoints

Create a checkpoint folder inside the new model folder:

```text
backend/model_t1_pd/checkpoints/
```

Place the new `.pth` files there and update the new folder's `config.py`:

```python
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
AUTOENCODER_CKPT = CHECKPOINTS_DIR / "autoencoder_best.pth"
LATENT_DIFFUSION_CKPT = CHECKPOINTS_DIR / "latent_diffusion_best.pth"
```

### Step 4: Update the new model configuration

In the copied `config.py`, update all values that are specific to the new model:

```python
SPATIAL_SIZE
PIXDIM
LATENT_CHANNELS
AUTOENCODER_CHANNELS
DIFFUSION_IN_CHANNELS
DIFFUSION_OUT_CHANNELS
NUM_TRAIN_TIMESTEPS
BETA_START
BETA_END
NUM_INFERENCE_STEPS
OUTPUT_FILENAME
```

The configuration must match the training notebook and checkpoint metadata.

### Step 5: Create a new API route

Copy the closest route file in `backend/api/` and rename it.

Example:

```text
backend/api/t1_pd_inference.py
```

Inside the new route file:

1. Import the new pipeline:

```python
from model_t1_pd.inference import InferencePipeline
```

2. Give the router a clear tag:

```python
router = APIRouter(prefix="/api", tags=["T1 to PD Inference"])
```

3. Rename the endpoint path:

```python
@router.post("/infer_t1_pd")
```

4. Rename the download endpoint if needed:

```python
@router.get("/download_t1_pd")
```

5. Update response labels, tensor keys, and download filenames to match the new
   output modality.

### Step 6: Register the new route in `backend/main.py`

Add an import:

```python
from api.t1_pd_inference import router as t1_pd_router
```

Then include the router:

```python
app.include_router(t1_pd_router)
```

After restarting the server, the new endpoint should appear in Swagger at:

```text
http://127.0.0.1:8000/docs
```

## Checkpoint Compatibility Notes

The diffusion loaders accept common checkpoint formats, including dictionaries
with keys such as:

```text
autoencoder_state_dict
model_state_dict
ema_unet_state_dict
unet_state_dict
state_dict
scale_factor
latent_channels
```

If a new training script saves different keys, update the relevant loader:

```text
backend/<model_folder>/models/autoencoder.py
backend/<model_folder>/models/diffusion_unet.py
```

The classifier loader accepts either a raw state dict or a dictionary containing:

```text
model_state_dict
```

## Quick Validation Checklist

Before considering the model added, confirm:

- The new checkpoint files exist in the folder configured by `config.py`.
- The checkpoint filenames in `config.py` or the API route match the actual files.
- The model architecture code matches the checkpoint architecture.
- `SPATIAL_SIZE`, `PIXDIM`, latent channels, scheduler settings, and beta schedule match training.
- The FastAPI server starts without import errors.
- The endpoint appears in `http://127.0.0.1:8000/docs`.
- A sample `.nii` or `.nii.gz` request returns `"success": true`.
- The generated output file can be downloaded from the returned download endpoint.
- The frontend is updated if a new endpoint path, mode name, or response field was added.

## Common Problems

| Problem | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError` for `.pth` | Checkpoint missing or wrong filename | Place file in the configured checkpoint folder or update the path in `config.py` |
| `Missing key(s) in state_dict` | Architecture changed | Update `models/autoencoder.py`, `models/diffusion_unet.py`, or classifier architecture |
| `size mismatch` when loading checkpoint | Config values or architecture do not match training | Check latent channels, model channels, and output classes |
| Endpoint does not appear in Swagger | Router not registered | Add import and `app.include_router(...)` in `backend/main.py` |
| API starts but inference fails | Model loads lazily only when endpoint is called | Check the full error returned by that endpoint and verify checkpoint/config compatibility |
| Frontend cannot call new model | Frontend endpoint list was not updated | Add the new API path and mode label in the frontend |

## Recommended Handover Practice

For each new trained model, keep a small note beside the checkpoint files with:

- Model purpose.
- Training notebook or script used.
- Date trained.
- Dataset used.
- Required input modality.
- Output modality or class labels.
- Expected checkpoint filenames.
- Important `config.py` values.
- Test sample used for verification.

This makes future model replacement much safer, especially when several
checkpoints use similar filenames.
