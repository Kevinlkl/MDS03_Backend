# MDS03 Backend

This repository contains the backend system and training notebooks for the MDS03 FYP project on diffusion-based MRI synthesis and classification.

The project supports:

- T1 → T2 MRI translation
- T1 → FLAIR MRI translation
- Synthetic T1 MRI generation
- Tumor classification analysis
- Diffusion-based latent MRI synthesis

---

# Backend Overview

The `backend/` folder contains the runnable FastAPI backend used for model inference and API integration.

Each model folder follows a standardized structure:

| File / Folder | Description |
|---|---|
| `checkpoints/` | Stores trained model weights |
| `models/` | Neural network architecture definitions |
| `config.py` | Model configuration and parameter settings |
| `preprocess.py` | Input preprocessing pipeline |
| `postprocess.py` | Output reconstruction and postprocessing |
| `inference.py` | Main inference logic |

---

# Model Checkpoints

Due to GitHub storage and Git LFS limitations, trained model checkpoints (`.pth`) are **not fully included** in this repository.

The checkpoints are stored separately in the shared Google Drive folder.
[https://drive.google.com/drive/folders/1sEODGyG4zEG3fh1G4L1y13us3wVqKAJo?usp=sharing]

Before running the backend, the required checkpoint files must be manually downloaded and placed into the corresponding `checkpoints/` folders.

Example:

```text
backend/model_t1_t2/checkpoints/
backend/model_t1_flair/checkpoints/
backend/model_t1_synthetic/checkpoints/
backend/model_classification/checkpoints/
```

Please ensure all required checkpoint files are uploaded correctly before starting the backend.

---

# Training Notebook Overview

The `training_notebook/` folder contains notebooks used during experimentation, training, evaluation, and dataset preparation.

| Notebook | Purpose |
|---|---|
| `classification_model.ipynb` | Tumor classification training |
| `diffusion_synthetic_t1.ipynb` | Synthetic T1 diffusion model training |
| `diffusion_synthetic_t1_evaluation.ipynb` | Evaluation and analysis for synthetic T1 generation |
| `diffusion_t1_to_flair.ipynb` | T1 → FLAIR translation training |
| `diffusion_t1_to_t2.ipynb` | T1 → T2 translation training |
| `synthetic_dataset_generation.ipynb` | Synthetic MRI dataset generation |

---

# Setup Instructions

## 1. Create Python Environment

```powershell
python -m venv venv
```

Activate environment:

### Windows

```powershell
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 2. Install Dependencies

```powershell
pip install -r requirements-sd.txt
```

---

# Running the Backend

Move into the backend folder:

```powershell
cd backend
```

Start the FastAPI server:

```powershell
python -m uvicorn main:app --reload
```

---

# Using the Backend

The backend is designed to serve MRI synthesis and classification inference through FastAPI endpoints.

## General Workflow

1. Start the FastAPI backend server.
2. Open the Swagger API interface.
3. Upload MRI `.nii` or `.nii.gz` files.
4. Select the desired inference endpoint.
5. Receive generated MRI outputs and evaluation metrics.

---

# API Documentation

After starting the server, open:

```text
http://127.0.0.1:8000/docs
```

Interactive Swagger UI documentation will be available there.

The available endpoints include:

| Endpoint | Purpose |
|---|---|
| `/api/t1_t2_inference` | T1 → T2 MRI translation |
| `/api/t1_flair_inference` | T1 → FLAIR MRI translation |
| `/api/t1_synthetic_inference` | Synthetic T1 MRI generation |
| `/api/classification_inference` | Tumor classification analysis |

---

# Features

- Diffusion-based latent MRI synthesis
- 3D MRI preprocessing pipeline
- FastAPI inference endpoints
- Synthetic MRI generation
- MRI modality translation
- Tumor classification comparison
- Dataset-level evaluation metrics

---

# Notes

- Large model checkpoints (`.pth`) are stored externally in Google Drive.
- Checkpoints must be manually placed into their respective `checkpoints/` folders before inference.
- The backend folder only contains runnable inference-related code.
- Training notebooks are separated for cleaner deployment structure.

---

# Technologies Used

- Python
- PyTorch
- MONAI
- FastAPI
- Uvicorn
- NumPy
- NiBabel
- Jupyter Notebook

---

# Authors

MDS03 FYP Team