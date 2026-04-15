from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import tempfile
import shutil
import torch

from model_t1_t2.preprocess import MRIProcessor
from model_t1_t2.postprocess import evaluate_batch
from model_t1_t2.config import Config

router = APIRouter(prefix="/api", tags=["Analysis"])

# Initialize processor once
processor = MRIProcessor(source_key="image")


@router.post("/analyze")
async def analyze(
    input_file: UploadFile = File(...),
    generated_file: UploadFile = File(...),
    ground_truth_file: UploadFile = File(...),
):
    # Validate file extensions
    for upload, name in [
        (input_file, "input_file"),
        (generated_file, "generated_file"),
        (ground_truth_file, "ground_truth_file"),
    ]:
        if not upload.filename.endswith((".nii", ".nii.gz")):
            raise HTTPException(
                status_code=400,
                detail=f"Only .nii or .nii.gz files are supported for {name}."
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded files
        input_path = Path(tmpdir) / "input.nii"
        with open(input_path, "wb") as f:
            shutil.copyfileobj(input_file.file, f)

        generated_path = Path(tmpdir) / "generated.nii"
        with open(generated_path, "wb") as f:
            shutil.copyfileobj(generated_file.file, f)

        gt_path = Path(tmpdir) / "ground_truth.nii"
        with open(gt_path, "wb") as f:
            shutil.copyfileobj(ground_truth_file.file, f)

        # Preprocess using MRIProcessor
        generated_item = processor.preprocess(str(generated_path), device=Config.DEVICE)
        gt_item = processor.preprocess(str(gt_path), device=Config.DEVICE)

        generated_tensor = generated_item["image"]
        gt_tensor = gt_item["image"]

        # Evaluate metrics
        results = evaluate_batch(generated_tensor, gt_tensor)

        # Return metrics safely
        return {
            "ssim": round(float(results["ssim"]), 2) if results["ssim"] is not None else None,
            "psnr": round(float(results["psnr"]), 2) if results["psnr"] is not None else None,
            "fid": round(float(results["fid"]), 2) if results["fid"] is not None else None,
        }
