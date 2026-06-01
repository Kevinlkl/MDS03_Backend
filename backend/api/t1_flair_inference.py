from pathlib import Path
import tempfile
import shutil
import base64
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from api.preview_util import tensor_middle_slice_to_png_bytes

from model_t1_flair.inference import InferencePipeline

router = APIRouter(prefix="/api", tags=["T1 to FLAIR Inference"])

pipeline: Optional[InferencePipeline] = None


def get_pipeline() -> InferencePipeline:
    """Create and cache the T1-to-FLAIR inference pipeline on first use."""
    global pipeline
    if pipeline is None:
        # Loading checkpoints here would crash app startup if files are missing.
        pipeline = InferencePipeline()
    return pipeline


def cleanup_files(*paths):
    """Delete temporary files created during request processing."""
    for p in paths:
        try:
            path = Path(p)
            if path.exists():
                path.unlink()
        except Exception:
            pass

def png_buffer_to_base64(png_buffer) -> str:
    """Convert an in-memory PNG buffer into a base64 string for JSON output."""
    return base64.b64encode(png_buffer.getvalue()).decode("utf-8")

@router.post("/infer_t1_flair")
async def infer_mri(
    file: UploadFile = File(...),
    ground_truth_file: Optional[UploadFile] = File(None),
    num_inference_steps: int = Form(1000),
):
    """Generate a synthetic FLAIR MRI volume from an uploaded T1 MRI file."""
    input_filename = file.filename or ""
    gt_filename = (ground_truth_file.filename or "") if ground_truth_file else ""

    # Validate input before writing upload contents to disk.
    if not input_filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Only .nii or .nii.gz files are supported for input file.",
        )

    if ground_truth_file and not gt_filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Only .nii or .nii.gz files are supported for ground truth file.",
        )

    if not (1 <= num_inference_steps <= 1000):
        raise HTTPException(
            status_code=400,
            detail="num_inference_steps must be between 1 and 1000.",
        )

    input_path = None
    gt_path = None
    output_path = None

    try:
        # FastAPI upload files are streamed into temporary NIfTI files for MONAI.
        input_suffix = ".nii.gz" if input_filename.endswith(".nii.gz") else ".nii"

        with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as tmp_input:
            shutil.copyfileobj(file.file, tmp_input)
            input_path = tmp_input.name

        if ground_truth_file is not None:
            gt_suffix = ".nii.gz" if gt_filename.endswith(".nii.gz") else ".nii"
            with tempfile.NamedTemporaryFile(delete=False, suffix=gt_suffix) as tmp_gt:
                shutil.copyfileobj(ground_truth_file.file, tmp_gt)
                gt_path = tmp_gt.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_output:
            output_path = tmp_output.name

        # Instantiate the model only when this endpoint is actually called.
        active_pipeline = get_pipeline()

        result = active_pipeline.run_and_evaluate(
            input_path=input_path,
            gt_path=gt_path,
            output_path=output_path,
            num_inference_steps=num_inference_steps,
        )

        t1 = result["t1"]
        pred_flair = result["pred_flair"]
        gt_flair = result["gt_flair"]
        metrics = result["metrics"]

        # Encode middle-slice previews so the frontend can render quick output.
        t1_preview_b64 = png_buffer_to_base64(
            tensor_middle_slice_to_png_bytes(t1)
        )
        pred_preview_b64 = png_buffer_to_base64(
            tensor_middle_slice_to_png_bytes(pred_flair)
        )
        gt_preview_b64 = (
            png_buffer_to_base64(tensor_middle_slice_to_png_bytes(gt_flair))
            if gt_flair is not None
            else None
        )

        if input_filename.endswith(".nii.gz"):
            base_name = input_filename[:-7]
        else:
            base_name = Path(input_filename).stem

        download_name = f"{base_name}_pred_flair_{num_inference_steps}steps.nii.gz"

        return {
            "success": True,
            "output_path": result["output_path"],
            "download_name": download_name,
            "has_ground_truth": gt_flair is not None,
            "mode": "t1-flair",
            "metrics": {
                # Per-case metrics
                "psnr": round(float(metrics["psnr"]), 4) if metrics.get("psnr") is not None else None,
                "ssim": round(float(metrics["ssim"]), 4) if metrics.get("ssim") is not None else None,

                # Dataset-level metrics
                "dataset_mean_psnr": round(float(metrics["dataset_mean_psnr"]), 4)
                if metrics.get("dataset_mean_psnr") is not None else None,

                "dataset_mean_ssim": round(float(metrics["dataset_mean_ssim"]), 4)
                if metrics.get("dataset_mean_ssim") is not None else None,

                "dataset_fid": round(float(metrics["dataset_fid"]), 4)
                if metrics.get("dataset_fid") is not None else None,

                "dataset_kid_mean": round(float(metrics["dataset_kid_mean"]), 6)
                if metrics.get("dataset_kid_mean") is not None else None,

                "dataset_kid_std": round(float(metrics["dataset_kid_std"]), 6)
                if metrics.get("dataset_kid_std") is not None else None,

                "dataset_metric_scope": metrics.get("dataset_metric_scope"),
            },
            "previews": {
                "input": t1_preview_b64,
                "generated": pred_preview_b64,
                "ground_truth": gt_preview_b64,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    finally:
        # keep output file if you want to download it later
        # remove input / gt temp files only
        cleanup_files(*(p for p in [input_path, gt_path] if p is not None))


@router.get("/download_t1_flair")
async def download_t1_flair(path: str):
    """Download a generated FLAIR NIfTI file from a previously returned path."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Output file not found.")
    return FileResponse(
        path=str(p),
        media_type="application/octet-stream",
        filename=p.name,
    )
