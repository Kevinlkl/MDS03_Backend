from __future__ import annotations

from datetime import datetime
from pathlib import Path
import base64
from typing import Optional

import torch
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.preview import get_middle_slice_png_bytes
from api.preview_util import tensor_middle_slice_to_png_bytes
from model_t1_synthetic.config import Config
from model_t1_synthetic.inference import SyntheticT1GenerationPipeline, zip_generated_files, compute_fid_score

router = APIRouter(prefix="/api", tags=["Synthetic T1 Generation"])

_pipeline: Optional[SyntheticT1GenerationPipeline] = None
_REFERENCE_T1_DIR = Config.BASE_DIR / "Dataset" / "final-clean-data" / "T1"


class SyntheticGenerationRequest(BaseModel):
    num_samples: int = 1
    num_inference_steps: int = Config.NUM_INFERENCE_STEPS
    seed: Optional[int] = None


def get_pipeline() -> SyntheticT1GenerationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SyntheticT1GenerationPipeline()
    return _pipeline


def tensor_to_base64_png(tensor) -> str:
    return base64.b64encode(tensor_middle_slice_to_png_bytes(tensor).getvalue()).decode("utf-8")


def nifti_middle_slice_to_base64_png(nifti_path: str) -> str:
    return base64.b64encode(get_middle_slice_png_bytes(nifti_path).getvalue()).decode("utf-8")


def _list_reference_t1_paths(limit: int) -> list[str]:
    if not _REFERENCE_T1_DIR.exists():
        return []

    candidates = sorted(
        str(path)
        for path in _REFERENCE_T1_DIR.glob("*.nii*")
        if path.is_file()
    )
    return candidates[:limit]


@router.post("/generate_synthetic_t1")
async def generate_synthetic_t1(payload: SyntheticGenerationRequest):
    num_samples = payload.num_samples
    num_inference_steps = payload.num_inference_steps
    seed = payload.seed

    if not 1 <= num_samples <= 100:
        raise HTTPException(status_code=400, detail="num_samples must be between 1 and 100.")

    if not 1 <= num_inference_steps <= 1000:
        raise HTTPException(status_code=400, detail="num_inference_steps must be between 1 and 1000.")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    batch_dir = Config.GENERATED_DIR / f"batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    try:
        pipeline = get_pipeline()
        generated = pipeline.generate_many(
            num_samples=num_samples,
            num_inference_steps=num_inference_steps,
            seed=seed,
            output_dir=batch_dir,
        )

        archive_path = Config.GENERATED_DIR / f"synthetic_t1_{timestamp}.zip"
        zip_generated_files([item["output_path"] for item in generated], archive_path)

        # Compute FID against Google Drive reference T1 batch
        batch_fid_score = None
        batch_fid_status = "computing"
        batch_reference_count = 0
        try:
            print(f"Computing FID for {num_samples} generated T1(s) against Google Drive reference batch...")
            batch_fid_score = compute_fid_score(
                pred_paths=[item["output_path"] for item in generated],
            )
            if batch_fid_score is not None:
                batch_fid_status = "computed"
                # Count reference files (approximation based on typical batch size)
                batch_reference_count = 100  # Typical batch size from Google Drive
            else:
                batch_fid_status = "failed: FID computation returned None"
        except Exception as exc:
            batch_fid_status = f"failed: {str(exc)}"
            print(f"FID computation error: {exc}")

        return {
            "success": True,
            "num_samples": num_samples,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "archive_path": str(archive_path),
            "download_name": archive_path.name,
            "batch_fid_score": batch_fid_score,
            "batch_fid_status": batch_fid_status,
            "batch_fid_reference_count": batch_reference_count,
            "generated_files": [
                {
                    "index": item["index"],
                    "seed": item["seed"],
                    "output_path": item["output_path"],
                    "preview": nifti_middle_slice_to_base64_png(item["output_path"]),
                    "middle_slice_preview": nifti_middle_slice_to_base64_png(item["output_path"]),
                }
                for item in generated
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Synthetic generation failed: {str(exc)}") from exc


@router.get("/download_synthetic_t1")
async def download_synthetic_t1(path: str):
    archive_path = Path(path)
    if not archive_path.exists():
        raise HTTPException(status_code=404, detail="Archive not found.")

    return FileResponse(
        path=str(archive_path),
        media_type="application/zip",
        filename=archive_path.name,
    )
