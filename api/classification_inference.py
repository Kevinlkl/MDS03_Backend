from io import BytesIO
from pathlib import Path
import shutil
import tempfile
import base64

from fastapi import APIRouter, UploadFile, File, HTTPException

import nibabel as nib
import numpy as np
from PIL import Image

from model_classification.inference import compare_two_classifiers

router = APIRouter(
    prefix="/api",
    tags=["Classification"]
)


# =========================================================
# Checkpoint Paths
# =========================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

BASELINE_MODEL_PATH = (
    BASE_DIR
    / "model_classification"
    / "checkpoints"
    / "baseline_classifier.pth"
)

SYNTHETIC_MODEL_PATH = (
    BASE_DIR
    / "model_classification"
    / "checkpoints"
    / "synthetic_aug_classifier.pth"
)
def extract_multiple_slices_as_pil(
    nifti_path,
    num_slices=15
):
    """
    Extract multiple axial slices from a NIfTI volume
    and convert them into PIL images.
    """

    nii = nib.load(nifti_path)

    image = nii.get_fdata()

    # Normalize
    image = image.astype(np.float32)

    image = (
        image - image.min()
    ) / (
        image.max() - image.min() + 1e-8
    )

    depth = image.shape[2]

    # Avoid edge slices
    start_idx = depth // 4
    end_idx = 3 * depth // 4

    # Evenly spaced slices
    slice_indices = np.linspace(
        start_idx,
        end_idx,
        num_slices,
        dtype=int
    )

    pil_images = []

    for idx in slice_indices:

        slice_img = image[:, :, idx]

        # Convert to uint8
        slice_img = (slice_img * 255).astype(np.uint8)

        # Convert to RGB PIL image
        pil_image = Image.fromarray(
            slice_img
        ).convert("RGB")

        pil_images.append(pil_image)

    return pil_images


def pil_images_to_base64(pil_images):
    """
    Convert a list of PIL images to base64-encoded JPEG strings.
    """
    base64_images = []
    for pil_img in pil_images:
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_images.append(f"data:image/jpeg;base64,{img_str}")
    return base64_images


# =========================================================
# Classification Endpoint
# =========================================================

@router.post("/classification")
async def run_classification(
    file: UploadFile = File(...)
):

    # Validate extension
    if not (
        file.filename.endswith(".nii")
        or file.filename.endswith(".nii.gz")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only .nii or .nii.gz files are supported."
        )

    try:

        # Create temp file
        suffix = Path(file.filename).suffix

        if file.filename.endswith(".nii.gz"):
            suffix = ".nii.gz"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            shutil.copyfileobj(file.file, tmp)

            temp_path = tmp.name

        # Extract multiple slices
        pil_images = extract_multiple_slices_as_pil(
            temp_path,
            num_slices=15
        )

        # Generate base64 previews
        base64_slices = pil_images_to_base64(pil_images)

        # Run both classifiers
        results = compare_two_classifiers(
            image=pil_images,
            baseline_checkpoint_path=BASELINE_MODEL_PATH,
            synthetic_aug_checkpoint_path=SYNTHETIC_MODEL_PATH
        )

        return {
            "success": True,
            "results": results,
            "previews": {
                "slices": base64_slices,
                "slice_count": len(base64_slices)
            }
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )