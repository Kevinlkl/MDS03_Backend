"""FastAPI routes for brain-tumor classification.

The endpoint converts an uploaded NIfTI volume into representative 2D slices,
runs both classifier checkpoints, and returns predictions plus image previews.
"""

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
    Function description:
        Extract multiple axial slices from a NIfTI volume and convert them into PIL images.

    Parameters:
        nifti_path (str | Path): Path to the NIfTI volume.
        num_slices (int): Number of axial slices to extract from the central region.

    Returns:
        list[Image.Image]: Extracted slices as RGB PIL images.
    """

    # Load the MRI volume from disk and convert it to a NumPy array.
    nii = nib.load(nifti_path)

    image = nii.get_fdata()

    # Normalize
    image = image.astype(np.float32)

    # Scale intensity values into [0, 1] so the classifier receives stable inputs.
    image = (
        image - image.min()
    ) / (
        image.max() - image.min() + 1e-8
    )

    depth = image.shape[2]

    # Avoid edge slices because they are usually less informative for classification.
    start_idx = depth // 4
    end_idx = 3 * depth // 4

    # Evenly spaced slices summarize the center of the 3D scan.
    slice_indices = np.linspace(
        start_idx,
        end_idx,
        num_slices,
        dtype=int
    )

    pil_images = []

    for idx in slice_indices:

        slice_img = image[:, :, idx]

        # Convert normalized grayscale values to image pixels.
        slice_img = (slice_img * 255).astype(np.uint8)

        # Convert to RGB PIL image
        pil_image = Image.fromarray(
            slice_img
        ).convert("RGB")

        pil_images.append(pil_image)

    return pil_images


def pil_images_to_base64(pil_images):
    """
    Function description:
        Convert a list of PIL images to base64-encoded JPEG strings.

    Parameters:
        pil_images (list[Image.Image]): PIL images to encode.

    Returns:
        list[str]: Data-URL JPEG strings for frontend previews.
    """
    base64_images = []
    for pil_img in pil_images:
        # JPEG keeps preview payloads smaller than PNG for multi-slice responses.
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
    """
    Function description:
        Classify an uploaded MRI volume using baseline and synthetic-augmented models.

    Parameters:
        file (UploadFile): Uploaded .nii or .nii.gz MRI volume.

    Returns:
        dict: Classification results and base64 preview slices.
    """

    # Validate extension before writing the upload to a temporary file.
    if not (
        file.filename.endswith(".nii")
        or file.filename.endswith(".nii.gz")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only .nii or .nii.gz files are supported."
        )

    try:

        # Create temp file for nibabel, which works with filesystem paths.
        suffix = Path(file.filename).suffix

        if file.filename.endswith(".nii.gz"):
            suffix = ".nii.gz"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            shutil.copyfileobj(file.file, tmp)

            temp_path = tmp.name

        # Extract multiple axial slices to classify the 3D scan as 2D images.
        pil_images = extract_multiple_slices_as_pil(
            temp_path,
            num_slices=15
        )

        # Generate base64 previews for display in the frontend.
        base64_slices = pil_images_to_base64(pil_images)

        # Run both checkpoints so the UI can compare baseline vs synthetic-augmented.
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

        # Return any preprocessing/model failure as a standard API error response.
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
