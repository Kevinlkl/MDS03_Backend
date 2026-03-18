import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path

def load_nifti(path: str) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32), img

def save_nifti(arr: np.ndarray, ref_img: nib.Nifti1Image, out_path: str) -> None:
    new_img = nib.Nifti1Image(arr, ref_img.affine, ref_img.header)
    nib.save(new_img, out_path)

def run_n4_bias_correction(volume: np.ndarray) -> np.ndarray:
    img = sitk.GetImageFromArray(volume)
    shrink_factor = 2
    img_small = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
    mask_small = sitk.OtsuThreshold(img_small, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([20, 20, 20])
    corrector.SetConvergenceThreshold(0.001)
    corrector.Execute(img_small, mask_small)
    log_bias_field = corrector.GetLogBiasFieldAsImage(img)
    corrected = img / sitk.Exp(log_bias_field)
    return sitk.GetArrayFromImage(corrected)

def run_normalize_to_uint8(volume: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Applies slice-wise percentile normalization along the axial axis (last axis)."""

    def normalize_slice(slice_data: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(slice_data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        brain_voxels = arr[arr > 0]
        if brain_voxels.size == 0:
            return np.zeros(arr.shape, dtype=np.uint8)
        lo = float(np.percentile(brain_voxels, p_low))
        hi = float(np.percentile(brain_voxels, p_high))
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)
        return (arr * 255).astype(np.uint8)

    return np.stack(
        [normalize_slice(volume[..., i]) for i in range(volume.shape[-1])],
        axis=-1
    )  # uint8, shape (H, W, D)

if __name__ == "__main__":
    input_dir = Path("/content/dataset/training-t1-t2")
    output_dir = Path("/content/dataset/training-t1-t2-preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(list(input_dir.glob("**/*.nii*")))
    print(f"Found {len(nifti_files)} NIfTI files")

    for i, nifti_path in enumerate(nifti_files):
        relative_path = nifti_path.relative_to(input_dir)
        out_path = output_dir / relative_path

        if out_path.exists():
            print(f"[{i+1}/{len(nifti_files)}] Skipping {nifti_path.name} (already done)")
            continue

        print(f"[{i+1}/{len(nifti_files)}] Processing {nifti_path.name}...")

        try:
            volume, ref_img = load_nifti(str(nifti_path))

            print("  Starting bias correction...")
            volume_n4 = run_n4_bias_correction(volume)

            print("  Starting normalization...")
            volume_norm = run_normalize_to_uint8(volume_n4)
            volume_final = (volume_norm.astype(np.float32) / 255.0) * 2.0 - 1.0

            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_nifti(volume_final, ref_img, str(out_path))
            print(f"  Saved to {out_path}")

        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    print("Done.")