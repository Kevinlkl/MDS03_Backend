import argparse
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, UnidentifiedImageError
import tqdm as tqdm
import SimpleITK as sitk

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def normalize_to_uint8(slice_data: np.ndarray, p_low: float=0.5, p_high: float=99.5) -> np.ndarray:
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

def apply_n4_bias_correction(volume: np.ndarray) -> np.ndarray:
    img = sitk.GetImageFromArray(volume)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(img, mask)
    return sitk.GetArrayFromImage(corrected)

def save_volume_as_png_slices(volume: np.ndarray, out_dir: Path, axis: int = 2) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    if volume.ndim == 4:
        volume = volume[..., 0]

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, but got shape {volume.shape}")

    volume = apply_n4_bias_correction(volume)

    volume = np.moveaxis(volume, axis, 2)
    num_slices = volume.shape[2]

    saved = 0
    for i in tqdm.tqdm(range(num_slices), desc="Saving slices"):
        slice_data = volume[:, :, i]
        png_data = normalize_to_uint8(slice_data)

        if is_empty_slice(png_data):
            continue

        img = Image.fromarray(png_data, mode='L')
        img = img.resize((256, 256), resample=Image.LANCZOS)
        img.save(out_dir / f"slice_{i:03d}.png")
        saved += 1

    return saved

def is_empty_slice(slice_data: np.ndarray, threshold: float = 5.0) -> bool:
    """
    Returns True if the slice is mostly empty (background).
    threshold: number of non-zero pixels needed to consider the slice non-empty
    """
    return np.count_nonzero(slice_data) < threshold


def convert_nifti_zip_to_png(input_dir: Path, output_dir: Path, axis: int=2) -> int:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist or is not a directory")
    
    zip_files = sorted(input_dir.glob("*.zip"))

    if not zip_files:
        print(f"No zip files found in {input_dir}")
        return
    
    total_nifti = 0
    total_slices = 0
    failed = 0

    for zip_file in zip_files:
        try: 
            with zipfile.ZipFile(zip_file, 'r') as zf:
                nifti_ext = [
                    m for m in zf.namelist()
                    if m.lower().endswith(('.nii', '.nii.gz'))
                ]

                if not nifti_ext:
                    print(f"No NIfTI files found in {zip_file}")
                    continue

                with tempfile.TemporaryDirectory() as temp:
                    tmp_dir = Path(temp)

                    for member in nifti_ext:
                        extracted_path = Path(zf.extract(member, path=tmp_dir))
                        nii_name = extracted_path.name
                        stem = nii_name.replace('.nii.gz', '').replace('.nii', '')

                        case_out_dir = output_dir / zip_file.stem / stem
                        img = nib.load(str(extracted_path))
                        volume = img.get_fdata()

                        saved = save_volume_as_png_slices(volume, case_out_dir, axis=axis)
                        total_nifti += 1
                        total_slices += saved
                        print(f"Converted {zip_file.name} -> {stem} ({saved} slices)")

        except Exception as e:
            failed += 1
            print(f"Failed {zip_file.name}: {e}")

    print("\nDone")
    print(f"Input folder   : {input_dir}")
    print(f"Output folder  : {output_dir}")
    print(f"NIfTI converted: {total_nifti}")
    print(f"PNG saved      : {total_slices}")
    print(f"Zip failed     : {failed}")                    

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert zipped NIfTI files to PNG slices")
    parser.add_argument(
        "--input",
        default="Dataset/T1_NiFti",
        help="Folder containing .zip files with NIfTI inside",
    )
    parser.add_argument(
        "--output",
        default="Dataset/T1_PNG",
        help="Output folder for PNG slices",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Slice axis: 0=sagittal, 1=coronal, 2=axial",
    )
    return parser.parse_args()

def print_first_n_metadata(folder: Path, n: int = 10) -> None:
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        print(f"Metadata folder not found: {folder}")
        return

    files = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )[:n]

    print(f"\nMetadata for first {len(files)} images in {folder}:")
    for i, p in enumerate(files, 1):
        try:
            stat = p.stat()
            with Image.open(p) as img:
                modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{i}] {p.name}")
                print(f"  format   : {img.format}")
                print(f"  mode     : {img.mode}")
                print(f"  size     : {img.size[0]}x{img.size[1]}")
                print(f"  file size: {stat.st_size} bytes")
                print(f"  modified : {modified}")
                print("-" * 50)
        except (UnidentifiedImageError, OSError) as exc:
            print(f"[{i}] {p.name} -> cannot read metadata: {exc}")


if __name__ == "__main__":
    args = parse_args()
    convert_nifti_zip_to_png(Path(args.input), Path(args.output), axis=args.axis)
    print("\nMetadata check:")
    print_first_n_metadata(Path("Dataset\\T1_PNG\\BraTS20_Training_001_t1.nii\\BraTS20_Training_001_t1"))
    print_first_n_metadata(Path("Dataset\\T1_PNG\\BraTS20_Training_001_t1.nii\\BraTS20_Training_002_t1"))