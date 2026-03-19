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
import cv2

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def normalize_to_uint8(slice_data: np.ndarray, p_low: float=0.5, p_high: float=99.5) -> np.ndarray:
    """
    Normalise the slice data to uint8 (0-255) using percentile clipping where
    p_low and p_high define the percentiles for clipping to reduce the effect of outliers,
    and apply CLAHE for enhanced constrast.

    args:
    slice_data: 2D numpy array of the slice
    p_low: lower percentile for clipping (default 0.5)
    p_high: upper percentile for clipping (default 99.5)

    returns:
    2D numpy array of type uint8 with values in range [0, 255]
    """
    # Replace NaN and inf values with 0 before processing
    arr = np.nan_to_num(slice_data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    brain_voxels = arr[arr > 0]
    if brain_voxels.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    # Use percentiles to determine the clipping range
    lo = float(np.percentile(brain_voxels, p_low))
    hi = float(np.percentile(brain_voxels, p_high))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    # Clip and normalise to 0-255
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    arr_uint8 = (arr * 255).astype(np.uint8)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(arr_uint8)

def apply_n4_bias_correction(volume: np.ndarray) -> np.ndarray:
    """
    Apply N4 bias field correction to the volume.

    args:
    volume: 3D numpy array of the brain scan

    returns:
    3D numpy array of the bias-corrected volume
    """
    img = sitk.GetImageFromArray(volume)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(img, mask)
    return sitk.GetArrayFromImage(corrected)

def save_volume_as_png_slices(volume: np.ndarray, out_dir: Path, axis: int = 2) -> int:
    """
    Saves a 3D volume as PNG slices along the specified axis after applying N4 
    bias correction and normalisation.

    args:
    volume: 3D numpy array of the brain scan
    out_dir: Path to the output directory where PNG slices will be saved
    axis: int, the axis along which to slice the volume (0=sagittal, 1=coronal, 2=axial)

    returns:
    int: number of slices saved
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if volume.ndim == 4:
        volume = volume[..., 0]

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, but got shape {volume.shape}")

    # Apply N4 bias correction to the entire volume before slicing
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
        # Resize to 256x256 using LANCZOS resampling for better quality
        img = img.resize((256, 256), resample=Image.LANCZOS)
        # Save with zero-padded slice index
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
    """
    Convert zipped NIfTI files to PNG slices. The function expects the input directory to contain .zip files,
    each containing one or more NIfTI files (.nii or .nii.gz). It extracts the NIfTI files, applies N4 bias correction,
    normalises the slices, and saves them as PNG images in an output directory that mirrors the input structure.

    args: 
    input_dir: Path to the directory containing .zip files with NIfTI files inside
    output_dir: Path to the directory where PNG slices will be saved
    axis: int, the axis along which to slice the volume (0=sagittal, 1=coronal, 2=axial)

    returns:
    int: total number of PNG slices saved
    """
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
                # Look for NIfTI files in the zip
                nifti_ext = [
                    m for m in zf.namelist()
                    if m.lower().endswith(('.nii', '.nii.gz'))
                ]

                if not nifti_ext:
                    print(f"No NIfTI files found in {zip_file}")
                    continue

                # Extract NIfTI files to a temporary directory for processing
                with tempfile.TemporaryDirectory() as temp:
                    tmp_dir = Path(temp)

                    # Process each NIfTI file found in the zip
                    for member in nifti_ext:
                        extracted_path = Path(zf.extract(member, path=tmp_dir))
                        nii_name = extracted_path.name
                        stem = nii_name.replace('.nii.gz', '').replace('.nii', '')

                        case_out_dir = output_dir / zip_file.stem / stem
                        # Load the NIfTI file and get the volume data
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

def convert_nifti_dir_to_png(input_dir: Path, output_dir: Path, axis: int=2) -> int:
    """
    Convert nested NIfTI files to PNG slices. The function expects the input directory 
    to contain subfolders (e.g. train, val) with NIfTI files (.nii or .nii.gz). 
    It applies N4 bias correction, normalises the slices, and saves them as PNG images 
    in an output directory that mirrors the input structure.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist or is not a directory")
    
    # Recursively find all NIfTI files
    nifti_files = list(input_dir.rglob("*.nii")) + list(input_dir.rglob("*.nii.gz"))
    # Remove duplicates if rglob matches .nii.gz for both
    nifti_files = list(set(nifti_files))

    if not nifti_files:
        print(f"No NIfTI files found in {input_dir} or its subdirectories.")
        return
    
    total_nifti = 0
    total_slices = 0
    failed = 0

    for nifti_file in nifti_files:
        try: 
            # Recreate the folder structure in the output directory
            rel_path = nifti_file.relative_to(input_dir)
            stem = nifti_file.name.replace('.nii.gz', '').replace('.nii', '')
            
            case_out_dir = output_dir / rel_path.parent / stem
            
            # Load the NIfTI file and get the volume data
            img = nib.load(str(nifti_file))
            volume = img.get_fdata()

            saved = save_volume_as_png_slices(volume, case_out_dir, axis=axis)
            total_nifti += 1
            total_slices += saved
            print(f"Converted {rel_path} -> ({saved} slices)")

        except Exception as e:
            failed += 1
            print(f"Failed {nifti_file}: {e}")

    print("\nDone")
    print(f"Input folder   : {input_dir}")
    print(f"Output folder  : {output_dir}")
    print(f"NIfTI converted: {total_nifti}")
    print(f"PNG saved      : {total_slices}")
    print(f"Failed files   : {failed}")                 

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
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
    """
    Prints metadata for the first n image files in the specified folder. Metadata 
    includes format, mode, size, file size, and last modified date.

    args:
    folder: Path to the folder containing image files
    n: number of files to print metadata for (default 10)
    """
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
    # convert_nifti_zip_to_png(Path(args.input), Path(args.output), axis=args.axis)
    convert_nifti_dir_to_png(Path(args.input), Path(args.output), axis=args.axis)
    # print("\nMetadata check:")
    # print_first_n_metadata(Path("Dataset\\T1_PNG\\BraTS20_Training_001_t1.nii\\BraTS20_Training_001_t1"))
    # print_first_n_metadata(Path("Dataset\\T1_PNG\\BraTS20_Training_002_t1.nii\\BraTS20_Training_002_t1"))