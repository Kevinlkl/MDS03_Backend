import dicom2nifti
import dicom2nifti.settings
import os
import pydicom
import numpy as np
from PIL import Image

def convert_dicom_to_nii(dicom_folder: str, output_folder: str):
    """
    Convert DICOM (.dcm) or IMA (.ima) files to NIfTI format.
    Expected structure:
        dicom_folder/
            any_folder/   ← .dcm or .ima files

    Args:
        dicom_folder: Path to root folder containing DICOM/IMA files or subfolders
        output_folder: Path to output folder for NIfTI files
    """
    os.makedirs(output_folder, exist_ok=True)
    converted = 0
    failed = 0

    # Disable strict validation for Siemens IMA files
    dicom2nifti.settings.disable_validate_slice_increment()

    for root, dirs, files in os.walk(dicom_folder):
        # Check for .dcm or .ima files in current folder
        dicom_files = [f for f in files if f.lower().endswith('.dcm') or f.lower().endswith('.ima')]
        if not dicom_files:
            continue

        # Mirror folder structure in output
        relative_path = os.path.relpath(root, dicom_folder)
        sub_output = os.path.join(output_folder, relative_path)
        os.makedirs(sub_output, exist_ok=True)
        try:
            dicom2nifti.convert_directory(root, sub_output, compression=False, reorient=True)
            print(f"✓ Converted: {root}  →  {sub_output} ({len(dicom_files)} files)")
            converted += 1
        except Exception as e:
            print(f"✗ Failed: {root} | Reason: {e}")
            failed += 1

    print(f"\nDone! {converted} series converted, {failed} failed. Files saved to: {output_folder}")


def convert_dicom_to_png(dicom_folder: str, output_folder: str):
    """
    Convert DICOM (.dcm) or IMA (.ima) files to PNG format.
    Expected structure:
        dicom_folder/
            any_folder/   ← .dcm or .ima files

    Args:
        dicom_folder: Path to root folder containing DICOM/IMA files or subfolders
        output_folder: Path to output folder for PNG files
    """
    os.makedirs(output_folder, exist_ok=True)
    converted = 0
    failed = 0

    for root, dirs, files in os.walk(dicom_folder):
        dicom_files = [f for f in files if f.lower().endswith('.dcm') or f.lower().endswith('.ima')]
        if not dicom_files:
            continue

        # Mirror folder structure in output
        relative_path = os.path.relpath(root, dicom_folder)
        sub_output = os.path.join(output_folder, relative_path)
        os.makedirs(sub_output, exist_ok=True)

        for dicom_file in dicom_files:
            file_path = os.path.join(root, dicom_file)
            try:
                ds = pydicom.dcmread(file_path)
                pixel_array = ds.pixel_array.astype(np.float32)

                # Normalise pixel values to 0-255
                pixel_min = pixel_array.min()
                pixel_max = pixel_array.max()
                if pixel_max > pixel_min:
                    pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min) * 255.0
                else:
                    pixel_array = np.zeros_like(pixel_array)

                image = Image.fromarray(pixel_array.astype(np.uint8))

                # Save as PNG with same base filename
                base_name = os.path.splitext(dicom_file)[0]
                output_path = os.path.join(sub_output, f"{base_name}.png")
                image.save(output_path)

                print(f"✓ Converted: {file_path}  →  {output_path}")
                converted += 1
            except Exception as e:
                print(f"✗ Failed: {file_path} | Reason: {e}")
                failed += 1

    print(f"\nDone! {converted} files converted, {failed} failed. Files saved to: {output_folder}")


if __name__ == "__main__":
    input_folder = r"" # insert path inside quotes
    output_folder = r"" # insert path inside quotes
    # Uncomment function to use
    # convert_dicom_to_nii(input_folder, output_folder)
    # convert_dicom_to_png(input_folder, output_folder)
