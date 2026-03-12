import dicom2nifti
import os
import zipfile
import shutil
import tempfile

def convert_dicom_to_nii(dicom_folder: str, output_folder: str):
    """
    Convert DICOM (.dcm) and zipped IMA (.ima) files to NIfTI format.
    
    Args:
        dicom_folder: Path to folder containing DICOM/IMA files
        output_folder: Path to output folder for NIfTI files
    """
    os.makedirs(output_folder, exist_ok=True)
    converted = 0

    for root, dirs, files in os.walk(dicom_folder):
        # Handle zipped files containing .ima files
        zip_files = [f for f in files if f.endswith('.zip')]
        for zip_file in zip_files:
            zip_path = os.path.join(root, zip_file)
            temp_dir = tempfile.mkdtemp()  # Create temp folder to extract to

            try:
                # Extract zip
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(temp_dir)
                    print(f"Extracted: {zip_file}")

                # Walk extracted contents for .ima files
                # Each subfolder (PET, CT) is treated as a separate series
                for ext_root, ext_dirs, ext_files in os.walk(temp_dir):
                    ima_files = [f for f in ext_files if f.endswith('.ima') or f.endswith('.IMA')]
                    if ima_files:
                        # Get the subfolder name (e.g., PET or CT) relative to temp_dir
                        modality_relative = os.path.relpath(ext_root, temp_dir)
                        
                        # Build output: output_folder / patient_relative_path / zip_name / PET or CT
                        relative_path = os.path.relpath(root, dicom_folder)
                        sub_output = os.path.join(
                            output_folder,
                            relative_path,
                            os.path.splitext(zip_file)[0],  # zip name as parent folder
                            modality_relative                # PET or CT subfolder
                        )
                        os.makedirs(sub_output, exist_ok=True)

                        dicom2nifti.convert_directory(ext_root, sub_output, compression=True, reorient=True)
                        print(f"✓ Converted IMA [{modality_relative}]: {zip_file}  →  {sub_output}")
                        converted += 1

            except Exception as e:
                print(f"✗ Failed: {zip_file} | Reason: {e}")
            finally:
                shutil.rmtree(temp_dir)  # Always clean up temp folder

        # Handle regular .dcm files
        dcm_files = [f for f in files if f.endswith('.dcm') or f.endswith('.DCM')]
        if dcm_files:
            relative_path = os.path.relpath(root, dicom_folder)
            sub_output = os.path.join(output_folder, relative_path)
            os.makedirs(sub_output, exist_ok=True)

            try:
                dicom2nifti.convert_directory(root, sub_output, compression=True, reorient=True)
                print(f"✓ Converted DCM: {root}  →  {sub_output}")
                converted += 1
            except Exception as e:
                print(f"✗ Failed: {root} | Reason: {e}")

    print(f"\nDone! {converted} series converted. Files saved to: {output_folder}")

if __name__ == "__main__":
    dicom_folder = r"C:\Users\jsoh2\Downloads\[68Ga]Ga-Pentixafor PETCT images of Glioma patients\[68Ga]Ga-Pentixafor PETCT images of Glioma patients\PETCT_data\24\24\PET"
    output_folder = r"C:\Users\jsoh2\Downloads\[68Ga]Ga-Pentixa for PETCT images of Glioma patients nii\PET"
    convert_dicom_to_nii(dicom_folder, output_folder)