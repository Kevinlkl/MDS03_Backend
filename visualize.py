"""Quick NIfTI visualization helper for synthetic_t1_00999.nii.

Usage:
	python visualize.py
	python visualize.py synthetic_t1_00999.nii --save preview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


DEFAULT_NIFTI_DIR = Path(__file__).parent / "generated"


def resolve_default_nifti_path() -> Path | None:
	"""Return the first NIfTI found in the project's `generated/` folder.

	Searches for any file with a `.nii` or `.nii.gz` extension (recursive).
	Returns `None` if no candidate is found.
	"""
	# Primary generated folder next to this script
	generated_dir = DEFAULT_NIFTI_DIR
	if generated_dir.exists():
		candidates = sorted(generated_dir.glob("**/*.nii*"))
		if candidates:
			return candidates[0]

	# Fallback: older location used in the project
	fallback_dir = Path(__file__).with_name("generated3") / "generated"
	if fallback_dir.exists():
		candidates = sorted(fallback_dir.glob("**/*.nii*"))
		if candidates:
			return candidates[0]

	return None


def load_volume(nifti_path: Path) -> np.ndarray:
	"""Load a 3D or 4D NIfTI volume and return a 3D numpy array."""
	image = nib.load(str(nifti_path))
	data = image.get_fdata().astype(np.float32)

	if data.ndim == 4:
		data = data[..., 0]

	if data.ndim != 3:
		raise ValueError(f"Expected 3D or 4D NIfTI, got shape {data.shape}")

	return np.nan_to_num(data)


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
	"""Normalize a 2D slice to the 0-1 range for display."""
	min_val = float(slice_2d.min())
	max_val = float(slice_2d.max())

	if max_val <= min_val:
		return np.zeros_like(slice_2d, dtype=np.float32)

	return (slice_2d - min_val) / (max_val - min_val)


def get_axial_slices(volume: np.ndarray) -> list[int]:
    """Return evenly-spaced slice indices from 15% to 85% of volume depth."""
    num_slices = 30
    depth = volume.shape[2]

    start = int(depth * 0.15)
    end = int(depth * 0.85)

    slice_indices = np.linspace(start, end - 1, num_slices).astype(int)
    return slice_indices.tolist()
		


def plot_volume_grid(volume: np.ndarray, title: str, save_path: Path | None = None) -> None:
    """Display axial slices in a 5x6 grid, matching the Evaluating.ipynb approach."""
    slice_indices = get_axial_slices(volume)

    # Normalize volume for display
    vol_min = float(volume.min())
    vol_max = float(volume.max())

    if vol_max > vol_min:
        volume_norm = (volume - vol_min) / (vol_max - vol_min)
    else:
        volume_norm = np.zeros_like(volume, dtype=np.float32)

    rows, cols = 5, 6
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    fig.suptitle(title, fontsize=12, weight="normal")

    axes_flat = axes.ravel()

    for ax, slice_idx in zip(axes_flat, slice_indices):
        slice_2d = volume_norm[:, :, slice_idx]
        ax.imshow(np.rot90(slice_2d), cmap="gray")
        ax.set_title(str(slice_idx), fontsize=8)
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    plt.show()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Visualize a NIfTI volume.")
	default_nifti = resolve_default_nifti_path()
	parser.add_argument(
		"nifti_path",
		nargs="?",
		default=str(default_nifti) if default_nifti is not None else None,
		help="Path to the .nii or .nii.gz file.",
	)
	parser.add_argument(
		"--save",
		dest="save_path",
		default=None,
		help="Optional path to save the rendered figure as an image.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	nifti_path = Path(args.nifti_path)

	if not nifti_path.exists():
		raise FileNotFoundError(f"Could not find NIfTI file: {nifti_path}")

	volume = load_volume(nifti_path)
	output_path = Path(args.save_path) if args.save_path else None
	plot_volume_grid(volume, title=nifti_path.name, save_path=output_path)


if __name__ == "__main__":
	main()
