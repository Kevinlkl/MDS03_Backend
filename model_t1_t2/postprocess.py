# postprocess.py

import numpy as np
import nibabel as nib
import torch


def tensor_to_numpy(pred: torch.Tensor) -> np.ndarray:
    pred = pred.detach().cpu()

    if pred.ndim == 5:
        pred = pred[0, 0]
    elif pred.ndim == 4:
        pred = pred[0]

    return pred.numpy()


def save_nifti(pred: torch.Tensor, output_path: str, affine=None):
    arr = tensor_to_numpy(pred)

    if affine is None:
        affine = np.eye(4)

    nii = nib.Nifti1Image(arr, affine)
    nib.save(nii, output_path)
    return output_path