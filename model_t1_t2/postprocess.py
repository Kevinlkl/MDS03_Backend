import numpy as np
import nibabel as nib
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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
    
# =========================
# Metric computation (CLEAN)
# =========================
def compute_psnr_ssim(pred_np: np.ndarray, gt_np: np.ndarray):
    psnr_list = []
    ssim_list = []

    B = pred_np.shape[0]
    print(B)
    for i in range(B):
        pred_vol = pred_np[i, 0]
        gt_vol = gt_np[i, 0]

        pred_vol = np.clip(pred_vol, -1.0, 1.0)
        gt_vol = np.clip(gt_vol, -1.0, 1.0)

        pred_vol = (pred_vol + 1.0) / 2.0
        gt_vol = (gt_vol + 1.0) / 2.0

        # PSNR (3D)
        psnr_val = psnr(gt_vol, pred_vol, data_range=1.0)

        # SSIM (slice-wise)
        ssim_slices = []
        for z in range(pred_vol.shape[0]):
            ssim_val = ssim(
                gt_vol[z],
                pred_vol[z],
                data_range=1.0
            )
            ssim_slices.append(ssim_val)

        ssim_val = float(np.mean(ssim_slices))
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


# =========================
# MAIN evaluation
# =========================
def evaluate_batch(pred_t2, gt_t2=None):
    pred_np = pred_t2.detach().cpu().numpy()

    results = {
        "psnr": None,
        "ssim": None,
        "fid": None  # removed (not valid for MRI)
    }

    if gt_t2 is None:
        return results

    gt_np = gt_t2.detach().cpu().numpy()

    try:
        psnr_val, ssim_val = compute_psnr_ssim(pred_np, gt_np)

        results["psnr"] = float(psnr_val) if np.isfinite(psnr_val) else None
        results["ssim"] = float(ssim_val) if np.isfinite(ssim_val) else None

    except Exception as e:
        print("Metric computation failed:", e)

    return results