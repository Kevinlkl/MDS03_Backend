from pathlib import Path
from typing import Dict, Tuple, Union

import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    CropForegroundd,
    SpatialPadd,
    CenterSpatialCropd,
    ScaleIntensityRangePercentilesd,
)
from monai.data import MetaTensor


PathLike = Union[str, Path]
DeviceLike = Union[str, torch.device]


class MRIProcessor:
    def __init__(
        self,
        spatial_size: Tuple[int, int, int] = (96, 96, 64),
        pixdim: Tuple[float, float, float] = (1.5, 1.5, 1.5),
        intensity_lower: float = 0.0,
        intensity_upper: float = 99.5,
        b_min: float = -1.0,
        b_max: float = 1.0,
    ) -> None:
        self.spatial_size = spatial_size
        self.pixdim = pixdim
        self.intensity_lower = intensity_lower
        self.intensity_upper = intensity_upper
        self.b_min = b_min
        self.b_max = b_max

        # Single-image transform: for inference without GT
        self.single_transforms = Compose([
            LoadImaged(keys=["t1"], image_only=False),
            EnsureChannelFirstd(keys=["t1"]),
            EnsureTyped(keys=["t1"], track_meta=True),
            Orientationd(keys=["t1"], axcodes="RAS"),
            Spacingd(
                keys=["t1"],
                pixdim=self.pixdim,
                mode=("bilinear",),
            ),
            CropForegroundd(keys=["t1"], source_key="t1"),
            SpatialPadd(keys=["t1"], spatial_size=self.spatial_size),
            CenterSpatialCropd(keys=["t1"], roi_size=self.spatial_size),
            ScaleIntensityRangePercentilesd(
                keys=["t1"],
                lower=self.intensity_lower,
                upper=self.intensity_upper,
                b_min=self.b_min,
                b_max=self.b_max,
                clip=True,
            ),
        ])

        # Paired transform: for aligned inference + evaluation with GT
        self.pair_transforms = Compose([
            LoadImaged(keys=["t1", "t2"], image_only=False),
            EnsureChannelFirstd(keys=["t1", "t2"]),
            EnsureTyped(keys=["t1", "t2"], track_meta=True),
            Orientationd(keys=["t1", "t2"], axcodes="RAS"),
            Spacingd(
                keys=["t1", "t2"],
                pixdim=self.pixdim,
                mode=("bilinear", "bilinear"),
            ),
            CropForegroundd(keys=["t1", "t2"], source_key="t1"),
            SpatialPadd(keys=["t1", "t2"], spatial_size=self.spatial_size),
            CenterSpatialCropd(keys=["t1", "t2"], roi_size=self.spatial_size),
            ScaleIntensityRangePercentilesd(
                keys=["t1", "t2"],
                lower=self.intensity_lower,
                upper=self.intensity_upper,
                b_min=self.b_min,
                b_max=self.b_max,
                clip=True,
            ),
        ])

    @staticmethod
    def _extract_meta(output: Dict, key: str) -> Dict:
        img = output[key]
        if isinstance(img, MetaTensor):
            return dict(img.meta)
        return output.get(f"{key}_meta_dict", {})

    def preprocess_input(
        self,
        image_path: PathLike,
        device: DeviceLike = "cpu",
    ) -> Dict:
        data = {"t1": str(image_path)}
        output = self.single_transforms(data)

        t1 = output["t1"].unsqueeze(0).to(device)
        meta = self._extract_meta(output, "t1")

        return {
            "t1": t1,
            "meta": meta,
            "processed_t1": output["t1"],
            "original_t1_path": str(image_path),
        }

    def preprocess_pair(
        self,
        t1_path: PathLike,
        t2_path: PathLike,
        device: DeviceLike = "cpu",
    ) -> Dict:
        data = {
            "t1": str(t1_path),
            "t2": str(t2_path),
        }
        output = self.pair_transforms(data)

        t1 = output["t1"].unsqueeze(0).to(device)
        t2 = output["t2"].unsqueeze(0).to(device)

        t1_meta = self._extract_meta(output, "t1")
        t2_meta = self._extract_meta(output, "t2")

        return {
            "t1": t1,
            "t2": t2,
            "t1_meta": t1_meta,
            "t2_meta": t2_meta,
            "processed_t1": output["t1"],
            "processed_t2": output["t2"],
            "original_t1_path": str(t1_path),
            "original_t2_path": str(t2_path),
        }