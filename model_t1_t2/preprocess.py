# preprocess.py
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


class MRIProcessor:
    def __init__(
        self,
        spatial_size: Tuple[int, int, int] = (96, 96, 96),
        pixdim: Tuple[float, float, float] = (1.5, 1.5, 1.5),
        source_key: str = "t1",
        intensity_lower: float = 0.0,
        intensity_upper: float = 99.5,
        b_min: float = -1.0,
        b_max: float = 1.0,
    ) -> None:
        self.source_key = source_key

        self.transforms = Compose([
            LoadImaged(keys=[source_key], image_only=False),
            EnsureChannelFirstd(keys=[source_key]),
            EnsureTyped(keys=[source_key], track_meta=True),
            Orientationd(keys=[source_key], axcodes="RAS"),
            Spacingd(
                keys=[source_key],
                pixdim=pixdim,
                mode=("bilinear",),
            ),
            CropForegroundd(keys=[source_key], source_key=source_key),
            SpatialPadd(keys=[source_key], spatial_size=spatial_size),
            CenterSpatialCropd(keys=[source_key], roi_size=spatial_size),
            ScaleIntensityRangePercentilesd(
                keys=[source_key],
                lower=intensity_lower,
                upper=intensity_upper,
                b_min=b_min,
                b_max=b_max,
                clip=True,
            ),
        ])

    def preprocess(self, image_path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> Dict:
        image_path = str(image_path)
        data = {self.source_key: image_path}
        output = self.transforms(data)

        img = output[self.source_key]

        if isinstance(img, MetaTensor):
            meta = dict(img.meta)
        else:
            meta = output.get(f"{self.source_key}_meta_dict", {})

        img = img.unsqueeze(0).to(device)

        return {
            "image": img,
            "meta": meta,
            "processed": output[self.source_key],
            "original_path": image_path,
        }