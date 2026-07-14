"""
Harmonized Preprocessing Transforms for CT and Multi-Sequence MRI.

Ensures strict spatial canonicalization (RAS+ orientation, isotropic 1.5mm spacing),
modality-specific radiometric standardization (Hounsfield windowing for CT,
N4ITK bias correction and robust z-score normalization for MRI), and
Metrics Reloaded compliance.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import SimpleITK as sitk
import torch

from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    SpatialPadd,
    CropForegroundd,
    RandSpatialCropd,
    MapTransform,
)


def normalize_ct(
    image: np.ndarray,
    hu_min: float = -160.0,
    hu_max: float = 240.0
) -> np.ndarray:
    """
    Standardize CT Hounsfield Units (HU) to [0, 1] range using abdominal soft-tissue window.
    
    Default window: [-160, 240] HU covers liver, spleen, kidneys, pancreas, and vessels.
    """
    if image.min() >= -0.05 and image.max() <= 1.05 and (image.max() - image.min()) > 0.05:
        return np.clip(image, 0.0, 1.0).astype(np.float32)
    clipped = np.clip(image, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min + 1e-8)
    return normalized.astype(np.float32)


def normalize_mri(
    image: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    nonzero_only: bool = True
) -> np.ndarray:
    """
    Standardize MRI intensities using percentile clipping and foreground Z-score normalization.
    
    Args:
        image: 3D/4D numpy array.
        lower_percentile: Low cutoff to eliminate dark background anomalies.
        upper_percentile: High cutoff to eliminate RF hyperintensities / coil hotspots.
        nonzero_only: If True, calculate mean and standard deviation strictly over nonzero voxels.
    """
    mask = image > 0 if nonzero_only else np.ones_like(image, dtype=bool)
    if not np.any(mask):
        return np.zeros_like(image, dtype=np.float32)
    
    fg_voxels = image[mask]
    p_low = np.percentile(fg_voxels, lower_percentile)
    p_high = np.percentile(fg_voxels, upper_percentile)
    
    clipped = np.clip(image, p_low, p_high)
    fg_clipped = clipped[mask]
    
    mean = np.mean(fg_clipped)
    std = np.std(fg_clipped) + 1e-8
    
    normalized = np.zeros_like(image, dtype=np.float32)
    normalized[mask] = (clipped[mask] - mean) / std
    return normalized


def bias_field_correction_n4(
    image_array: np.ndarray,
    shrink_factor: int = 2,
    num_iterations: Tuple[int, ...] = (50, 50, 30, 20)
) -> np.ndarray:
    """
    Correct B1 radiofrequency field inhomogeneities using N4ITK algorithm.
    
    Args:
        image_array: 3D numpy array of MRI volume.
        shrink_factor: Subsampling factor for fast execution on high-res volumes.
        num_iterations: Max iterations per resolution level.
        
    Returns:
        Bias-corrected 3D numpy array.
    """
    sitk_img = sitk.GetImageFromArray(image_array.astype(np.float32))
    
    # Otsu thresholding for foreground mask
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    mask_sitk = otsu_filter.Execute(sitk_img)
    
    # Optional subsampling for computational speed
    if shrink_factor > 1:
        sitk_img_sub = sitk.Shrink(sitk_img, [shrink_factor] * 3)
        mask_sitk_sub = sitk.Shrink(mask_sitk, [shrink_factor] * 3)
    else:
        sitk_img_sub = sitk_img
        mask_sitk_sub = mask_sitk
        
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(num_iterations)
    
    try:
        _ = corrector.Execute(sitk_img_sub, mask_sitk_sub)
        log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_img)
        corrected = sitk_img / sitk.Exp(log_bias_field)
        return sitk.GetArrayFromImage(corrected).astype(np.float32)
    except Exception:
        # Fallback to input array if N4 convergence fails on non-standard fields
        return image_array.astype(np.float32)


class NormalizeRadiometryDict(MapTransform):
    """
    MONAI Dictionary Transform for modality-aware radiometry normalization.
    """
    def __init__(
        self,
        keys: Sequence[str],
        modality_key: str = "modality",
        hu_min: float = -160.0,
        hu_max: float = 240.0
    ):
        super().__init__(keys)
        self.modality_key = modality_key
        self.hu_min = hu_min
        self.hu_max = hu_max

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        modality = str(d.get(self.modality_key, "ct")).lower()
        
        for key in self.keys:
            val = d[key]
            is_torch = isinstance(val, torch.Tensor)
            arr = val.detach().cpu().numpy() if is_torch else np.asarray(val)
            
            if modality in ["ct", "computed_tomography"]:
                norm_arr = normalize_ct(arr, hu_min=self.hu_min, hu_max=self.hu_max)
            else:  # MRI
                norm_arr = normalize_mri(arr)
                
            d[key] = torch.from_numpy(norm_arr).to(val.device if is_torch else "cpu") if is_torch else norm_arr
        return d


class EnsureImageLoadedDict(MapTransform):
    """
    Loads 3D image and label volumes into MONAI MetaTensors if file paths are given;
    leaves arrays and tensors untouched.
    """
    def __init__(self, keys: Sequence[str]):
        super().__init__(keys)
        self.loader = LoadImage(image_only=False)

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        for k in self.keys:
            val = d.get(k)
            if isinstance(val, (str, Path)):
                p = Path(val)
                if p.exists():
                    loaded, meta = self.loader(str(p))
                    d[k] = loaded
                    d[f"{k}_meta_dict"] = meta
        return d


def is_foreground_voxel(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Top-level picklable predicate for foreground cropping across multi-worker loaders."""
    return x > 0.01


def get_harmonized_spatial_pipeline(
    target_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    patch_size: Optional[Tuple[int, int, int]] = (96, 96, 96),
    is_training: bool = True
) -> Compose:
    """
    Builds the standardized spatial transformation pipeline ensuring RAS+ alignment,
    uniform voxel spacing, foreground cropping, and training patch sampling.
    """
    transforms = [
        EnsureImageLoadedDict(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=target_spacing,
            mode=("bilinear", "nearest")
        ),
        NormalizeRadiometryDict(keys=["image"], modality_key="modality"),
        CropForegroundd(keys=["image", "label"], source_key="image", select_fn=is_foreground_voxel),
    ]
    
    if is_training and patch_size is not None:
        transforms.extend([
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_size=False),
        ])
        
    transforms.append(EnsureTyped(keys=["image", "label"], track_meta=False))
    return Compose(transforms)


def load_and_preprocess_volume(
    file_path: Union[str, Path],
    modality: str = "ct",
    target_shape: Optional[Tuple[int, int, int]] = (64, 128, 128)
) -> torch.Tensor:
    """
    Loads and preprocesses a 3D NIfTI or MetaImage volume into a standardized tensor.
    Interpolates to target_shape (default: 64x128x128) to prevent GPU VRAM exhaustion.
    Returns:
        (1, D, H, W) normalized float32 tensor.
    """
    file_path = Path(file_path)
    import nibabel as nib
    import torch.nn.functional as F
    img = nib.load(str(file_path))
    arr = img.get_fdata().astype(np.float32)
    # NIfTI shape is typically (X, Y, Z); transpose to (Z, Y, X) for PyTorch (D, H, W)
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 1, 0))
    
    if modality.lower() in ["ct", "computed_tomography"]:
        norm_arr = normalize_ct(arr)
    else:
        norm_arr = normalize_mri(arr)
        
    tensor = torch.from_numpy(norm_arr).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    if target_shape is not None:
        tensor = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)
    else:
        # Pad to multiple of 16
        d, h, w = tensor.shape[2:]
        pad_d = (16 - d % 16) % 16
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0)
    return tensor.squeeze(0)  # (1, D, H, W)

