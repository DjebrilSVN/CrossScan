"""
Volumetric DataModule and DataLoader Pipeline for CrossScan.

Supports multi-dataset loading (AMOS, TotalSegmentator, CHAOS), dynamic modality routing,
online physics domain randomization, masked annotation tracking, and patient stratification.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from monai.data import CacheDataset, PersistentDataset
from monai.transforms import Compose

from crossscan.core.ontology import NUM_CLASSES, get_present_classes_mask
from crossscan.data.transforms import get_harmonized_spatial_pipeline
from crossscan.data.domain_randomization import DomainRandomization3D


class CrossScanDataset(Dataset):
    """
    Volumetric dataset that loads NIfTI/DICOM pairs, tracks present anatomical classes,
    and applies modality-specific harmonization transforms.
    """
    def __init__(
        self,
        data_manifest: List[Dict[str, Union[str, int]]],
        transform: Optional[Compose] = None,
        is_training: bool = True,
        use_domain_randomization: bool = True
    ):
        self.data_manifest = data_manifest
        self.transform = transform
        self.is_training = is_training
        self.use_domain_randomization = use_domain_randomization
        self.randomizer = DomainRandomization3D() if (is_training and use_domain_randomization) else None

    def __len__(self) -> int:
        return len(self.data_manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.data_manifest[idx]
        image_path = entry["image"]
        label_path = entry.get("label", "")
        modality = entry.get("modality", "ct").lower()
        dataset_name = entry.get("dataset", "amos").lower()
        patient_id = entry.get("patient_id", f"case_{idx:04d}")
        
        # Load item dictionary
        item = {
            "image": image_path,
            "label": label_path if label_path and Path(label_path).exists() else np.zeros((1, 1, 1), dtype=np.int16),
            "modality": modality,
            "dataset": dataset_name,
            "patient_id": patient_id
        }
        
        # Apply MONAI spatial and radiometry transforms
        if self.transform is not None:
            item = self.transform(item)
            
        # Extract present classes mask for masked marginal loss
        present_mask = get_present_classes_mask(dataset_name, num_classes=NUM_CLASSES)
        item["present_mask"] = present_mask
        
        # Apply domain randomization during training if enabled
        if self.is_training and self.randomizer is not None and isinstance(item["image"], torch.Tensor):
            item["image"] = self.randomizer(item["image"])
            
        return item


def collate_crossscan_batch(batch: List[Dict[str, Union[torch.Tensor, str]]]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    Custom collate function that stacks volumetric images, labels, and present masks
    while preserving string metadata (patient_id, modality, dataset).
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    
    # Check if labels are present
    has_labels = all("label" in item and isinstance(item["label"], torch.Tensor) for item in batch)
    if has_labels:
        labels = torch.stack([item["label"] for item in batch], dim=0)
    else:
        labels = None
        
    present_masks = torch.stack([item["present_mask"] for item in batch], dim=0)
    
    return {
        "image": images,
        "label": labels,
        "present_mask": present_masks,
        "modality": [item["modality"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
        "patient_id": [item["patient_id"] for item in batch],
    }


class CrossScanDataModule:
    """
    Orchestrates DataLoaders for Training, Validation, and External Testing.
    """
    def __init__(
        self,
        splits_manifest_path: Union[str, Path],
        target_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        batch_size: int = 2,
        num_workers: int = 0 if os.name == "nt" else 2,
        use_cache: bool = False,
        cache_rate: float = 1.0,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        self.splits_manifest_path = Path(splits_manifest_path)
        self.target_spacing = target_spacing
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_cache = use_cache
        self.cache_rate = cache_rate
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        self.splits: Dict[str, List[Dict[str, str]]] = {}
        self.load_splits()
        
        # Pipelines
        self.train_transform = get_harmonized_spatial_pipeline(
            target_spacing=target_spacing, patch_size=patch_size, is_training=True
        )
        self.val_transform = get_harmonized_spatial_pipeline(
            target_spacing=target_spacing, patch_size=None, is_training=False
        )

    def load_splits(self) -> None:
        if not self.splits_manifest_path.exists():
            raise FileNotFoundError(f"Splits manifest not found at {self.splits_manifest_path}")
        with open(self.splits_manifest_path, "r") as f:
            self.splits = json.load(f)

    def get_train_dataloader(self) -> DataLoader:
        train_entries = self.splits.get("train", [])
        dataset = CrossScanDataset(
            data_manifest=train_entries,
            transform=self.train_transform,
            is_training=True,
            use_domain_randomization=True
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_crossscan_batch,
            pin_memory=torch.cuda.is_available()
        )

    def get_val_dataloader(self) -> DataLoader:
        val_entries = self.splits.get("val", [])
        dataset = CrossScanDataset(
            data_manifest=val_entries,
            transform=self.val_transform,
            is_training=False,
            use_domain_randomization=False
        )
        return DataLoader(
            dataset,
            batch_size=1,  # Full volume evaluation
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_crossscan_batch
        )

    def get_test_dataloader(self) -> DataLoader:
        test_entries = self.splits.get("test", [])
        dataset = CrossScanDataset(
            data_manifest=test_entries,
            transform=self.val_transform,
            is_training=False,
            use_domain_randomization=False
        )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_crossscan_batch
        )

