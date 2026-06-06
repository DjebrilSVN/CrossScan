"""Core configurations, ontologies, and typing for CrossScan."""
from crossscan.core.ontology import (
    ORGAN_ONTOLOGY,
    ORGAN_NAMES,
    ORGAN_INDEX_TO_NAME,
    ORGAN_NAME_TO_INDEX,
    DATASET_LABEL_MAPS,
    harmonize_labels,
    get_present_classes_mask
)

__all__ = [
    "ORGAN_ONTOLOGY",
    "ORGAN_NAMES",
    "ORGAN_INDEX_TO_NAME",
    "ORGAN_NAME_TO_INDEX",
    "DATASET_LABEL_MAPS",
    "harmonize_labels",
    "get_present_classes_mask",
]

