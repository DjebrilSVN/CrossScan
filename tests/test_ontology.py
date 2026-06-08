"""Unit tests for anatomical ontology and cross-dataset label mapping."""
import pytest
import numpy as np
import torch
from crossscan.core.ontology import (
    ORGAN_ONTOLOGY,
    ORGAN_NAMES,
    NUM_CLASSES,
    harmonize_labels,
    get_present_classes_mask,
    AMOS_TO_UNIFIED,
    CHAOS_MR_TO_UNIFIED
)


def test_ontology_integrity():
    assert len(ORGAN_ONTOLOGY) == NUM_CLASSES
    assert ORGAN_ONTOLOGY[0].name == "background"
    assert ORGAN_ONTOLOGY[1].name == "liver"
    assert ORGAN_ONTOLOGY[1].snomed_ct == "10200004"
    assert ORGAN_ONTOLOGY[1].radlex == "RID58"
    assert ORGAN_ONTOLOGY[1].nsd_tolerance_mm == 2.0


def test_harmonize_labels_amos():
    # AMOS: 1 is spleen (mapped to 2), 6 is liver (mapped to 1), 7 is stomach (mapped to 6)
    amos_mask = np.array([0, 1, 6, 7, 14], dtype=np.int32)
    harmonized = harmonize_labels(amos_mask, source_dataset="amos")
    # Expected: 0 -> 0, 1 -> 2 (spleen), 6 -> 1 (liver), 7 -> 6 (stomach), 14 -> 0 (duodenum ignored)
    np.testing.assert_array_equal(harmonized, np.array([0, 2, 1, 6, 0], dtype=np.int32))


def test_harmonize_labels_torch():
    chaos_tensor = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    # CHAOS MR: 1->liver(1), 2->kidney_right(3), 3->kidney_left(4), 4->spleen(2)
    harmonized = harmonize_labels(chaos_tensor, source_dataset="chaos_mr")
    expected = torch.tensor([0, 1, 3, 4, 2], dtype=torch.long)
    assert torch.equal(harmonized, expected)


def test_present_classes_mask():
    mask = get_present_classes_mask("chaos_ct")
    # CHAOS CT only has liver (1) + background (0)
    assert mask[0] == True
    assert mask[1] == True
    assert mask[2] == False
    assert mask[3] == False

