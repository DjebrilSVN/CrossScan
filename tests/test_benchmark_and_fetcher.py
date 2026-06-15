"""Unit tests for Phase 2: Per-Modality Baselines and Benchmark Matrix."""
from pathlib import Path
from crossscan.core.ontology import NUM_CLASSES
from crossscan.models.joint_segmentor import CrossScanJointSegmentor
from crossscan.data.fetcher import AMOS_CT_REPO, AMOS_MR_REPO


def test_baseline_architecture_without_domain_randomization():
    # Phase 2 nnU-Net style baseline has use_domain_randomization=False
    model = CrossScanJointSegmentor(
        in_channels=1,
        out_classes=NUM_CLASSES,
        feature_size=16,
        num_scales=3,
        deep_supervision=False,
        use_domain_randomization=False
    )
    assert not model.use_domain_randomization


def test_fetcher_repo_constants():
    assert "amos22-ct-dataset" in AMOS_CT_REPO
    assert "amos22-mri-dataset" in AMOS_MR_REPO

