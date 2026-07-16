"""
Unit Tests for Phase 7: Ablation Study, Model Card, and Interactive Portfolio Gallery.
"""

from pathlib import Path
import json
import tempfile
import pytest

from crossscan.evaluation.ablation import (
    AblationStudyRunner,
    ABLATION_CONDITIONS,
    AblationCondition,
    AblationMetrics
)
from crossscan.visualization.gallery import (
    build_clinical_gallery_html,
    export_clinical_gallery
)
from crossscan.core.ontology import ORGAN_ONTOLOGY


def test_ablation_study_runner_metrics():
    runner = AblationStudyRunner(random_seed=123)
    
    # 1. Single condition test
    cond_full = [c for c in ABLATION_CONDITIONS if "Full" in c.name][0]
    metrics = runner.evaluate_condition(cond_full)
    
    assert isinstance(metrics, AblationMetrics)
    assert metrics.condition_name == cond_full.name
    
    # Check bootstrap confidence intervals
    assert metrics.ct_dice["ci_lower"] <= metrics.ct_dice["mean"] <= metrics.ct_dice["ci_upper"]
    assert metrics.mri_dice["ci_lower"] <= metrics.mri_dice["mean"] <= metrics.mri_dice["ci_upper"]
    assert abs(metrics.transfer_delta) < 0.05  # Joint model has minimal transfer drop
    assert metrics.failure_rate_pct < 5.0
    
    # 2. Conformal CRC condition test
    cond_crc = [c for c in ABLATION_CONDITIONS if "CRC" in c.name][0]
    metrics_crc = runner.evaluate_condition(cond_crc)
    assert metrics_crc.failure_rate_pct == 0.0  # Zero unflagged failures in auto-accepted cohort!
    assert metrics_crc.coverage_pct > 70.0


def test_ablation_study_export_json():
    runner = AblationStudyRunner(random_seed=42)
    results = runner.run_all()
    assert len(results) == len(ABLATION_CONDITIONS)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "ablation_results.json"
        runner.export_summary_json(results, json_path)
        assert json_path.exists()
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == len(ABLATION_CONDITIONS)
        assert "condition_name" in data[0]
        assert "ct_dice" in data[0]


def test_model_card_structural_and_clinical_integrity():
    model_card_p = Path("MODEL_CARD.md")
    assert model_card_p.exists(), "MODEL_CARD.md must exist in root repository"
    text = model_card_p.read_text(encoding="utf-8")
    
    assert len(text) > 3000, "MODEL_CARD.md must be comprehensive (>3000 chars)"
    
    # Essential sections required by Mitchell et al. & Metrics Reloaded
    required_sections = [
        "## 1. Model Details",
        "## 2. Intended Clinical Use",
        "## 3. Clinical Contraindications",
        "## 4. Multi-Center Training Data",
        "## 5. Metrics Reloaded Quantitative Evaluation",
        "## 6. Systematic Multi-Factor Ablation Findings",
        "## 7. Failure Modes Taxonomy",
        "## 8. Environmental Impact & Hardware Profiling",
        "## 9. Regulatory Disclaimer"
    ]
    for sec in required_sections:
        assert sec in text, f"Missing required section in MODEL_CARD.md: {sec}"
        
    # Check ontology terminology consistency
    assert "SNOMED CT" in text
    assert "RadLex" in text
    assert "10200004" in text  # Liver SNOMED
    assert "78961009" in text  # Spleen SNOMED


def test_clinical_gallery_generator():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_html = Path(tmpdir) / "test_gallery.html"
        res_p = export_clinical_gallery(output_html_path=out_html)
        assert res_p.exists()
        
        html = res_p.read_text(encoding="utf-8")
        assert "<canvas id=\"medicalCanvas\"" in html
        assert "Conformal Risk Control" in html
        assert "AUTO_ACCEPT" in html
        assert "FLAG_FOR_REVIEW" in html
        assert "REJECT" in html
        assert "OHIF Web Viewer" in html
        assert "3D Slicer" in html
        assert "SNOMED" in html

