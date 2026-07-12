"""
Unit Tests for Phase 6 Clinical Deployment, Orthanc DICOMweb, MONAI Deploy, and ONNX Profiling.
"""

from pathlib import Path
import tempfile
import pytest
import numpy as np
import torch
import onnx

from crossscan.models.joint_segmentor import CrossScanJointSegmentor
from crossscan.core.ontology import NUM_CLASSES
from crossscan.deployment.orthanc_client import OrthancDICOMwebClient
from crossscan.deployment.monai_deploy import (
    HarmonizationOperator,
    CrossScanInferenceOperator,
    ConformalTriageOperator
)
from crossscan.deployment.profiler import InferenceProfiler
from crossscan.reliability.conformal import ConformalTriageEngine


def test_orthanc_dicomweb_client():
    client = OrthancDICOMwebClient(base_url="http://localhost:8042")
    
    # Check multipart body formatting
    sample_bytes = b"MOCK_DICOM_CONTENT"
    body, c_type = client.create_stow_multipart_payload(sample_bytes)
    assert b"Content-Type: application/dicom" in body
    assert sample_bytes in body
    assert "multipart/related" in c_type
    
    # Check viewer hooks
    study_uid = "1.2.3.4.5.6.7"
    ohif_url = client.get_ohif_viewer_url(study_uid)
    assert f"StudyInstanceUIDs={study_uid}" in ohif_url
    
    slicer_cmd = client.get_slicer_command("C:/data/dicom", "C:/data/seg.dcm")
    assert "Slicer.exe" in slicer_cmd
    
    # Health check returns offline status gracefully when server not running
    health = client.check_health()
    assert "is_online" in health


def test_monai_deploy_operator_chain():
    dev = torch.device("cpu")
    model = CrossScanJointSegmentor(in_channels=1, out_classes=NUM_CLASSES, feature_size=8, num_scales=3).to(dev)
    
    # 1. Harmonization Operator
    harm_op = HarmonizationOperator()
    vol = np.ones((16, 24, 24), dtype=np.float32)
    data = {"volume": vol}
    data = harm_op.compute(data, modality="ct")
    assert "tensor" in data
    assert data["tensor"].shape == (1, 1, 16, 32, 32)  # Padded to multiple of 16
    
    # 2. Inference Operator
    infer_op = CrossScanInferenceOperator(model, dev)
    data = infer_op.compute(data)
    assert "pred_mask" in data
    assert data["pred_mask"].shape == (16, 24, 24)  # Restored to original unpadded spatial dimensions
    
    # 3. Conformal Triage Operator
    engine = ConformalTriageEngine(target_risk_alpha=0.10)
    triage_op = ConformalTriageOperator(triage_engine=engine)
    data = triage_op.compute(data, case_id="Test_Patient_001")
    assert "triage_decision" in data
    assert data["triage_decision"].action in ["AUTO_ACCEPT", "FLAG_FOR_REVIEW", "REJECT"]


def test_inference_profiler_and_onnx_export():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossScanJointSegmentor(in_channels=1, out_classes=NUM_CLASSES, feature_size=8, num_scales=3).to(dev)
    
    profiler = InferenceProfiler(model=model, device=dev)
    
    # 1. Latency & VRAM benchmarking
    metrics = profiler.benchmark_latency_and_vram(volume_shape=(16, 32, 32), num_warmup=1, num_runs=2, use_amp=False)
    assert "mean_latency_sec" in metrics
    assert "throughput_mvox_sec" in metrics
    assert metrics["mean_latency_sec"] > 0.0
    
    # 2. ONNX graph export & verification
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = Path(tmpdir) / "test_model.onnx"
        profiler.export_onnx(output_path=onnx_file, input_shape=(16, 32, 32))
        
        assert onnx_file.exists()
        assert onnx_file.stat().st_size > 0
        
        # Verify ONNX model passes official onnx checker
        loaded_onnx = onnx.load(str(onnx_file))
        onnx.checker.check_model(loaded_onnx)

