"""Unit tests for CrossScan Joint 3D Segmentor Architecture."""
import torch
from crossscan.models.joint_segmentor import CrossScanJointSegmentor
from crossscan.core.ontology import NUM_CLASSES


def test_joint_segmentor_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Miniature volume: (B=1, C=1, D=32, H=32, W=32)
    x = torch.randn(1, 1, 32, 32, 32, dtype=torch.float32, device=device)
    model = CrossScanJointSegmentor(
        in_channels=1,
        out_classes=NUM_CLASSES,
        feature_size=16,
        num_scales=4,
        deep_supervision=True
    ).to(device)
    
    # In training mode with deep supervision: returns (B, S, C, D, H, W)
    model.train()
    out_train = model(x)
    assert isinstance(out_train, torch.Tensor)
    assert out_train.shape[2] == NUM_CLASSES  # C dimension
    assert out_train.shape[3:] == (32, 32, 32)  # spatial dimensions
    
    # In eval mode: returns single tensor at full resolution
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert isinstance(out_eval, torch.Tensor)
    assert out_eval.shape == (1, NUM_CLASSES, 32, 32, 32)


def test_extract_bottleneck_representation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 1, 32, 32, 32, dtype=torch.float32, device=device)
    model = CrossScanJointSegmentor(
        in_channels=1,
        out_classes=NUM_CLASSES,
        feature_size=16,
        num_scales=4
    ).to(device)
    model.eval()
    with torch.no_grad():
        bottleneck = model.extract_bottleneck_representation(x)
        
    assert bottleneck.ndim == 2
    assert bottleneck.shape[0] == 1
    assert bottleneck.shape[1] == model.bottleneck_dim


def test_sliding_window_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Volume larger than patch size: (1, 1, 48, 48, 48)
    vol = torch.randn(1, 1, 48, 48, 48, dtype=torch.float32, device=device)
    model = CrossScanJointSegmentor(
        in_channels=1,
        out_classes=NUM_CLASSES,
        feature_size=16,
        num_scales=3,
        deep_supervision=False
    ).to(device)
    logits = model.predict_volume_sliding_window(
        vol,
        roi_size=(32, 32, 32),
        sw_batch_size=1,
        overlap=0.25
    )
    assert logits.shape == (1, NUM_CLASSES, 48, 48, 48)
