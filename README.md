# CrossScan: Robust 3D CT/MRI Multi-Organ Segmentation & Self-Auditing Triage

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI Core](https://img.shields.io/badge/MONAI-1.6+-lightgrey.svg)](https://monai.io/)
[![DICOM-SEG Standard](https://img.shields.io/badge/DICOM--SEG-Compliant-success.svg)](https://www.dicomstandard.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **CrossScan** is a production-grade 3D semantic segmentation system engineered to generalize across Computed Tomography (CT) and Magnetic Resonance Imaging (MRI) on scanners and pulse sequences it has never encountered, equipped with a **self-auditing reliability layer** that detects its own segmentation failures without ground-truth masks.

---

## Key Questions Answered

1. **Cross-Modality Invariance**: *Can a single unified neural network segment abdominal organs across both CT and MRI without modality-specific branching?*
2. **Scanner & Sequence Generalization**: *How resilient is the model when deployed on unseen scanner vendors (Siemens vs. GE vs. Philips), field strengths (1.5T vs. 3T), and imaging protocols?*
3. **Failure Self-Auditing (The Product Angle)**: *Can the system predict its own segmentation errors without ground truth, automatically routing confident cases to PACS while flagging risky cases for radiologist review?*

---

## System Architecture

```
                    ┌─────────────────────────┐
                    │ Multi-Site CT & MRI     │
                    │ AMOS, TotalSegmentator  │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────▼───────────────┐
                 │ Harmonization Engine          │
                 │ - RAS+ Canonical Space        │
                 │ - 1.5mm Isotropic Resampling  │
                 │ - Soft-Tissue HU Windowing    │
                 │ - N4ITK Bias-Field Correction │
                 └───────────────┬───────────────┘
                                 │
                 ┌───────────────▼───────────────┐
                 │ Physics Domain Randomization  │
                 │ - Rician Noise & Quantum SNR  │
                 │ - Quadratic B1 Bias Fields    │
                 │ - k-Space Phase Ghosting      │
                 │ - Slice Thickness Downsample  │
                 └───────────────┬───────────────┘
                                 │
                 ┌───────────────▼───────────────┐
                 │ Modality-Agnostic 3D Segmentor│
                 │ - Residual Encoder UNet       │
                 │ - Masked Marginal Dice + CE   │
                 │ - Bottleneck Latent Hooks     │
                 └───────┬───────────────┬───────┘
                         │               │
      ┌──────────────────▼──────┐ ┌──────▼──────────────────┐
      │ Uncertainty Engine      │ │ Latent OOD Detector     │
      │ - 8-Pass Monte Carlo TTA│ │ - Bottleneck Energy     │
      │ - Voxel Shannon Entropy │ │ - Mahalanobis Distance  │
      └──────────────────┬──────┘ └──────┬──────────────────┘
                         │               │
                 ┌───────▼───────────────▼───────┐
                 │ Label-Free QC Regressor       │
                 │ - Predicts Organ & Case Dice  │
                 │ - Zero Ground-Truth Required  │
                 └───────────────┬───────────────┘
                                 │
                 ┌───────────────▼───────────────┐
                 │ Conformal Risk Control Triage │
                 │ Alpha = 0.10 (Error <= 10%)   │
                 └───────┬───────────────┬───────┘
                         │               │
     ┌───────────────────▼───┐       ┌───▼───────────────────┐
     │ Auto-Accept (Green)   │       │ Radiologist Review    │
     │ Direct to PACS / VNA  │       │ Highlight Uncertainty │
     └───────────┬───────────┘       └───────────────────────┘
                 │
     ┌───────────▼───────────────────────────┐
     │ DICOM-SEG (SOP 1.2.840...66.4)        │
     │ Standard SNOMED CT Anatomical Coded   │
     │ Orthanc DICOMweb / 3D Slicer / OHIF   │
     └───────────────────────────────────────┘
```

---

## Anatomical Ontology & Metrics Reloaded Compliance

CrossScan reconciles divergent multi-center annotation protocols into a unified 8-organ space with Metrics Reloaded boundary tolerance thresholds ($\tau$):

| Class ID | Anatomical Structure | SNOMED CT Code | RadLex ID | Surface Tolerance ($\tau$) |
|:---:|:---|:---:|:---:|:---:|
| **1** | Liver | `10200004` | `RID58` | 2.0 mm |
| **2** | Spleen | `78961009` | `RID86` | 1.5 mm |
| **3** | Right Kidney | `24964005` | `RID205` | 1.5 mm |
| **4** | Left Kidney | `18639004` | `RID206` | 1.5 mm |
| **5** | Pancreas | `15776009` | `RID170` | 1.5 mm |
| **6** | Stomach | `34516001` | `RID80` | 2.0 mm |
| **7** | Abdominal Aorta | `281130003` | `RID1001` | 1.0 mm |
| **8** | Inferior Vena Cava | `64131007` | `RID1220` | 1.0 mm |

---

## Partial Labeling Formulation (Masked Marginal Loss)

Public datasets annotate non-identical organ subsets. A standard cross-entropy loss would penalize missing annotations as false negatives. CrossScan introduces a **Masked Marginal Loss**:

$$\mathcal{L} = \mathcal{L}_{\text{Masked-Dice}} + \mathcal{L}_{\text{Masked-CE}}$$

Where:
$$\mathcal{L}_{\text{Masked-Dice}} = 1 - \frac{1}{|\mathcal{C}_{\text{present}}|} \sum_{c \in \mathcal{C}_{\text{present}}} \frac{2 \sum_i p_{i, c} y_{i, c} + \epsilon}{\sum_i p_{i, c}^2 + \sum_i y_{i, c}^2 + \epsilon}$$

and $\mathcal{L}_{\text{Masked-CE}}$ re-normalizes softmax probabilities dynamically across known structures $\mathcal{C}_{\text{present}}$, preventing unannotated organs from corrupting the gradient.

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/djebr/CrossScan.git
cd CrossScan

# Install package in editable mode
pip install -e . --no-build-isolation
```

---

## CLI Usage

### 1. System Diagnostics
```bash
crossscan info
```

### 2. Dataset Manifests
```bash
crossscan download-manifest --dataset amos_ct
```

### 3. Run Test Suite
```bash
pytest -v
```

---

## Deployment & Standards

- **Standard DICOM-SEG**: Exports full multi-frame binary segments compatible with **3D Slicer**, **OHIF Viewer**, and **Horos**.
- **Orthanc PACS**: Interoperable via standard DICOMweb (`STOW-RS`, `WADO-RS`, `QIDO-RS`).
- **Low-Footprint 8GB VRAM Support**: Employs sliding-window inference with Gaussian blending to execute complete high-resolution 3D abdominal volumes on local RTX 3080 hardware.

---

## License

This project is licensed under the Apache 2.0 License.

