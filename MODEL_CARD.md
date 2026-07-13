# Model Card: CrossScan-Joint-ResUNet-v1

An open, modality-agnostic 3D deep semantic segmentation system for cross-modality abdominal CT and MRI, equipped with a clinically calibrated self-auditing reliability layer (Uncertainty Estimation, Label-Free QC, and Conformal Risk Control Triage) and production DICOM-SEG deployment.

---

## 1. Model Details

- **Model Identifier**: `CrossScan-Joint-ResUNet-v1`
- **Model Developer**: Djebril Mohammed CHENNOUFI
- **Release Date**: July 2026
- **Model Type**: 3D Deep Convolutional Residual UNet with Deep Supervision
- **Framework & Libraries**: PyTorch 2.4, MONAI 1.4, SimpleITK, HighDICOM, ONNX Runtime
- **Repository**: [https://github.com/DjebrilSVN/CrossScan](https://github.com/DjebrilSVN/CrossScan)
- **License**: Apache 2.0
- **Primary Contacts & Citation**: See [CITATION.cff](CITATION.cff) and repository issues.

### Architectural Specifications
- **Input Tensor**: `(1, 1, D, H, W)` single-channel 3D volume resampled to canonical $1.5 \times 1.5 \times 1.5\text{ mm}$ isometric spacing and RAS+ spatial orientation.
- **Output Tensor**: `(1, 14, D, H, W)` 14-channel voxel logit tensor corresponding to background + 13 abdominal organs.
- **Base Feature Depth**: 32 initial feature maps with 4 resolution scales $(32 \to 64 \to 128 \to 256)$ and deep supervision heads.
- **Bottleneck Latent Vector**: 256-dimensional pooled embedding space used for latent Mahalanobis OOD distance scoring.
- **Deployment Artifacts**: PyTorch checkpoint (`checkpoints/joint_model/joint_model_best.pt`, 24.2 MB) and dynamic ONNX graph (`checkpoints/deployment/crossscan_joint.onnx`, 21.56 MB).

---

## 2. Intended Clinical Use & Clinical Workflow

### 2.1 Intended Use
- **Assisted Multi-Organ Contouring**: Automated segmentations for cross-sectional abdominal CT and MRI to expedite radiation therapy contouring, surgical resection planning, and organ volumetry.
- **Cross-Modality Cohort Studies**: Retrospective clinical trials spanning heterogeneous multi-scanner CT and MRI cohorts where running separate per-modality models introduces domain discrepancy.
- **PACS-Integrated Triage**: Pre-populates PACS archives with standardized DICOM-SEG objects while routing anomalous or uncertain scans to radiologists for manual inspection.

### 2.2 Target Clinical Persona
- **Attending & Resident Radiologists**: Interactive segmentation review and boundary refinement via 3D Slicer / OHIF.
- **Radiation Oncologists & Medical Physicists**: Organ-at-Risk (OAR) delineation with millimeter surface tolerance guarantees.
- **Clinical Data Engineers**: Automated DICOMweb PACS pipeline integration via MONAI Deploy Application Package (MAP).

### 2.3 Radiologist-in-the-Loop Protocol
```
[DICOM Ingestion] ──► [CrossScan 3D Ingestion] ──► [Conformal CRC Triage Engine]
                                                           │
          ┌────────────────────────────────────────────────┴───────────────────────────────┐
          ▼                                                ▼                               ▼
    [AUTO_ACCEPT]                                 [FLAG_FOR_REVIEW]                     [REJECT]
(Top 85% high confidence)                    (Boundary ribbons / uncertainty)      (OOD / Severe Artifact)
          │                                                │                               │
          ▼                                                ▼                               ▼
Direct PACS STOW-RS Archival                 High-Entropy Slices Dispatched        Aborted from PACS
(Radiologist notification)                   to OHIF / 3D Slicer for Edits        (Radiologist re-scan)
```

---

## 3. Clinical Contraindications & Out-of-Scope Deployments

> [!WARNING]
> **Out-of-Scope Deployments**:
> 1. **Pediatric Cohorts (< 16 Years)**: Pediatric abdominal anatomy exhibits markedly different organ proportions, reduced retroperitoneal fat planes, and developmental renal variants not present in the adult training cohort.
> 2. **Low-Field Extremity Scanners (< 0.5 Tesla)**: Extremely low SNR regimes and severe magnetic field inhomogeneity fall outside the calibrated Mahalanobis in-distribution threshold ($D_M > 0.707$).
> 3. **Autonomous Primary Diagnostic Decision-Making**: The model must never be used autonomously to establish a primary medical diagnosis or dictate emergent surgical intervention without physician review.
> 4. **Post-Surgical Anatomies with Major Resection**: Patients with hemi-hepatectomy, total gastrectomy, or major organ transpositions must be manually audited, as topological assumptions may be violated.

---

## 4. Multi-Center Training Data & Scanner Coverage

The model was developed and validated strictly on real-world, peer-reviewed clinical datasets from diverse geographical centers:

| Dataset | Modality | Patient Volumes | Clinical Centers | Scanner Manufacturers | Field Strengths / Acquisition |
|:---|:---:|:---:|:---:|:---|:---:|
| **AMOS22** | CT & MRI | 600 (500 CT, 100 MRI) | Multi-Center (Asia/Europe) | Siemens, GE, Philips | 1.5T, 3.0T / Contrast & Non-contrast |
| **TotalSegmentator v2** | CT & MRI | 2,204 (1,204 CT, 1,000+ MRI) | University Hospital Basel | Siemens Somatom, Magnetom | Standard Clinical Protocols |
| **CHAOS Challenge** | MRI (T1/T2) | 80 Multi-Sequence | Dokuz Eylul University | Philips Achieva, Intera | 1.5T / T1-DUAL in/out-phase, T2-SPIR |

### Harmonization & Partial Labeling
- **Radiometric Normalization**: CT windowed to $[-160, 240]\text{ HU} \to [0, 1]$; MRI normalized via N4ITK bias correction + 1st–99th percentile intensity clipping and $z$-score standardization.
- **Masked Marginal Loss**:
  $$\mathcal{L} = \mathcal{L}_{\text{Masked-Dice}} + \mathcal{L}_{\text{Masked-CE}}$$
  Dynamically re-normalizes loss gradients over verified present structures ($\mathcal{C}_{\text{present}}$), eliminating negative gradient penalties for missing annotations.

---

## 5. Metrics Reloaded Quantitative Evaluation

Evaluated across held-out multi-center clinical test cohorts with **patient-level non-parametric bootstrap 95% confidence intervals** ($B = 1000$ resamples) conforming strictly to *Metrics Reloaded* (Maier-Hein et al., Nature Methods 2024):

### 5.1 Multi-Modality Benchmark Comparison

| Architectural Model | CT Dice [95% CI] | MRI Dice [95% CI] | Transfer Drop ($\Delta$) | NSD @ $\tau$ | HD95 (mm) | Failure Rate (< 0.70) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **nnU-Net CT Baseline** | **0.916** [0.908, 0.924] | 0.412 [0.386, 0.439] | <span style="color:red">+0.504</span> | 0.648 | 16.1 mm | 51.6% |
| **nnU-Net MRI Baseline** | 0.458 [0.418, 0.495] | **0.867** [0.857, 0.881] | <span style="color:red">-0.409</span> | 0.662 | 15.3 mm | 46.2% |
| **CrossScan Joint Model** | **0.884** [0.876, 0.892] | **0.894** [0.884, 0.904] | <span style="color:green">**-0.010**</span> | **0.879** | **4.6 mm** | **1.7%** |
| **CrossScan + 3D LoRA Adapter** | **0.903** [0.897, 0.910] | **0.893** [0.884, 0.902] | <span style="color:green">**+0.010**</span> | **0.891** | **4.0 mm** | **1.0%** |
| **CrossScan + Conformal CRC** | **0.897** [0.890, 0.904] | **0.900** [0.892, 0.907] | <span style="color:green">**-0.003**</span> | **0.904** | **3.5 mm** | **0.0%** |

### 5.2 Structure-Specific Metrics Reloaded Breakdown (CrossScan Joint Model)

| Anatomical Structure | SNOMED CT Code | RadLex ID | Tolerance ($\tau$) | CT Dice | MRI Dice | NSD @ $\tau$ | HD95 (mm) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Liver** | `10200004` | `RID58` | 2.0 mm | 0.952 | 0.941 | 0.938 | 3.2 mm |
| **Spleen** | `78961009` | `RID86` | 1.5 mm | 0.931 | 0.915 | 0.912 | 3.8 mm |
| **Right Kidney** | `24964005` | `RID205` | 1.5 mm | 0.925 | 0.910 | 0.904 | 4.0 mm |
| **Left Kidney** | `18639004` | `RID206` | 1.5 mm | 0.918 | 0.902 | 0.898 | 4.2 mm |
| **Pancreas** | `15776009` | `RID170` | 1.5 mm | 0.835 | 0.812 | 0.795 | 7.5 mm |
| **Stomach** | `34516001` | `RID80` | 2.0 mm | 0.892 | 0.868 | 0.860 | 5.8 mm |
| **Abdominal Aorta** | `281130003` | `RID1001` | 1.0 mm | 0.914 | 0.895 | 0.890 | 2.9 mm |
| **Inferior Vena Cava** | `64131007` | `RID1220` | 1.0 mm | 0.885 | 0.862 | 0.852 | 3.5 mm |

---

## 6. Systematic Multi-Factor Ablation Findings

Ablation benchmarking across 7 conditions revealed three critical scientific findings:
1. **SynthSeg-Style Domain Randomization** is the single greatest factor in cross-modality feature transfer, cutting the transfer drop from $-0.504$ down to $-0.010$.
2. **Masked Marginal Loss** resolves the multi-center label mismatch problem, boosting joint Dice by $+0.057$ across partial label annotations.
3. **Conformal Risk Control (CRC)** eliminates catastrophic failures ($0.0\%$ unflagged failures) by filtering borderline cases into the radiologist review queue.

---

## 7. Failure Modes Taxonomy & Mitigation Matrix

| Failure Mode | Clinical Mechanism | Frequency | Risk Tier | Mitigation Strategy |
|:---|:---|:---:|:---:|:---|
| **Boundary Ambiguity** | Iso-intense interfaces (Pancreas head vs Duodenum C-loop) | 4.2% | Medium | Voxel Shannon entropy ribbons highlight boundary voxels for interactive clicks. |
| **Kidney Laterality Confusion** | Bilateral anatomical symmetry across midline | < 0.1% | High | Hard geometric audit: Left kidney $X$-centroid must strictly exceed Right kidney $X$-centroid. |
| **Severe B1 Inhomogeneity** | Dielectric shading at 3.0T MRI | 2.8% | Medium | Mahalanobis detector flags OOD latent distance ($D_M > 0.707$) $\to$ routes to `FLAG_FOR_REVIEW`. |
| **Respiratory Ghosting Artifact** | Phase-encoding motion blurs organ margins | 3.5% | Low | Simulated k-space motion augmentation during training provides intrinsic robustness. |

---

## 8. Environmental Impact & Hardware Profiling

Evaluated on an NVIDIA RTX 3080 Laptop GPU across standard abdominal 3D volumes ($64 \times 128 \times 128 = 1.05\text{ MVox}$):

- **Inference Latency (FP16 AMP)**: **0.028 seconds** per volume (35.7 volumes/sec).
- **Inference Throughput**: **37.02 MVox/sec**.
- **Peak CUDA VRAM**: **0.51 GB**.
- **Energy Footprint**: $< 0.002\text{ kWh}$ per 100 patient segmentations.
- **Hardware Portability**: Runs in real-time on consumer laptops with 8GB VRAM; does not require multi-GPU cloud clusters.

---

## 9. Regulatory Disclaimer & Licensing

```
DISCLAIMER: FOR RESEARCH AND INVESTIGATIONAL USE ONLY.
This software and associated model weights have NOT received 510(k) clearance or PMA approval
from the United States FDA, nor CE-mark certification under the EU Medical Device Regulation (MDR).
CrossScan is designed strictly as an assistive research tool for qualified healthcare professionals
and must NOT be used for autonomous clinical diagnosis or patient triage.
```

**License**: Apache License 2.0. Copyright (c) 2026 Djebril Mohammed CHENNOUFI.

