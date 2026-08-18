# KLA PS01 — AI-Based Restoration of Degraded Images

**KLA Problem Statement PS01 — AI-Based Restoration of Degraded Images for Semiconductor Inspection**

**GitHub:** https://github.com/kamaleshs2k/kla-ps01-image-restoration

---

## 1. Overview

This project addresses the KLA PS01 image-restoration challenge.

The objective is to restore clean, full-resolution grayscale images from degraded observations affected by:

* Speckle noise
* Additive Gaussian noise
* Spatial downsampling

The proposed solution uses a lightweight residual CNN. The degraded low-resolution image is first upsampled by **2× using bicubic interpolation**. The CNN then predicts a residual correction that is added to the upsampled image to produce the restored output.

The model is designed to balance restoration quality, model size, and inference efficiency.

---

## 2. Solution Pipeline

```text
Degraded NoisyLR .npy
        |
        v
Load Grayscale Image
        |
        v
2× Bicubic Upsampling
        |
        v
Residual CNN
8 Residual Blocks
64 Feature Channels
        |
        v
Predicted Residual
        |
        v
Bicubic Input + Residual
        |
        v
Restored Image
        |
        v
Output .npy
```

---

## 3. Repository Structure

```text
kla-ps01-image-restoration/
│
├── README.md
├── requirements.txt
├── inference.py
├── train.py
├── dataset.py
├── model.py
├── explore_data.py
├── train_ids.txt
├── val_ids.txt
│
├── checkpoints/
│   └── best_model.pt
│
└── sample_outputs/
    ├── predictions_check.png
    └── test_restored/
        ├── 000000.npy
        ├── 000001.npy
        ├── ...
        └── 000399.npy
```

### Main Files

| File                        | Description                                                     |
| --------------------------- | --------------------------------------------------------------- |
| `inference.py`              | Standalone evaluation script for restoring degraded test images |
| `train.py`                  | Complete training script for reproducing model training         |
| `dataset.py`                | Dataset loading, preprocessing and augmentation                 |
| `model.py`                  | Residual CNN architecture                                       |
| `explore_data.py`           | Dataset inspection and train/validation split utility           |
| `checkpoints/best_model.pt` | Trained model weights                                           |
| `train_ids.txt`             | Fixed training split containing 2,720 samples                   |
| `val_ids.txt`               | Fixed validation split containing 480 samples                   |
| `sample_outputs/`           | Validation examples and restored test outputs                   |
| `requirements.txt`          | Python dependencies                                             |

---

## 4. Dataset

The training dataset consists of paired:

```text
GT       → Clean Ground Truth
NoisyLR  → Degraded Low-Resolution Image
```

### Dataset Statistics

| Property               |     Value |
| ---------------------- | --------: |
| Total paired samples   |      3200 |
| Training samples       |      2720 |
| Validation samples     |       480 |
| Train/Validation split | 85% / 15% |

### Example Dimensions

```text
GT       : 256 × 256
NoisyLR  : 128 × 128
```

The 128×128 degraded input is upsampled to 256×256 before being passed to the CNN.

```text
128 × 128 NoisyLR
        |
        | 2× Bicubic
        v
256 × 256 CNN Input
        |
        v
256 × 256 Restored Output
```

The degraded input values are not clipped before model processing because noise can produce values slightly outside `[0,1]`.

---

## 5. Environment Setup

### Clone Repository

```bash
git clone https://github.com/kamaleshs2k/kla-ps01-image-restoration.git
cd kla-ps01-image-restoration
```

### Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows:

```powershell
.venv\Scripts\activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Development Environment

* **Python:** 3.12
* **Framework:** PyTorch
* **Model:** Residual CNN

CUDA is automatically detected when a CUDA-enabled PyTorch installation and NVIDIA GPU are available.

---

## 6. Evaluation / Inference

The primary evaluation script is:

```text
inference.py
```

It is a standalone Python script and does not require a Jupyter notebook.

### Basic Usage

```bash
python inference.py \
    --input_dir /path/to/Test_NoisyLR/NoisyLR \
    --output_dir /path/to/restored_outputs
```

### Example

```bash
python inference.py \
    --input_dir ./Test_NoisyLR/NoisyLR \
    --output_dir ./restored_outputs
```

The trained checkpoint is automatically loaded from:

```text
checkpoints/best_model.pt
```

### Optional Arguments

| Argument        | Description              | Default                     |
| --------------- | ------------------------ | --------------------------- |
| `--checkpoint`  | Path to model checkpoint | `checkpoints/best_model.pt` |
| `--batch_size`  | Inference batch size     | `16`                        |
| `--num_workers` | DataLoader workers       | `2`                         |
| `--device`      | Device selection         | Automatic CUDA/CPU          |

### Full Example

```bash
python inference.py \
    --input_dir ./Test_NoisyLR/NoisyLR \
    --output_dir ./restored_outputs \
    --checkpoint checkpoints/best_model.pt \
    --batch_size 16 \
    --device cuda
```

### Output

For every input:

```text
input/000001.npy
        ↓
output/000001.npy
```

The output is:

* Grayscale
* Single channel
* `float32`
* Restored target resolution
* Pixel values in `[0,1]`

Ground-truth images are **not required for inference**.

---

## 7. Inference Benchmark

The prototype was tested on **400 test images** using an NVIDIA Tesla T4.

### Hardware

| Parameter   | Value           |
| ----------- | --------------- |
| GPU         | NVIDIA Tesla T4 |
| GPU Memory  | 15 GB           |
| Execution   | CUDA            |
| Test Images | 400             |

### Results

| Metric               |          Result |
| -------------------- | --------------: |
| Total inference time |   61.27 seconds |
| Throughput           | 6.53 images/sec |
| Average latency      |  153.2 ms/image |

The above throughput is the measured prototype performance on the Tesla T4.

> **Note:** The final hackathon benchmarking environment may use a different GPU.

---

## 8. Model Architecture

The model is implemented in `model.py`.

```text
Input
1 × H × W
    |
    v
3×3 Conv
1 → 64 Channels
    |
    v
8 × Residual Blocks
    |
    +-- Conv 3×3
    +-- ReLU
    +-- Conv 3×3
    +-- Skip Connection
    |
    v
3×3 Conv
64 → 1 Channel
    |
    v
Predicted Residual
    |
    +----------------------+
                           |
Bicubic-Upsampled Input ---+
                           |
                           v
                     Restored Image
```

### Model Specifications

| Parameter        |   Value |
| ---------------- | ------: |
| Input channels   |       1 |
| Output channels  |       1 |
| Feature channels |      64 |
| Residual blocks  |       8 |
| Parameters       | 592,065 |
| Checkpoint size  | ~2.3 MB |

The model predicts a correction to the bicubic-upsampled image instead of reconstructing the complete image from scratch.

---

## 9. Training

Training is implemented in:

```text
train.py
```

### Training Configuration

| Parameter             |             Value |
| --------------------- | ----------------: |
| Optimizer             |              Adam |
| Initial learning rate |            `1e-4` |
| Scheduler             | CosineAnnealingLR |
| Epochs                |                60 |
| Batch size            |                16 |
| L1 loss weight        |               1.0 |
| SSIM loss weight      |               0.3 |

### Loss Function

```text
Total Loss = 1.0 × L1 + 0.3 × (1 − SSIM)
```

L1 loss provides pixel-level fidelity while SSIM provides structural and contrast-aware supervision.

### Training Dataset Structure

```text
train/
└── train/
    ├── GT/
    │   ├── 000000.npy
    │   ├── 000001.npy
    │   └── ...
    │
    └── NoisyLR/
        ├── 000000.npy
        ├── 000001.npy
        └── ...
```

The fixed split files are:

```text
train_ids.txt
val_ids.txt
```

### Train From Scratch

```bash
python train.py
```

The best checkpoint is selected using validation PSNR and saved to:

```text
checkpoints/best_model.pt
```

---

## 10. Training Result

The final 60-epoch training run achieved:

### Best Validation Result

```text
Best validation PSNR: 25.59 dB
```

### Final Epoch

```text
Epoch:            60 / 60
Train loss:       0.1095
Validation loss:  0.1048
Validation PSNR:  25.55 dB
Best PSNR:        25.59 dB
```

---

## 11. Baseline Comparison

The bicubic-only baseline achieved:

```text
23.19 dB PSNR
```

The proposed residual CNN achieved:

```text
25.59 dB PSNR
```

### PSNR Improvement

```text
25.59 − 23.19 = +2.40 dB
```

| Method                |         PSNR |
| --------------------- | -----------: |
| Bicubic-only          |     23.19 dB |
| Proposed Residual CNN | **25.59 dB** |
| Improvement           | **+2.40 dB** |

---

## 12. Quality Metrics

| Metric | Bicubic Baseline | Proposed Model |
| ------ | ---------------: | -------------: |
| PSNR   |         23.19 dB |   **25.59 dB** |
| SSIM   |              TBD |            TBD |
| LPIPS  |              TBD |            TBD |

SSIM and LPIPS are intentionally not fabricated. They should be calculated using the final submitted checkpoint before the final submission.

---

## 13. Visual Results

### Validation Comparison

```text
sample_outputs/predictions_check.png
```

The comparison contains:

```text
Input / Bicubic
       ↓
Model Output
       ↓
Ground Truth
```

### Restored Test Outputs

```text
sample_outputs/test_restored/
```

The prototype generated:

```text
400 restored test images
```

---

## 14. Reproducibility

The repository contains:

* Fixed train/validation split
* Dataset implementation
* Model architecture
* Training script
* Standalone inference script
* Trained model checkpoint
* Dependency specification
* Validation visualization
* Restored test outputs

The inference pipeline does **not** require the training dataset.

A reviewer can clone the repository, install the dependencies, provide a test-image directory, and run the inference script without modifying the source code.

---

## 15. Clean Environment Test

The repository was tested from a fresh Git clone.

The following components were verified:

```text
GitHub repository cloning       ✓
Repository files                ✓
Model source loading            ✓
Checkpoint loading              ✓
Inference script                ✓
Input directory support         ✓
Output directory support        ✓
```

---

## 16. Limitations

The current prototype has the following limitations:

* The lightweight model may lose some very fine image structures.
* The reported PSNR is based on the fixed validation split.
* Throughput was measured on a Tesla T4 rather than the final hackathon H100 environment.
* SSIM and LPIPS still need to be calculated for the final results table.
* The current architecture uses bicubic upsampling followed by residual refinement rather than a learned super-resolution upsampling module.

---

## 17. Prototype Status

| Component                |  Status |
| ------------------------ | :-----: |
| Dataset inspection       |    ✓    |
| Paired dataset loader    |    ✓    |
| Train/validation split   |    ✓    |
| Residual CNN             |    ✓    |
| L1 + SSIM training       |    ✓    |
| 60-epoch training        |    ✓    |
| Best checkpoint          |    ✓    |
| Standalone inference     |    ✓    |
| CUDA inference           |    ✓    |
| 400 test images restored |    ✓    |
| T4 benchmark             |    ✓    |
| GitHub repository        |    ✓    |
| Fresh-clone verification |    ✓    |
| PSNR baseline comparison |    ✓    |
| SSIM metric              | 0.758344 |
| LPIPS metric             | 0.317474 |
| Final H100 benchmark     | Pending |

---

## 18. Quick Start

For evaluation only:

```bash
git clone https://github.com/kamaleshs2k/kla-ps01-image-restoration.git
cd kla-ps01-image-restoration

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python inference.py \
    --input_dir /path/to/Test_NoisyLR/NoisyLR \
    --output_dir ./restored_outputs
```

Restored images will be available in:

```text
./restored_outputs/
```

**No training is required to run inference.**

---

## 19. GitHub Repository

https://github.com/kamaleshs2k/kla-ps01-image-restoration
