# KLA PS01 — AI-Based Restoration of Degraded Images

Semicon India Hackathon — Team **{TEAM_NAME}**

A residual CNN that restores semiconductor inspection images degraded by
speckle noise, additive Gaussian noise, and 2x spatial downsampling
(in any combination and order) back to clean, full-resolution images.

---

## 1. Repository contents

| File | Purpose |
|---|---|
| `model.py` | `BaselineRestorer` — residual CNN architecture (8 ResBlocks, 64 channels, ~592K params). |
| `dataset.py` | `KLARestorationDataset` — loads paired GT/NoisyLR `.npy` files, bicubic-upsamples the degraded input to GT resolution, optional flip/rotation augmentation. |
| `train.py` | Full training script (reproduces the model from scratch): Adam + cosine annealing LR over 60 epochs, combined **L1 + SSIM** loss, saves the best checkpoint by validation PSNR. |
| `inference.py` | **Standalone evaluation script.** Takes a folder of degraded `.npy` images, runs the trained model, writes restored `.npy` images to an output folder. This is the script KLA will run as-is to benchmark quality and throughput. |
| `checkpoints/best_model.pt` | Trained model weights. |
| `train_ids.txt` / `val_ids.txt` | Fixed 85/15 train/validation split (for reproducibility). |
| `sample_outputs/` | Example restored outputs on held-out validation images. |
| `requirements.txt` | Python dependencies (see note in the file — replace with real `pip freeze` before final submission). |

---

## 2. Setup

```bash
git clone <THIS_REPO_URL>
cd <THIS_REPO_NAME>

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.1+, on both CPU and NVIDIA GPU (T4 for
training, H100 target for evaluation/deployment).

If the trained checkpoint (`checkpoints/best_model.pt`) is too large for a
normal git push, it is hosted via {Git LFS / Google Drive / HuggingFace —
**fill in actual link here**} — download it into `checkpoints/` before
running inference.

---

## 3. Running inference (what KLA will run)

```bash
python inference.py \
    --input_dir  /path/to/test/NoisyLR \
    --output_dir /path/to/restored_outputs
```

- `--input_dir`: a folder of degraded `.npy` images (grayscale, single
  channel, float32). No ground truth needed or expected.
- `--output_dir`: created automatically if it doesn't exist. One `.npy`
  file per input, same filename, containing the restored image as
  float32 in `[0, 1]`.
- Optional flags: `--checkpoint` (default `checkpoints/best_model.pt`),
  `--batch_size` (default 16), `--num_workers` (default 2), `--device`
  (default: auto-detects CUDA, falls back to CPU).

The script prints total wall-clock time and images/sec at the end (used
for the throughput scoring axis). No manual edits are required — it will
run on any folder of `.npy` files at any resolution.

**Degradation/resolution assumption:** per the problem statement, spatial
resolution reduction is always a 2x factor (e.g. 128→256, 256→512).
`inference.py` bicubic-upsamples each input by 2x before feeding it to
the model, matching exactly what `dataset.py` does during training — so
train/test preprocessing is identical.

**Out-of-range pixel values are preserved, not clipped.** Speckle noise
can push degraded pixel values slightly outside `[0, 1]`; this is
expected behavior per the dataset spec, and both `dataset.py` and
`inference.py` deliberately do not clip inputs, since the model was
trained to make use of that signal.

---

## 4. Reproducing training from scratch

```bash
# expects a folder ./train/train/{GT,NoisyLR}/ with paired .npy files,
# and train_ids.txt / val_ids.txt listing filenames for each split
pip install pytorch-msssim
python train.py
```

This runs 60 epochs of Adam + cosine-annealing LR, using a combined
`1.0 * L1 + 0.3 * SSIM` loss, and saves the best checkpoint (by validation
PSNR) to `checkpoints/best_model.pt`. Training log prints per-epoch train
loss, val loss, val PSNR, current LR, and wall-clock time per epoch.

To regenerate the train/val split from scratch instead of using the
provided `train_ids.txt` / `val_ids.txt`, see `{explore_data.py / split
script — reference here if included in repo}`.

---

## 5. Model architecture

Residual CNN (`BaselineRestorer` in `model.py`):

```
input (1×H×W)
  -> Conv2d head (1 -> 64 channels)
  -> 8x ResBlock(64)      # each: Conv3x3 -> ReLU -> Conv3x3, skip connection
  -> Conv2d tail (64 -> 1 channels)
  -> add back to input (residual/correction prediction)
```

~592K parameters. The model predicts a *correction* added to the bicubic-
upsampled input rather than reconstructing the image from scratch, which
stabilizes training and keeps the network small and fast enough for the
H100 throughput requirement.

---

## 6. Results

| Metric | Bicubic-only baseline | Our model |
|---|---|---|
| Val PSNR (dB) | 23.19 | {FINAL_PSNR — fill in after L1+SSIM/60-epoch run} |
| Val SSIM | – | {FINAL_SSIM} |
| Val LPIPS | – | {FINAL_LPIPS} |
| Inference throughput (images/sec, H100) | – | {THROUGHPUT} |

See `sample_outputs/` for degraded input / model output / ground truth
comparisons on validation images.

---

## 7. Notes on reproducibility / training hygiene

- Train/val split is fixed and saved (`train_ids.txt`, `val_ids.txt`) —
  re-running `train.py` uses the exact same split every time.
- Best checkpoint is selected by validation PSNR, not final-epoch weights.
- All hyperparameters (LR, loss weights, epoch count, batch size) are set
  directly in `train.py` for full reproducibility — no hidden config.
- No test-set data or labels were used at any point during training or
  model selection.
