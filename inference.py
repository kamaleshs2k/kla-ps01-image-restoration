"""
KLA PS01 - Image Restoration - Standalone Evaluation Script
=============================================================

Reads degraded (noisy, low-resolution) .npy images from an input directory,
runs them through the trained restoration model, and writes restored
(denoised, super-resolved) .npy images to an output directory.

This script is meant to be run AS-IS by the benchmarking team:

    python inference.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/restored_outputs

It intentionally has NO dependency on dataset.py's file-listing / split-file
logic (train_ids.txt / val_ids.txt), since the real test set has no such
files and no ground truth. It only needs a folder of .npy files.

Preprocessing matches training exactly:
    1. Load degraded image as float32 (values may lie slightly outside [0,1] --
       this is expected, caused by speckle noise, and is NOT clipped).
    2. Bicubic-upsample by 2x to bring it to the same resolution the model
       was trained to output (the dataset's only degradation-resolution rule
       is a 2x spatial reduction, e.g. 128->256 or 256->512).
    3. Run through the trained residual CNN.
    4. Clamp the output to [0, 1] (ground truth range) and save as float32 .npy,
       same filename as the input, in --output_dir.

Batched for throughput on GPU. Prints end-to-end wall-clock time and
images/sec at the end, since inference throughput is one of the scored
evaluation axes.
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import BaselineRestorer


# ----------------------------------------------------------------------
# Dataset: just reads whatever .npy files exist in a folder, no ground
# truth, no split file. Keeps this script fully self-contained.
# ----------------------------------------------------------------------
class InferenceFolderDataset(Dataset):
    def __init__(self, input_dir, upsample_factor=2):
        self.input_dir = input_dir
        self.upsample_factor = upsample_factor
        self.filenames = sorted(
            f for f in os.listdir(input_dir) if f.lower().endswith(".npy")
        )
        if len(self.filenames) == 0:
            raise RuntimeError(
                f"No .npy files found in {input_dir}. "
                f"Check --input_dir points at a folder of degraded .npy images."
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        arr = np.load(os.path.join(self.input_dir, fname)).astype(np.float32)

        # NOTE: intentionally NOT clipped to [0,1]. Speckle noise can push
        # values outside the true range; that's a real feature of this
        # dataset, not a bug, and clipping here would throw away signal
        # the model was trained to make use of.
        lr = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

        target_h = lr.shape[-2] * self.upsample_factor
        target_w = lr.shape[-1] * self.upsample_factor
        lr_up = F.interpolate(
            lr.unsqueeze(0), size=(target_h, target_w),
            mode="bicubic", align_corners=False,
        ).squeeze(0)  # (1, H*2, W*2)

        return lr_up, fname


def collate_variable_size(batch):
    """
    Images in this challenge can come at different native resolutions
    (e.g. some pairs are 128->256, others 256->512). To keep batching
    simple and robust, we group by matching shape within a batch and
    fall back to batch_size=1 collation when shapes differ.
    """
    shapes = {t.shape for t, _ in batch}
    if len(shapes) == 1:
        tensors = torch.stack([t for t, _ in batch], dim=0)
        names = [n for _, n in batch]
        return tensors, names
    # Mixed shapes in this batch: return as a list, handled one-by-one downstream.
    return [t for t, _ in batch], [n for _, n in batch]


def run_inference(input_dir, output_dir, checkpoint_path, batch_size=16,
                   num_workers=2, device_str=None):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(
        device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    model = BaselineRestorer().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model from {checkpoint_path} ({n_params:,} params)")

    ds = InferenceFolderDataset(input_dir)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_variable_size,
    )

    print(f"Found {len(ds)} images in {input_dir}. Writing outputs to {output_dir}")

    total_images = 0
    t_start = time.time()

    with torch.no_grad():
        for batch, names in loader:
            if isinstance(batch, list):
                # Mixed shapes in this batch -> process one at a time
                for tensor, name in zip(batch, names):
                    tensor = tensor.unsqueeze(0).to(device, non_blocking=True)
                    pred = model(tensor).clamp(0, 1).squeeze(0).squeeze(0).cpu().numpy()
                    np.save(os.path.join(output_dir, name), pred.astype(np.float32))
                    total_images += 1
            else:
                batch = batch.to(device, non_blocking=True)
                preds = model(batch).clamp(0, 1).squeeze(1).cpu().numpy()
                for pred, name in zip(preds, names):
                    np.save(os.path.join(output_dir, name), pred.astype(np.float32))
                    total_images += 1

    elapsed = time.time() - t_start
    print(f"\nDone. Restored {total_images} images in {elapsed:.2f}s "
          f"({total_images / elapsed:.2f} images/sec, "
          f"{1000 * elapsed / total_images:.1f} ms/image avg).")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run KLA PS01 restoration model on a folder of degraded .npy images."
    )
    p.add_argument("--input_dir", type=str, required=True,
                    help="Directory containing degraded (NoisyLR) .npy images.")
    p.add_argument("--output_dir", type=str, required=True,
                    help="Directory to write restored .npy images to.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                    help="Path to trained model weights (default: checkpoints/best_model.pt).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None,
                    help="Force a device, e.g. 'cuda' or 'cpu'. Default: auto-detect.")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"ERROR: --input_dir does not exist: {args.input_dir}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(
            f"ERROR: checkpoint not found at {args.checkpoint}. "
            f"Pass --checkpoint /path/to/best_model.pt if it lives elsewhere."
        )

    run_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
