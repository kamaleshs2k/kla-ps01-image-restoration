import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class KLARestorationDataset(Dataset):
    """
    Loads (GT, NoisyLR) pairs.
    Upsamples NoisyLR to GT resolution using bicubic interpolation,
    so the model receives two same-sized images: a degraded 256x256
    input and a clean 256x256 target.
    """

    def __init__(self, root_dir, ids_file, augment=False):
        self.gt_dir = os.path.join(root_dir, "GT")
        self.lr_dir = os.path.join(root_dir, "NoisyLR")
        with open(ids_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file_id = self.ids[idx]

        gt = np.load(os.path.join(self.gt_dir, file_id))     # (256, 256)
        lr = np.load(os.path.join(self.lr_dir, file_id))     # (128, 128)

        # Note: we do NOT clip lr to [0,1] here — the out-of-range values
        # carry information about noise strength. Clipping throws that away.

        gt = torch.from_numpy(gt).float().unsqueeze(0)   # (1, 256, 256)
        lr = torch.from_numpy(lr).float().unsqueeze(0)   # (1, 128, 128)

        # Upsample LR to match GT resolution so the model does
        # denoise + deblur + SR all as one image-to-image mapping
        lr_up = F.interpolate(
            lr.unsqueeze(0), size=gt.shape[-2:], mode="bicubic", align_corners=False
        ).squeeze(0)   # (1, 256, 256)

        if self.augment:
            gt, lr_up = self._augment(gt, lr_up)

        return lr_up, gt, file_id

    def _augment(self, gt, lr_up):
        # Random horizontal flip
        if torch.rand(1).item() < 0.5:
            gt = torch.flip(gt, dims=[-1])
            lr_up = torch.flip(lr_up, dims=[-1])
        # Random vertical flip
        if torch.rand(1).item() < 0.5:
            gt = torch.flip(gt, dims=[-2])
            lr_up = torch.flip(lr_up, dims=[-2])
        # Random 90-degree rotation
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            gt = torch.rot90(gt, k, dims=[-2, -1])
            lr_up = torch.rot90(lr_up, k, dims=[-2, -1])
        return gt, lr_up


if __name__ == "__main__":
    # quick sanity check
    from torch.utils.data import DataLoader

    train_ds = KLARestorationDataset("train/train", "train_ids.txt", augment=True)
    val_ds = KLARestorationDataset("train/train", "val_ids.txt", augment=False)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")

    lr_up, gt, file_id = train_ds[0]
    print(f"Sample {file_id}: lr_up shape {lr_up.shape}, gt shape {gt.shape}")
    print(f"lr_up range: [{lr_up.min():.3f}, {lr_up.max():.3f}]")
    print(f"gt range:    [{gt.min():.3f}, {gt.max():.3f}]")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    batch_lr, batch_gt, batch_ids = next(iter(train_loader))
    print(f"Batch shapes: lr {batch_lr.shape}, gt {batch_gt.shape}")
