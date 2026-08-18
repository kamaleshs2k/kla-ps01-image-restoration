import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_msssim import SSIM

from dataset import KLARestorationDataset
from model import BaselineRestorer


def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


class CombinedLoss(nn.Module):
    """L1 for pixel-level fidelity + SSIM for structural/contrast fidelity."""
    def __init__(self, l1_weight=1.0, ssim_weight=0.3):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=1)
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        # SSIM returns similarity (1 = identical), so loss = 1 - SSIM
        ssim_loss = 1 - self.ssim(pred.clamp(0, 1), target)
        return self.l1_weight * l1_loss + self.ssim_weight * ssim_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = KLARestorationDataset("train/train", "train_ids.txt", augment=True)
    val_ds = KLARestorationDataset("train/train", "val_ids.txt", augment=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = BaselineRestorer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    criterion = CombinedLoss(l1_weight=1.0, ssim_weight=0.3)

    best_psnr = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for lr_up, gt, _ in train_loader:
            lr_up, gt = lr_up.to(device), gt.to(device)

            optimizer.zero_grad()
            pred = model(lr_up)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * lr_up.size(0)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for lr_up, gt, _ in val_loader:
                lr_up, gt = lr_up.to(device), gt.to(device)
                pred = model(lr_up)
                loss = criterion(pred, gt)
                val_loss += loss.item() * lr_up.size(0)
                val_psnr += psnr(pred.clamp(0, 1), gt).item() * lr_up.size(0)

        val_loss /= len(val_ds)
        val_psnr /= len(val_ds)
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"val_psnr {val_psnr:.2f} dB | lr {current_lr:.2e} | {elapsed:.1f}s")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"  -> saved new best model (PSNR {best_psnr:.2f} dB)")

        scheduler.step()

    print(f"\nTraining done. Best val PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
