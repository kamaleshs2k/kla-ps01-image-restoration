import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return x + out


class BaselineRestorer(nn.Module):
    """
    Simple residual CNN: predicts the residual (GT - input),
    which is easier to learn than predicting the full image directly.
    """
    def __init__(self, channels=64, num_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(1, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        residual = self.tail(feat)
        return x + residual  # predict input + learned correction


if __name__ == "__main__":
    model = BaselineRestorer()
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
