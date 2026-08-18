import numpy as np
import os
import random
import matplotlib.pyplot as plt

root = "train/train"
gt_dir = os.path.join(root, "GT")
lr_dir = os.path.join(root, "NoisyLR")

ids = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy"))
pairs = [(os.path.join(gt_dir, f), os.path.join(lr_dir, f)) for f in ids]
print(len(pairs), "pairs found")

random.seed(42)
ids_shuffled = ids.copy()
random.shuffle(ids_shuffled)

val_size = int(0.15 * len(ids_shuffled))
val_ids = ids_shuffled[:val_size]
train_ids = ids_shuffled[val_size:]
print(f"Train: {len(train_ids)} | Val: {len(val_ids)}")

# Save split so you don't reshuffle differently each run
with open("train_ids.txt", "w") as f:
    f.write("\n".join(train_ids))
with open("val_ids.txt", "w") as f:
    f.write("\n".join(val_ids))
print("Saved train_ids.txt and val_ids.txt")
