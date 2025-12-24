import csv
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class DeepfakeVideoDataset(Dataset):
    """
    Loads a sequence of face frames for each video ID listed in a CSV file.
    CSV format:
        video_id,label
        0001,0
        0002,1
    """

    def __init__(
        self,
        csv_path,
        frames_root,
        seq_len=16,
        augment=False,
    ):
        self.frames_root = Path(frames_root)
        self.seq_len = seq_len
        self.augment = augment

        # ---------------- Load CSV ----------------
        self.samples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row["video_id"]
                label = int(row["label"])

                frame_dir = self.frames_root / video_id
                if frame_dir.exists():
                    self.samples.append((video_id, label))

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found in dataset")

        # ---------------- Transforms ----------------
        base_tf = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                *base_tf,
            ])
        else:
            self.transform = transforms.Compose(base_tf)

    # --------------------------------------------------
    # REQUIRED BY PYTORCH
    # --------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, label = self.samples[idx]
        frame_dir = self.frames_root / video_id

        frames = sorted(frame_dir.glob("*.jpg"))
        if len(frames) == 0:
            raise RuntimeError(f"No frames found for video {video_id}")

        # -------- Frame sampling --------
        if len(frames) >= self.seq_len:
            indices = torch.linspace(
                0, len(frames) - 1, self.seq_len
            ).long()
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < self.seq_len:
                frames.append(frames[-1])

        # -------- Load & transform --------
        imgs = []
        for f in frames:
            img = Image.open(f).convert("RGB")
            img = self.transform(img)
            imgs.append(img)

        x = torch.stack(imgs)           # (T, 3, H, W)
        y = torch.tensor(label).long()  # scalar

        return x, y
