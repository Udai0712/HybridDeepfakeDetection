import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models.hybrid_model import HybridModel
from utils.dataset import DeepfakeVideoDataset
from utils.metrics import compute_metrics


# --------------------------------------------------
# Utility
# --------------------------------------------------

def get_device(device_arg):
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------
# Threshold Search
# --------------------------------------------------

def search_threshold(args):
    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    # ---------------- Dataset ----------------
    val_ds = DeepfakeVideoDataset(
        csv_path=args.val_csv,
        frames_root=args.frames_root,
        seq_len=args.seq_len,
        augment=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ---------------- Model ----------------
    model = HybridModel(seq_len=args.seq_len)
    ckpt = torch.load(args.checkpoint, map_location=device)

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    all_labels = []
    all_probs = []

    # ---------------- Collect probabilities ----------------
    with torch.no_grad():
        for frames, labels in tqdm(val_loader, desc="Collecting probs"):
            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ---------------- Threshold sweep ----------------
    thresholds = np.linspace(0.05, 0.95, 91)
    best_f1 = 0.0
    best_threshold = 0.5

    print("\nSearching best threshold...\n")

    for t in thresholds:
        preds = (all_probs >= t).astype(int)

        metrics = compute_metrics(
            y_true=all_labels,
            y_pred=preds,
            y_prob=all_probs,
        )

        f1 = metrics["f1"]

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print("===== Threshold Search Result =====")
    print(f"Best Threshold : {best_threshold:.2f}")
    print(f"Best F1-score  : {best_f1:.4f}")
    print("==================================")

    return best_threshold, best_f1


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hybrid Deepfake Detection Threshold Search")

    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--frames_root", type=str, default="data/frames")
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=16)

    parser.add_argument("--device", type=str, default="mps",
                        choices=["mps", "cuda", "cpu"])

    args = parser.parse_args()
    search_threshold(args)
