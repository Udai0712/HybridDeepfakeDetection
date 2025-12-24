import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.hybrid_model import HybridModel
from utils.dataset import DeepfakeVideoDataset
from utils.metrics import compute_metrics


# --------------------------------------------------
# Device helper
# --------------------------------------------------
def get_device(device_arg):
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------
# Train / Validation loops
# --------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for frames, labels in tqdm(loader, desc="Train", leave=False):
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Val", leave=False):
            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    metrics = compute_metrics(y_true, y_pred, y_prob)
    return total_loss / len(loader), metrics


# --------------------------------------------------
# Main
# --------------------------------------------------
def main(args):
    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---------------- Dataset ----------------
    train_ds = DeepfakeVideoDataset(
        csv_path=args.train_csv,
        frames_root=args.frames_root,
        seq_len=args.seq_len,
        augment=True,
    )

    val_ds = DeepfakeVideoDataset(
        csv_path=args.val_csv,
        frames_root=args.frames_root,
        seq_len=args.seq_len,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    # ---------------- Model ----------------
    model = HybridModel(seq_len=args.seq_len).to(device)

    # 🔒 FULLY FREEZE CNN BACKBONE (CRITICAL)
    for p in model.parameters():
        if p.ndim > 2:  # CNN parameters
            p.requires_grad = False

    # ---------------- LOSS (STABLE) ----------------
    class_weights = torch.tensor([1.0, 1.2]).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1  # 🔥 PREVENTS CLASS COLLAPSE
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    best_f1 = 0.0

    # ---------------- Training ----------------
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "f1": best_f1,
                },
                os.path.join(args.checkpoint_dir, "best_model.pth"),
            )
            print("[INFO] Saved best model")

    print("\n[INFO] Training completed successfully.")


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Hybrid Deepfake Detection Training (Stable Version)"
    )

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--frames_root", type=str, default="data/frames")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=24)

    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
    )

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    main(args)
