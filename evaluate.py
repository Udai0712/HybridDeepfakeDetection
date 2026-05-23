import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from models.hybrid_model import HybridModel
from utils.dataset import DeepfakeVideoDataset


# --------------------------------------------------
# Argument parser
# --------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser("Final Test Evaluation")

    parser.add_argument("--ckpt", required=True, help="Path to best_model.pth")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--frames_root", required=True)

    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cuda", "cpu"],
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.74,   # 🔥 optimized threshold
        help="Decision threshold for FAKE class",
    )

    return parser.parse_args()


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------
@torch.no_grad()
def main():
    args = get_args()

    # Device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[INFO] Using device: {device}")

    # ---------------- Dataset ----------------
    test_ds = DeepfakeVideoDataset(
        csv_path=args.test_csv,
        frames_root=args.frames_root,
        seq_len=args.seq_len,
        augment=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    # ---------------- Model ----------------
    model = HybridModel(seq_len=args.seq_len).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    probs_all = []
    labels_all = []

    # ---------------- Inference ----------------
    for frames, labels in test_loader:
        frames = frames.to(device)

        logits = model(frames)                     # (B, 2)
        probs = torch.softmax(logits, dim=1)[:, 1] # FAKE probability

        probs_all.extend(probs.cpu().numpy())
        labels_all.extend(labels.numpy())

    probs_all = np.array(probs_all)
    labels_all = np.array(labels_all)

    preds = (probs_all >= args.threshold).astype(int)

    # ---------------- Metrics ----------------
    acc = accuracy_score(labels_all, preds)
    prec = precision_score(labels_all, preds)
    rec = recall_score(labels_all, preds)
    f1 = f1_score(labels_all, preds)
    auc = roc_auc_score(labels_all, probs_all)

    print("\n===== FINAL TEST RESULTS =====")
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Precision : {prec*100:.2f}%")
    print(f"Recall    : {rec*100:.2f}%")
    print(f"F1-score  : {f1*100:.2f}%")
    print(f"AUC       : {auc*100:.2f}%")
    print(f"Threshold : {args.threshold:.2f}")


# --------------------------------------------------
if __name__ == "__main__":
    main()
