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
# Load model
# --------------------------------------------------
def load_model(ckpt_path, device, seq_len):
    model = HybridModel(seq_len=seq_len).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# --------------------------------------------------
# Evaluate
# --------------------------------------------------
def evaluate(args):
    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    # Dataset
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

    # Model
    model = load_model(args.checkpoint, device, args.seq_len)

    criterion = nn.CrossEntropyLoss()

    y_true, y_pred, y_prob = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for frames, labels in tqdm(test_loader, desc="Evaluating"):
            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs > args.threshold).long()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    metrics = compute_metrics(y_true, y_pred, y_prob)

    print("\n===== Test Results =====")
    print(f"Test Loss: {total_loss / len(test_loader):.4f}")

    print("\n===== Evaluation Metrics =====")
    print(f"ACCURACY  : {metrics['accuracy']:.4f}")
    print(f"PRECISION : {metrics['precision']:.4f}")
    print(f"RECALL    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"AUC       : {metrics['auc']:.4f}")
    print("===============================")


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hybrid Deepfake Detection Evaluation")

    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--frames_root", type=str, default="data/frames")
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.38,  # 🔥 BEST threshold from search
    )

    args = parser.parse_args()
    evaluate(args)
