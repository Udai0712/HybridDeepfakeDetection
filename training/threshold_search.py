# training/threshold_search.py
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.hybrid_model import HybridModel
from utils.dataset import VideoDataset
from utils.inference import infer_video_probs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid Deepfake Detection Threshold Search"
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        required=True,
        help="Validation CSV file"
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/frames",
        help="Root directory of extracted frames"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (keep 1 for video-level inference)"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=16,
        help="Number of frames per video sequence"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["mps", "cuda", "cpu"],
        default="cpu"
    )
    return parser.parse_args()


def search_threshold(args):
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # ---------------- Load model ----------------
    model = HybridModel()
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # handle different checkpoint formats safely
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # ---------------- Load dataset ----------------
    dataset = VideoDataset(
        csv_file=args.val_csv,
        frames_root=args.frames_root,
        seq_len=args.seq_len,
        training=False
    )

    all_probs = []
    all_labels = []

    print("[INFO] Running inference on validation set...")

    for idx in tqdm(range(len(dataset))):
        video_tensor, label = dataset[idx]

        video_tensor = video_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(video_tensor)
            prob = torch.sigmoid(logits).item()

        all_probs.append(prob)
        all_labels.append(label)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ---------------- Threshold search ----------------
    print("\n[INFO] Searching best threshold...\n")

    best_f1 = 0.0
    best_threshold = 0.5

    for th in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs >= th).astype(int)

        acc = accuracy_score(all_labels, preds)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)

        print(
            f"Threshold {th:.2f} | "
            f"Acc {acc:.3f} | "
            f"Prec {prec:.3f} | "
            f"Recall {rec:.3f} | "
            f"F1 {f1:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    print("\n==============================")
    print(f"BEST THRESHOLD : {best_threshold:.2f}")
    print(f"BEST F1 SCORE  : {best_f1:.3f}")
    print("==============================\n")


if __name__ == "__main__":
    args = parse_args()
    search_threshold(args)
