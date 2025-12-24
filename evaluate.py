import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from models.hybrid_model import HybridModel
from utils.dataset import VideoDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    best_thresh = ckpt.get("best_thresh", 0.5)

    test_ds = VideoDataset(args.test_csv, args.frames_root, args.seq_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = HybridModel(seq_len=args.seq_len).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    probs_all = []
    labels_all = []

    for x, y in test_loader:
        x = x.to(device)
        logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits)

        probs_all.extend(probs.cpu().numpy())
        labels_all.extend(y.numpy())

    probs_all = np.array(probs_all)
    labels_all = np.array(labels_all)

    preds = (probs_all >= best_thresh).astype(int)

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
    print(f"Threshold : {best_thresh:.2f}")


if __name__ == "__main__":
    main()
