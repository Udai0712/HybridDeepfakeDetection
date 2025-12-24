import torch
import cv2
import numpy as np
from torchvision import transforms

from models.hybrid_model import HybridModel

# ------------------ transforms ------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------ load model ------------------
def load_model(checkpoint_path, device, seq_len=None):
    model = HybridModel(freeze_backbone=False)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


# ------------------ video inference ------------------
@torch.no_grad()
def infer_video(video_path, model, device, threshold=0.5, seq_len=16):
    cap = cv2.VideoCapture(video_path)

    frames = []
    while len(frames) < seq_len:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)

    cap.release()

    if len(frames) < seq_len:
        return 0, 0.0  # default REAL if video too short

    frames = torch.stack(frames).unsqueeze(0).to(device)
    logits = model(frames)
    probs = torch.softmax(logits, dim=1)

    fake_prob = probs[0, 1].item()
    label = 1 if fake_prob >= threshold else 0

    return label, fake_prob
