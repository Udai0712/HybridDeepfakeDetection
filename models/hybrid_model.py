import torch
import torch.nn as nn
from models.xception import XceptionNet
from models.bilstm_attention import BiLSTMAttention


class HybridModel(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()

        # -------------------------------
        # Xception backbone
        # -------------------------------
        self.backbone = XceptionNet(
            pretrained=True,
            freeze_backbone=freeze_backbone
        )

        # -------------------------------
        # Temporal modeling
        # -------------------------------
        self.temporal = BiLSTMAttention(
            input_dim=2048,
            hidden_dim=256
        )

        # -------------------------------
        # Classifier (MATCHES CHECKPOINT)
        # -------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),   # classifier.0
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # classifier.3
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        # Frame-wise feature extraction
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)          # (B*T, 2048)

        # Temporal sequence
        feats = feats.view(B, T, -1)      # (B, T, 2048)
        temporal_feat = self.temporal(feats)  # (B, 512)

        # Classification
        return self.classifier(temporal_feat)
