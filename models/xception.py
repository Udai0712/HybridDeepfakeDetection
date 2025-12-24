import torch
import torch.nn as nn
import timm


class XceptionNet(nn.Module):
    """
    Xception backbone for frame-level deepfake feature extraction.

    Output:
        (B, feature_dim) per frame
    """

    def __init__(
        self,
        pretrained: bool = True,
        feature_dim: int = 2048,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        checkpoint_path: str | None = None,
    ):
        super().__init__()

        # Load Xception backbone
        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,        # remove classifier
            global_pool="avg"
        )

        backbone_out = self.backbone.num_features  # usually 2048

        # Feature projection head
        self.feature_head = nn.Sequential(
            nn.Linear(backbone_out, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Load optional deepfake-pretrained checkpoint
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        # Freeze backbone initially (recommended)
        if freeze_backbone:
            self.freeze_backbone()

    # ----------------- Utility methods -----------------

    def freeze_backbone(self):
        """Freeze Xception backbone"""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze Xception backbone for fine-tuning"""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def _load_checkpoint(self, path: str):
        """Load weights safely (ignores mismatched layers)"""
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]

        # Remove classifier keys if present
        filtered = {k.replace("backbone.", ""): v
                    for k, v in state.items()
                    if k.startswith("backbone.")}

        self.backbone.load_state_dict(filtered, strict=False)
        print(f"[INFO] Loaded Xception weights from {path}")

    # ----------------- Forward -----------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224)

        Returns:
            features: (B, feature_dim)
        """
        feats = self.backbone(x)       # (B, backbone_out)
        feats = self.feature_head(feats)
        return feats
