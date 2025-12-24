import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMAttention(nn.Module):
    """
    Bi-directional LSTM with Attention for temporal modeling.

    Input:
        (B, T, F)  -> sequence of frame-level features

    Output:
        (B, hidden_dim * 2) -> video-level representation
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)  sequence of features

        Returns:
            context: (B, hidden_dim * 2)
        """
        # LSTM output
        lstm_out, _ = self.lstm(x)     # (B, T, 2*hidden_dim)

        # Attention scores
        attn_scores = self.attention(lstm_out)   # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return context
