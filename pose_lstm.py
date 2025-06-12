import torch
import torch.nn as nn


class PoseLSTMClassifier(nn.Module):
    """Классифицирует последовательность поз на категории выполнения."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, 3)  # 3 класса

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Принимает тензор формы (batch, seq_len, input_size)."""
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]
        return self.classifier(out)
