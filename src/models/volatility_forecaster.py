import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Optional

class VolatilityForecaster(pl.LightningModule):
    """
    LSTM/GRU-based volatility forecaster with optional GARCH component.
    """
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        # TODO: Integrate GARCH component
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr) 