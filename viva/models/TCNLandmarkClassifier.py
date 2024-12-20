from typing import Tuple, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim


class TemporalBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            padding: int,
            dropout: float
    ):
        super(TemporalBlock, self).__init__()
        # Use calculated padding to ensure same output size
        calculated_padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=calculated_padding
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=calculated_padding
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)


class TCN(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_classes: int,
            num_channels: Tuple[int, ...],
            kernel_size: int = 3,
            dropout: float = 0.2
    ):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = x.mean(dim=-1)  # Global average pooling
        return self.fc(x)


class TCNLandmarkClassifier(pl.LightningModule):
    def __init__(
            self,
            input_size: int = 478 * 3,
            num_classes: int = 2,
            num_channels: Tuple[int, ...] = (64, 128, 256),
            kernel_size: int = 3,
            dropout: float = 0.2,
            learning_rate: float = 1e-3,
            loss_fn: Callable = nn.CrossEntropyLoss()
    ):
        super(TCNLandmarkClassifier, self).__init__()
        self.save_hyperparameters()

        self.model = TCN(
            input_size=self.hparams.input_size,
            num_classes=self.hparams.num_classes,
            num_channels=self.hparams.num_channels,
            kernel_size=self.hparams.kernel_size,
            dropout=self.hparams.dropout,
        )
        self.loss_fn = self.hparams.loss_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input for TCN (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
