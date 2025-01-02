from typing import Tuple, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics


class LSTM(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            num_classes: int,
            dropout: float = 0.2
    ):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM expects input of shape (B, T, C)
        lstm_out, _ = self.lstm(x)  # Output shape: (B, T, hidden_size)
        x = lstm_out[:, -1, :]  # Take the output from the last time step
        return self.fc(x)


class LSTMLandmarkClassifier(pl.LightningModule):
    def __init__(
            self,
            input_size: int = 478 * 3,
            num_classes: int = 2,
            hidden_size: int = 512,
            num_layers: int = 2,
            dropout: float = 0.2,
            learning_rate: float = 1e-3,
            loss_fn: Callable = nn.CrossEntropyLoss()
    ):
        super(LSTMLandmarkClassifier, self).__init__()
        self.save_hyperparameters(ignore=['loss_fn'])

        self.model = LSTM(
            input_size=self.hparams.input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            num_classes=self.hparams.num_classes,
            dropout=self.hparams.dropout,
        )
        self.loss_fn = loss_fn

        # Add accuracy metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Update and log accuracy
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, y.argmax(dim=1))
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Update and log accuracy
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y.argmax(dim=1))
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def on_train_epoch_end(self):
        # Reset metrics at the end of the epoch
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        # Reset metrics at the end of the epoch
        self.val_accuracy.reset()
