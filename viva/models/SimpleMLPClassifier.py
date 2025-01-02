from typing import Tuple, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics


class SimpleMLPClassifier(pl.LightningModule):
    def __init__(
            self,
            input_size: int,
            num_classes: int = 2,
            hidden_units: Tuple[int, ...] = (512, 256, 128, 64),
            learning_rate: float = 1e-3,
            loss_fn: Callable = nn.CrossEntropyLoss()
    ):
        super(SimpleMLPClassifier, self).__init__()
        self.save_hyperparameters()

        layers = []
        in_features = input_size

        # Build hidden layers
        for hidden in hidden_units:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_features = hidden

        # Output layer
        layers.append(nn.Linear(in_features, num_classes))

        self.model = nn.Sequential(*layers)
        self.loss_fn = self.hparams.loss_fn

        # Add accuracy metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Calculate and log training accuracy
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, y.argmax(dim=1))
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Calculate and log validation accuracy
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
