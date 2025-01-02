from typing import Tuple, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics


class Transformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, max_seq_size: int = 15,
                 dim_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()

        self.input_proj = nn.Linear(input_size, d_model)  # Project input to d_model dimension

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_size, d_model))  # For 15 time steps

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)  # (B, T, d_model)
        x += self.positional_encoding[:, :x.size(1), :]  # Add positional encodings
        x = self.transformer_encoder(x)  # (B, T, d_model)
        x = x.mean(dim=1)  # Pool over time steps
        return self.fc(x)


class TransformerLandmarkClassifier(pl.LightningModule):
    def __init__(
            self,
            input_size: int = 148 * 3,
            num_classes: int = 2,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 3,
            max_seq_size: int = 15,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
            learning_rate: float = 1e-3,
            loss_fn: Callable = nn.CrossEntropyLoss()
    ):
        super(TransformerLandmarkClassifier, self).__init__()
        self.save_hyperparameters(ignore=['loss_fn'])

        self.model = Transformer(
            input_size=self.hparams.input_size,
            d_model=self.hparams.d_model,
            nhead=self.hparams.nhead,
            num_layers=self.hparams.num_layers,
            num_classes=self.hparams.num_classes,
            max_seq_size=max_seq_size,
            dim_feedforward=self.hparams.dim_feedforward,
            dropout=self.hparams.dropout,
        )

        # Initialize weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.model.apply(init_weights)

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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5),
                'monitor': 'val_loss',
            },
            'clip_grad_norm': 1.0,  # Clip gradients
        }

    def on_train_epoch_end(self):
        # Reset metrics at the end of the epoch
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        # Reset metrics at the end of the epoch
        self.val_accuracy.reset()
