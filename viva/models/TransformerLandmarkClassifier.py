import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerLandmarkClassifier(pl.LightningModule):
    def __init__(
            self,
            input_size: int = 478 * 3,  # Number of input features
            seq_length: int = 10,  # Length of the input sequence
            num_classes: int = 2,  # Number of output classes
            d_model: int = 512,  # Embedding size for each input
            nhead: int = 8,  # Number of attention heads
            num_layers: int = 3,  # Number of transformer layers
            dim_feedforward: int = 1024,  # Hidden layer size in feedforward network
            dropout: float = 0.2,  # Dropout rate
            learning_rate: float = 1e-3,
            loss_fn: nn.Module = nn.CrossEntropyLoss()
    ):
        super(TransformerLandmarkClassifier, self).__init__()
        self.save_hyperparameters(ignore=['loss_fn'])

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, input_size))

        # Linear projection for input features to d_model
        self.input_projection = nn.Linear(input_size, d_model)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output classification layer
        self.fc = nn.Linear(d_model, num_classes)

        # Loss function
        self.loss_fn = loss_fn

        # Accuracy metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Project input to d_model
        x = self.input_projection(x)

        # Pass through the transformer encoder
        x = self.transformer_encoder(x)

        # Global average pooling (reduce across sequence dimension)
        x = x.mean(dim=1)

        # Final classification layer
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, y.argmax(dim=1))
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y.argmax(dim=1))
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def on_train_epoch_end(self):
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        self.val_accuracy.reset()
