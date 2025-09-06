from typing import Tuple, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

from viva.models.TCNLandmarkClassifier import TCNLandmarkClassifier, TemporalBlock, TCN
from viva.utils.FocalLoss import FocalLoss


class ImprovedTCNLandmarkClassifier(TCNLandmarkClassifier):
    def __init__(self,
                 input_size: int = 478 * 3,
                 num_classes: int = 2,
                 num_channels: Tuple[int, ...] = (64, 128, 256),
                 kernel_size: int = 3,
                 dropout: float = 0.2,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 loss_fn: Callable = None,
                 max_lr: float = 1e-2,
                 use_focal_loss: bool = False,
                 focal_alpha: float = 1,
                 focal_gamma: float = 2):
        """
        Initializes the Improved TCN Landmark Classifier.

        Parameters:
            use_focal_loss (bool): If True, use Focal Loss; otherwise use loss_fn or default CrossEntropyLoss.
            focal_alpha (float): Alpha parameter for Focal Loss.
            focal_gamma (float): Gamma parameter for Focal Loss.
            loss_fn (Callable): Optional loss function; ignored if use_focal_loss is True.
        """
        # Choose the loss function based on the flag
        if use_focal_loss:
            loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        super(ImprovedTCNLandmarkClassifier, self).__init__(
            input_size=input_size,
            num_classes=num_classes,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            learning_rate=learning_rate,
            loss_fn=loss_fn
        )
        self.weight_decay = weight_decay
        self.max_lr = max_lr

    # Nested classes remain unchanged
    class TemporalBlockWithBN(TemporalBlock):
        def __init__(self,
                     in_channels: int,
                     out_channels: int,
                     kernel_size: int,
                     stride: int,
                     dilation: int,
                     padding: int,
                     dropout: float):
            super().__init__(in_channels, out_channels, kernel_size, stride, dilation, padding, dropout)
            # Add Batch Normalization layers
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)
            # Redefine the sequential block to include BN
            self.net = nn.Sequential(
                self.conv1, self.bn1, self.relu1, self.dropout1,
                self.conv2, self.bn2, self.relu2, self.dropout2
            )

    class TCNWithImprovements(TCN):
        def __init__(self,
                     input_size: int,
                     num_classes: int,
                     num_channels: Tuple[int, ...],
                     kernel_size: int = 3,
                     dropout: float = 0.2):
            super().__init__(input_size, num_classes, num_channels, kernel_size, dropout)
            self.layers = []
            for i in range(len(num_channels)):
                dilation_size = 2 ** i
                in_channels = input_size if i == 0 else num_channels[i - 1]
                out_channels = num_channels[i]
                self.layers.append(
                    ImprovedTCNLandmarkClassifier.TemporalBlockWithBN(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=1,
                        dilation=dilation_size,
                        padding=(kernel_size - 1) * dilation_size,
                        dropout=dropout,
                    )
                )
            self.network = nn.Sequential(*self.layers)
            self.fc = nn.Linear(num_channels[-1], num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.network(x)
            x = nn.AdaptiveAvgPool1d(1)(x).squeeze(-1)  # Adaptive pooling to get fixed-size representation
            return self.fc(x)

    def __init_model(self):
        self.model = ImprovedTCNLandmarkClassifier.TCNWithImprovements(
            self.hparams.input_size,
            self.hparams.num_classes,
            self.hparams.num_channels,
            self.hparams.kernel_size,
            self.hparams.dropout
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = CyclicLR(
            optimizer,
            base_lr=self.hparams.learning_rate,
            max_lr=self.max_lr,
            step_size_up=2000,
            mode="triangular"
        )
        return [optimizer], [scheduler]
