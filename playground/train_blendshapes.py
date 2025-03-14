import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Path to the dataset JSON file.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 regularization).")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate after LSTM.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--save_path", type=str, default="best_model.pth", help="Path to save the best model weights.")
    parser.add_argument("--block-size", type=int, default=20, help="Blocksize")
    return parser.parse_args()


def collate_fn(batch):
    """
    Each item in batch is a tuple (x, y):
      - x: numpy array of shape (10, 52)
      - y: a list/array of length 10 (assumed consistent; use first value as label)
    """
    xs, ys = [], []
    for x, y in batch:
        xs.append(x)
        ys.append(y[0])  # use the first label as the sequence label
    xs = torch.tensor(xs, dtype=torch.float32)
    # Convert ys from float32 (0.0 or 1.0) to int (0 or 1) for classification.
    ys = torch.tensor(ys, dtype=torch.float32).long()
    return xs, ys


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=52, hidden_dim=64, num_layers=1, num_classes=2, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, (hn, cn) = self.lstm(x)
        # Use the last hidden state of the final LSTM layer.
        final_state = hn[-1]  # shape: (batch, hidden_dim)
        final_state = self.dropout(final_state)
        logits = self.fc(final_state)  # shape: (batch, num_classes)
        return logits


def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for xs, ys in dataloader:
            xs = xs.to(device)
            ys = ys.to(device)
            outputs = model(xs)
            loss = criterion(outputs, ys)
            total_loss += loss.item() * xs.size(0)
            total_samples += xs.size(0)
    return total_loss / total_samples


def evaluate_accuracy(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for xs, ys in dataloader:
            xs = xs.to(device)
            ys = ys.to(device)
            outputs = model(xs)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == ys).sum().item()
            total_samples += ys.size(0)
    return total_correct / total_samples


def main() -> None:
    args = parse_args()

    block_size = int(args.block_size)
    stride = 1

    # Read dataset JSON file
    dataset_path = Path(args.dataset)
    data = json.loads(dataset_path.read_text(encoding="utf-8"))

    # Create datasets using the provided metadata paths.
    train_dataset = FaceLandmarkDataset(
        metadata_paths=data["train"],
        block_length=block_size,
        stride=stride,
        use_blend_shapes=True,
    )
    val_dataset = FaceLandmarkDataset(
        metadata_paths=data["val"],
        block_length=block_size,
        stride=stride,
        use_blend_shapes=True,
    )
    test_dataset = FaceLandmarkDataset(
        metadata_paths=data["test"],
        block_length=block_size,
        stride=stride,
        use_blend_shapes=True,
    )

    # Create dataloaders.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Set up device and model.
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = LSTMClassifier(input_dim=52, hidden_dim=64, num_layers=4, num_classes=2, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_acc = 0.0
    epochs_without_improvement = 0

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for xs, ys in train_loader:
            xs = xs.to(device)
            ys = ys.to(device)

            optimizer.zero_grad()
            outputs = model(xs)
            loss = criterion(outputs, ys)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}/{args.epochs}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc * 100:.2f}%")

        # Early stopping and best model saving.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
            # Save the best model weights to disk.
            torch.save(best_model_state, args.save_path)
            print(f"Saved best model with Val Accuracy: {val_acc * 100:.2f}%")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"No improvement in validation accuracy for {args.patience} epochs. Early stopping.")
            break

    # Load the best model before testing.
    model.load_state_dict(torch.load(args.save_path))
    test_acc = evaluate_accuracy(model, test_loader, device)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
