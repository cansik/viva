import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=52, hidden_dim=64, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        # Use the final hidden state as the sequence encoding
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # shape: (batch, hidden_dim)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Path to the dataset file.")
    parser.add_argument("--mode", default="train", type=str, help="Which mode to select.")
    return parser.parse_args()


def extract_features(dataset, encoder, device):
    encoder.eval()
    features, labels = [], []

    # Wrap the dataset in a DataLoader for batch processing
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=lambda batch: batch)
    with torch.no_grad():
        for batch in loader:
            # Each batch item is a tuple (x, y)
            x_batch = [torch.tensor(item[0], dtype=torch.float32) for item in batch]
            y_batch = [item[1][0] for item in batch]  # assume label consistency within sequence

            # Stack sequences (batch, seq_len, features)
            x_tensor = torch.stack(x_batch, dim=0).to(device)
            # Get sequence encoding from LSTM
            feat_batch = encoder(x_tensor)  # shape: (batch, hidden_dim)
            features.append(feat_batch.cpu().numpy())
            labels.extend(y_batch)
    return np.concatenate(features, axis=0), np.array(labels)


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    dataset_mode = str(args.mode)

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset = FaceLandmarkDataset(
        metadata_paths=data[dataset_mode],
        block_length=10,
        stride=1,
        use_blend_shapes=True,
    )

    # Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = LSTMEncoder(input_dim=52, hidden_dim=64).to(device)

    # If you have a trained encoder, load the weights here.
    # For demonstration, we assume the encoder is either pre-trained or you plan to train it.
    # Otherwise, you could train it on a classification or autoencoding task.

    # Extract features using the sequence encoder
    X, y = extract_features(dataset, encoder, device)

    # ----- Dimensionality Reduction for Visualization -----

    # Using PCA to reduce dimensions to 2D.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=60)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Visualization of LSTM-Encoded Face Landmark Data")
    plt.colorbar(scatter, label="Speaking (1) vs Not Speaking (0)")
    plt.show()

    # Alternatively, you can also use t-SNE:
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="viridis", s=60)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title("t-SNE Visualization of LSTM-Encoded Face Landmark Data")
    plt.colorbar(scatter, label="Speaking (1) vs Not Speaking (0)")
    plt.show()

    # ----- Feature Importance via Logistic Regression on Encoded Features -----

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    coef = clf.coef_[0]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(coef)), np.abs(coef))
    plt.xlabel("Encoded Feature Index")
    plt.ylabel("Absolute Coefficient")
    plt.title("Encoded Feature Importance (Logistic Regression)")
    plt.show()

    sorted_idx = np.argsort(np.abs(coef))[::-1]
    print("Encoded feature indices sorted by importance (most influential first):")
    for idx in sorted_idx:
        print(f"Feature {idx}: coefficient = {coef[idx]:.4f}")


if __name__ == "__main__":
    main()
