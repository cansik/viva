import json
from pathlib import Path

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset


def main():
    dataset_path = Path("wildvvad") / "dataset.json"

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    train_dataset = FaceLandmarkDataset(metadata_paths=data["train"], block_length=15)

    for r in range(100):
        for i in range(len(train_dataset)):
            data = train_dataset[i]

    print("done!")


if __name__ == "__main__":
    main()
