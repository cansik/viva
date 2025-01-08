import json
from pathlib import Path

from tqdm import tqdm
from visiongraph import vg

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset
from viva.data.augmentations.CollapseLabels import CollapseLabels
from viva.data.augmentations.FilterLandmarkIndices import FilterLandmarkIndices
from viva.data.augmentations.FlattenLandmarks import FlattenLandmarks
from viva.data.augmentations.NormalizeLandmarks import NormalizeLandmarks
from viva.data.augmentations.OneHotEncodeLabels import OneHotEncodeLabels


def main():
    dataset_path = Path("viva-data") / "dataset.json"

    with vg.Watch("Open Dataset"):
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        train_dataset = FaceLandmarkDataset(metadata_paths=data["train"], block_length=15,
                                            transforms=[
                                                NormalizeLandmarks(),
                                                FilterLandmarkIndices(vg.BlazeFaceMesh.FEATURES_148),
                                                FlattenLandmarks(full=True),

                                                CollapseLabels(),
                                                OneHotEncodeLabels()
                                            ])

    # print("loading all data and saving to zarr")
    # zarr_path = dataset_path.with_name("train.zarr")
    # train_dataset.save_as_zarr(zarr_path)

    # series = zarr_io.load_from_zarr(zarr_path)

    # series2 = load_face_landmark_series_in_parallel(train_dataset.metadata_paths, max_threads=4)

    # series = []
    # for path in tqdm(train_dataset.metadata_paths):
    #    series.append(FaceLandmarkSeries.load(path))

    print(f"Length: {len(train_dataset)}")
    index_range = range(len(train_dataset))
    with vg.Watch("Loading Samples"):
        for i in tqdm(index_range):
            data = train_dataset[i]

    print("done!")


if __name__ == "__main__":
    main()
