import argparse

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset
from viva.data.VideoPreProcessor import VideoPreProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("mode", help="Which mode to start viva in.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = str(args.mode)

    if mode == "data":
        # p = AVPreProcessor("data/", "data/")
        p = VideoPreProcessor("wildvvad/", "wildvvad/")
        p.process(num_workers=1)
    elif mode == "dataset":
        dataset = FaceLandmarkDataset("wildvvad/")
        first = dataset[0]


if __name__ == "__main__":
    main()
