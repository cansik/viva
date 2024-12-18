# Viva
Visual voice activity detector.

## Installation

Please install `uv` (`pip install uv`) and use uv to install all the necessary packages:

```bash
uv sync
```

## Usage

### Pre-Process
- Please download the WildVVAD dataset (speaking and silent) from [here](https://team.inria.fr/perception/research/vvad/).
- Unpack the contents, so that there are two subfolders, with speaking and silent videos.
- Use `viva preprocess` to process the videos in both folders.

```bash
python -m viva preprocess ./silent
python -m viva preprocess ./speaking --speaking
```

### Dataset
After that, create the dataset split using the `viva dataset` tool.

```bash
tbd
```

## Tools

- Dataset preparation (load all videos and analyse them with whisper speech / simplified audio analysis pipeline)
- Model Training (LSTM based model which)
- Model Testing
- Model live test