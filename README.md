# Viva - Visual Voice Activity Detector (VVAD)

Viva aims to detect voice activity solely by analyzing a person's face. This approach is useful in situations where relying on acoustic detection is challenging, such as in museums, where beamforming and source localization require specialized hardware and careful calibration. For chatbot scenarios, where accurate turn-taking and interruption handling are crucial, a visual method can be more effective, as the camera naturally limits the field of view. It can even support multi-person interactions with a chatbot, allowing the system to identify and respond to the specific person who is speaking without needing to separate overlapping voices.

This repository serves as a playground for a VVAD system built on facial landmark analysis over time. It includes an automatic video annotation pipeline, training code, and tools for testing and evaluation. Please note that this is an active research project and still under development.

![vvad-demo](https://github.com/user-attachments/assets/e8f454ad-0a81-4953-968b-e47b4bcdabcf)

*Source: [Pexels](https://www.pexels.com/video/a-woman-reading-a-script-while-doing-a-streaming-8552830/)*

## Installation

Install the required `uv` package and use it to synchronize all necessary dependencies:

```bash
pip install uv
uv sync
```

## Usage

### Pre-Processing
1. Download the **WildVVAD Dataset** (speaking and silent) from [this link](https://team.inria.fr/perception/research/vvad/).
2. Extract the dataset, ensuring it contains two subfolders: `silent` and `speaking`.
3. Preprocess the videos in both folders using the `viva preprocess` command:

```bash
python -m viva preprocess ./wildvvad/silent
python -m viva preprocess ./wildvvad/speaking --speaking
```

To optimize processing time, leverage multi-processing by specifying the number of workers with `--num-workers` (e.g., `--num-workers 8`).

### Creating the Dataset

After preprocessing, create dataset splits using the `viva dataset` tool:

```bash
python -m viva dataset --split ./wildvvad
```

This generates a `dataset.json` file containing paths to the processed files.

#### Balancing the Dataset
The dataset often has an imbalance (more speaking than non-speaking samples). Use the `--balance` argument to automatically balance it:

```bash
python -m viva dataset --split ./wildvvad --balance
```

### Data Inspection

Verify the integrity of the extracted data using the `inspect` tool:

```bash
python -m viva inspect wildvvad/dataset.json --mode val --stride 3 --block-size 5
```

#### Sampling Chunks
Define `stride` and `block-size` (also referred to as `block-length`) to split extracted frames into chunks:

- **`block-size`**: Number of samples in each chunk.
- **`stride`**: Number of samples skipped between chunks.

**Examples:**

```text
Samples: 0 1 2 3 4 5 6 7 8 9

# With block-size = 3 and stride = 1
Chunks: [0 1 2], [1 2 3], [2 3 4], ...

# With block-size = 3 and stride = 2
Chunks: [0 2 4], [1 3 5], [2 4 6], ...
```

Additional flags:
- `--samples`: Display individual samples.
- `--norm`: Display normalized landmarks.

### Training

Train the model using the `viva train` tool. Only the `block` training strategy is currently supported, which uses data chunks to detect voice activity.

Example training command:

```bash
python -m viva train wildvvad/dataset.json block batch_size=1024 network=tcn-i block_length=30 stride=1
```

You can overwrite specific training parameters using the format `param=value`.

#### Implemented Networks
The following networks are currently available:

- **`tcn`**: Temporal Convolutional Network with layers (64, 128, 256).
- **`tcn-i`**: Improved TCN with batch normalization and cyclic learning rate.
- **`mlp`**: Multi-Layer Perceptron with layers (512, 256, 128, 64).
- **`transformer`**: Simple transformer with 4 heads and 3 layers.
- **`lstm`**: LSTM with 2 layers of size 512.

#### Results
Training checkpoints are stored under the `runs/` directory, e.g.,

```text
runs/ImprovedTCNLandmarkClassifier/version_1/checkpoints/
```

Adjust the network name and version according to your training.

### Demo

To test the trained model, use the `demo` tool. Ensure the `block-size` matches the one used during training:

```bash
python -m viva demo runs/ImprovedTCNLandmarkClassifier/version_1/checkpoints/best.ckpt --block-size 10
```

### Export

Export the model to ONNX or OpenVINO format for deployment. Quantize the model using `nncf` with the `export` tool:

```bash
python -m viva export runs/ImprovedTCNLandmarkClassifier/version_1/checkpoints/best.ckpt wildvvad/dataset.json --block-length 30 --stride 1
```

For dynamic batch size exports, add the `--dynamic` flag.

## About

Copyright (c) 2025 Florian Bruggisser
