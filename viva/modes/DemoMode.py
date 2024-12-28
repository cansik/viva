import argparse
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import torch
from rich.console import Console
from visiongraph import vg
from visiongraph.result.ResultDict import DEFAULT_IMAGE_KEY
from visiongraph.result.spatial.face.BlazeFaceMesh import BlazeFaceMesh

from viva.data.augmentations.BaseLandmarkAugmentation import BaseLandmarkAugmentation
from viva.data.augmentations.FilterLandmarkIndices import FilterLandmarkIndices
from viva.data.augmentations.FlattenLandmarks import FlattenLandmarks
from viva.models.TCNLandmarkClassifier import TCNLandmarkClassifier
from viva.modes.VivaBaseMode import VivaBaseMode
from viva.utils.RollingBuffer import RollingBuffer


class TCNPredictor:
    def __init__(self, checkpoint_path: Path, block_size: int):
        self.checkpoint_path = checkpoint_path
        self.block_size = block_size

        self.transforms: List[BaseLandmarkAugmentation] = [
            FilterLandmarkIndices(vg.BlazeFaceMesh.FEATURES_148),
            FlattenLandmarks(full=False),
        ]

        self.model = TCNLandmarkClassifier.load_from_checkpoint(str(checkpoint_path))
        self.model.eval()

        # Initialize the rolling buffer
        feature_shape = len(vg.BlazeFaceMesh.FEATURES_148) * 3,
        self.buffer = RollingBuffer(block_size=block_size, feature_shape=feature_shape)

    def predict(self, face_mesh: BlazeFaceMesh) -> bool:
        landmarks = self._transform_landmarks(face_mesh.normalize_landmarks())

        # Add landmarks to the rolling buffer
        self.buffer.add(landmarks)

        # Get the current buffer state and convert to a PyTorch tensor
        tensor_landmarks = torch.tensor(self.buffer.get(), dtype=torch.float32).unsqueeze(0)
        tensor_landmarks = tensor_landmarks.to(self.model.device)

        # Perform prediction
        logits = self.model(tensor_landmarks)
        predicted_class = torch.argmax(logits, dim=1).item()  # Get the predicted class

        # Return the prediction as a boolean for binary classification
        return bool(predicted_class)

    def _transform_landmarks(self, x: np.ndarray) -> np.ndarray:
        x = np.expand_dims(x, axis=0)
        for transform in self.transforms:
            x, _ = transform(x, None, None, None, None)
        return x[0]


class DemoMode(VivaBaseMode):
    def __init__(self, console: Console):
        super().__init__(console)

        self.predictor: Optional[TCNPredictor] = None

    def run(self):
        args = self._parse_args()
        checkpoint_path = Path(args.checkpoint)
        block_size = int(args.block_size)

        self.predictor = TCNPredictor(checkpoint_path, block_size)

        def run(data: vg.ResultDict) -> vg.ResultDict:
            image = data[DEFAULT_IMAGE_KEY]
            face_mesh_results: vg.ResultList[vg.BlazeFaceMesh] = data["face_mesh"]

            if len(face_mesh_results) > 0:
                is_speaking = self.predictor.predict(face_mesh_results[0])

                if is_speaking:
                    cv2.putText(image, f"Speaking!", (15, 30),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            return data

        graph = (
            vg.create_graph(name="Viva Demo", input_node=args.input(), handle_signals=True)

            # run detection and pass image through for annotation
            .apply(
                image=vg.passthrough(),
                face_mesh=vg.MediaPipeFaceMeshEstimator(output_facial_transformation_matrixes=True),
            )

            # annotate result
            .then(
                vg.custom(run),
                vg.ResultAnnotator(),
                vg.ImagePreview("Preview")
            )
            .build()
        )
        graph.configure(args)

        # start graph
        graph.open()

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="viva demo")
        parser.add_argument("checkpoint", type=str, help="Checkpoint to load for inference.")
        parser.add_argument("--block-size", type=int, default=15, help="Block size.")
        vg.VisionGraph.add_params(parser)
        return parser.parse_args()
