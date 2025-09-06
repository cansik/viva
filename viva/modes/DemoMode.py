import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from visiongraph import vg
from visiongraph.result.ResultDict import DEFAULT_IMAGE_KEY
from visiongraph.result.spatial.face.BlazeFaceMesh import BlazeFaceMesh

from viva.data.augmentations.BaseLandmarkAugmentation import BaseLandmarkAugmentation
from viva.data.augmentations.FilterLandmarkIndices import FilterLandmarkIndices
from viva.data.augmentations.FlattenLandmarks import FlattenLandmarks
from viva.models.ImprovedTCNLandmarkClassifier import ImprovedTCNLandmarkClassifier
from viva.modes.VivaBaseMode import VivaBaseMode
from viva.utils.RollingBuffer import RollingBuffer


@dataclass
class VVADResult:
    speaking: bool
    speaking_confidence: float
    non_speaking_confidence: float


class TCNPredictor:
    def __init__(self, checkpoint_path: Path, block_size: int):
        self.checkpoint_path = checkpoint_path
        self.block_size = block_size

        self.transforms: List[BaseLandmarkAugmentation] = [
            FilterLandmarkIndices(vg.BlazeFaceMesh.FEATURES_148),
            FlattenLandmarks(full=False),
        ]

        self.model = ImprovedTCNLandmarkClassifier.load_from_checkpoint(str(checkpoint_path))
        self.model.eval()

        # Initialize the rolling buffer
        feature_shape = len(vg.BlazeFaceMesh.FEATURES_148) * 3,
        self.buffer = RollingBuffer(block_size=block_size, feature_shape=feature_shape)
        self.landmark_buffer = RollingBuffer(block_size=block_size, feature_shape=(478, 3))

    def predict(self, face_mesh: BlazeFaceMesh) -> VVADResult:
        landmarks = face_mesh.normalize_landmarks()
        self.landmark_buffer.add(landmarks)

        landmarks = self._transform_landmarks(landmarks)

        # Add landmarks to the rolling buffer
        self.buffer.add(landmarks)

        # Get the current buffer state and convert to a PyTorch tensor
        data = self.buffer.get()
        tensor_landmarks = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        tensor_landmarks = tensor_landmarks.to(self.model.device)

        # Perform prediction
        logits = self.model(tensor_landmarks)
        softmax = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()  # Get the predicted class

        # Return the prediction as a boolean for binary classification
        return VVADResult(bool(predicted_class),
                          speaking_confidence=float(softmax[0][1]),
                          non_speaking_confidence=float(softmax[0][0]))

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

        norm_image = np.zeros((512, 512, 3), dtype=np.uint8)

        def run(data: vg.ResultDict) -> vg.ResultDict:
            image = data[DEFAULT_IMAGE_KEY]
            face_mesh_results: vg.ResultList[vg.BlazeFaceMesh] = data["face_mesh"]

            if len(face_mesh_results) > 0:
                face_mesh = face_mesh_results[0]
                result = self.predictor.predict(face_mesh)
                normalized_landmarks = self.predictor.landmark_buffer.get()

                is_speaking = result.speaking
                is_speaking = result.speaking_confidence > 0.4
                # todo: find out if softmax is necessary
                color = (0, 255, 0) if is_speaking else (0, 0, 255)
                text = "Speaking" if is_speaking else "Nothing"

                cv2.putText(image,
                            f"{text} ({result.speaking_confidence:.2f} | {result.non_speaking_confidence:.2f})",
                            (15, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, color)

                h, w = image.shape[:2]
                for lm_index in face_mesh.FEATURES_148:
                    lm = face_mesh.landmarks[lm_index]
                    center = round(lm.x * w), round(lm.y * h)
                    cv2.circle(image, center, 1, color, -1)

                # visualize normalisation
                h, w = norm_image.shape[:2]
                samples = normalized_landmarks
                samples[:, :, 0] = ((samples[:, :, 0] + 1) / 2) * w
                samples[:, :, 1] = h - ((samples[:, :, 1] + 1) / 2) * h

                color = (0, 255, 255)

                norm_image.fill(0)
                sample = samples[-1]
                for lm in sample:
                    center = round(lm[0]), round(lm[1])
                    cv2.circle(norm_image, center, 1, color, -1)
                cv2.imshow("Normalized Landmarks", norm_image)

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
                # vg.ResultAnnotator(),
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
