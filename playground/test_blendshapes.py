import argparse

import cv2
import numpy as np
import torch
from visiongraph import vg
from visiongraph.result.ResultDict import DEFAULT_IMAGE_KEY

from playground.train_blendshapes import LSTMClassifier
from viva.utils.RollingBuffer import RollingBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    vg.VisionGraph.add_params(parser)
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to the saved model weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set up the device and load the trained model.
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"using device: {device}")
    model = LSTMClassifier(input_dim=52, hidden_dim=64, num_layers=4, num_classes=2, dropout=0.5).to(device)
    model_data = torch.load(args.model_path, map_location=device)
    model.load_state_dict(model_data)
    model.eval()

    # Create a rolling buffer to collect a fixed-length sequence.
    # Ensure block_size here matches the sequence length the model was trained on.
    buffer = RollingBuffer(block_size=10, feature_shape=(52,))

    def run(data: vg.ResultDict) -> vg.ResultDict:
        image = data[DEFAULT_IMAGE_KEY]
        face_mesh_results = data["face_mesh"]

        if len(face_mesh_results) > 0:
            face_mesh = face_mesh_results[0]

            # Extract blendshapes from the face mesh result.
            blend_shapes = np.array([bs.value for bs in face_mesh.blend_shapes], np.float32)
            buffer.add(blend_shapes)

            buffer_data = buffer.get()  # Shape: (block_size, 52)
            # Prepare the input tensor: add batch dimension -> (1, block_size, 52)
            input_tensor = torch.tensor(buffer_data, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)
                _, pred = torch.max(logits, 1)
                pred_class = pred.item()
                pred_score = float(torch.softmax(logits, 1)[0][pred_class])

            # Annotate the image with the inference result.
            text = "Speaking" if pred_class == 1 else "Not Speaking"
            text = f"{text} ({pred_score * 100:.1f}%)"
            cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return data

    graph = (
        vg.create_graph(name="Viva BS Demo", input_node=args.input(), handle_signals=True)
        .apply(
            image=vg.passthrough(),
            face_mesh=vg.MediaPipeFaceMeshEstimator(
                output_facial_transformation_matrixes=True,
                output_face_blendshapes=True,
            ),
        )
        .then(
            vg.custom(run),
            vg.ImagePreview("Preview")
        )
        .build()
    )
    graph.configure(args)
    graph.open()


if __name__ == "__main__":
    main()
