import argparse
import json
from pathlib import Path

import onnx
import pytorch_lightning as pl
import torch
from onnxsim import simplify
from rich.console import Console
from visiongraph import vg

from viva.data.FaceLandmarkDataset import FaceLandmarkDataset
from viva.data.augmentations.FilterLandmarkIndices import FilterLandmarkIndices
from viva.data.augmentations.FlattenLandmarks import FlattenLandmarks
from viva.data.augmentations.NormalizeLandmarks import NormalizeLandmarks
from viva.models.ImprovedTCNLandmarkClassifier import ImprovedTCNLandmarkClassifier
from viva.modes.VivaBaseMode import VivaBaseMode


class ExportMode(VivaBaseMode):
    def __init__(self, console: Console):
        super().__init__(console)

    def run(self):
        args = self._parse_args()
        checkpoint_path = Path(args.checkpoint)
        dataset_path = Path(args.dataset)
        block_length = int(args.block_length)
        stride = int(args.stride)
        use_dynamic_axis = bool(args.dynamic)

        # load dataset
        data = json.loads(dataset_path.read_text(encoding="utf-8"))

        val_dataset = FaceLandmarkDataset(metadata_paths=data["val"],
                                          block_length=block_length,
                                          stride=stride,
                                          transforms=[
                                              NormalizeLandmarks(),
                                              FilterLandmarkIndices(vg.BlazeFaceMesh.FEATURES_148),
                                              FlattenLandmarks(full=False)
                                          ])

        # load model from checkpoint
        model: pl.LightningModule = ImprovedTCNLandmarkClassifier.load_from_checkpoint(str(checkpoint_path))
        model.eval()

        # specify output paths
        onnx_path = checkpoint_path.with_suffix(".onnx")
        onnx_simplified_path = onnx_path.with_stem(f"{onnx_path.stem}-simplified")
        openvino_path = onnx_path.with_suffix(".xml")
        openvino_quantized_path = openvino_path.with_stem(f"{openvino_path.stem}-q8")

        # export to onnx
        with self.console.status("Exporting ONNX"):
            x, _ = val_dataset[0]
            input_tensor = torch.tensor(x, device=model.device).unsqueeze(0)

            dynamic_axes = None
            if use_dynamic_axis:
                dynamic_axes = {  # Dynamic axis allows for multi-batch prediction
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                }

            torch.onnx.export(
                model,
                input_tensor,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes
            )

        with self.console.status("Simplifying ONNX"):
            onnx_model = onnx.load(onnx_path)
            model_simplified, check = simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(model_simplified, onnx_simplified_path)

        with self.console.status("Checking ONNX"):
            onnx_model = onnx.load(onnx_simplified_path)
            onnx.checker.check_model(onnx_model)

        with self.console.status("Exporting OpenVINO"):
            import openvino.runtime as ov
            core = ov.Core()

            ov_model = core.read_model(onnx_simplified_path)
            ov.save_model(ov_model, openvino_path, compress_to_fp16=False)

        # Quantize OpenVINO with NNCF
        with self.console.status("Quantizing OpenVINO"):
            import nncf
            import openvino.runtime as ov

            def transform_fn(data_item):
                return data_item[0].numpy()

            nncf_val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                num_workers=1
            )
            calibration_dataset = nncf.Dataset(nncf_val_dataloader, transform_fn)

            quantized_model = nncf.quantize(
                ov_model,
                calibration_dataset,
                target_device=nncf.TargetDevice.ANY
            )

            # Save quantized model
            ov.save_model(quantized_model, openvino_quantized_path)

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="viva export")
        parser.add_argument("checkpoint", type=str, help="Checkpoint to load for inference.")
        parser.add_argument("dataset", type=str, help="Path to the dataset file.")
        parser.add_argument("--block-length", type=int, default=15,
                            help="Dataset block-length (how much data per inference block).")
        parser.add_argument("--stride", type=int, default=1, help="Stride of the samples.")
        parser.add_argument("--dynamic", action="store_true", help="Use dynamic axis for export.")
        return parser.parse_args()
