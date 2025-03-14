import argparse
from pathlib import Path

from rich.console import Console

from viva.data.pre_processor.AudioVisionPreProcessor import AudioVisionPreProcessor
from viva.data.pre_processor.VideoPreProcessor import VideoPreProcessingOptions, VideoPreProcessor
from viva.modes.VivaBaseMode import VivaBaseMode


class PreProcessMode(VivaBaseMode):
    def __init__(self, console: Console):
        super().__init__(console)

    def run(self):
        args = self._parse_args()
        data_path = Path(args.data)
        output_path = data_path if args.output is None else Path(args.output)
        is_speaking = bool(args.speaking)
        is_debug = bool(args.debug)
        use_audio_prediction = bool(args.audio)
        num_workers = int(args.num_workers)
        num_face_mesh_workers = int(args.num_face_mesh_workers)
        num_whisper_workers = int(args.num_whisper_workers)
        stream_block_size = int(args.block_size)
        cache_vad_output = bool(args.cache_vad)

        options = VideoPreProcessingOptions(stream_block_size=stream_block_size,
                                            is_speaking=is_speaking,
                                            is_debug=is_debug)

        if use_audio_prediction:
            p = AudioVisionPreProcessor(data_path, output_path, options,
                                        num_workers=num_workers,
                                        num_face_mesh_workers=num_face_mesh_workers,
                                        num_vad_workers=num_whisper_workers,
                                        cache_vad_output=cache_vad_output)
        else:
            p = VideoPreProcessor(data_path, output_path, options, num_workers, num_face_mesh_workers)
        p.process()

        self.console.print("done!")

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="viva preprocess")
        parser.add_argument("data", type=str, help="Path to the data to pre-process.")
        parser.add_argument("--output", default=None, type=str, help="Output path, by default data-path.")
        parser.add_argument("--speaking", action="store_true", help="If the data is speaking data.")
        parser.add_argument("--block-size", default=100, type=int, help="Stream block size.")
        parser.add_argument("--debug", action="store_true", help="If is in debug mode.")
        parser.add_argument("--num-workers", default=4, type=int, help="Num parallel worker threads.")
        parser.add_argument("--num-face-mesh-workers", default=4, type=int, help="Num parallel face-mesh workers.")
        parser.add_argument("--audio", action="store_true", help="Use audio to predict speaking label.")
        parser.add_argument("--cache-vad", action="store_true", help="Cache vad output.")
        parser.add_argument("--num-whisper-workers", default=4, type=int, help="Num parallel whisper workers.")
        return parser.parse_args()
