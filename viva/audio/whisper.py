import platform
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Tuple, List, Dict

import numpy as np


class BaseWhisper(ABC):
    def transcribe(
            self,
            audio_file: Union[str, Any],
            verbose: Optional[bool] = None,
            temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold: Optional[float] = 2.4,
            logprob_threshold: Optional[float] = -1.0,
            no_speech_threshold: Optional[float] = 0.6,
            condition_on_previous_text: bool = True,
            initial_prompt: Optional[str] = None,
            word_timestamps: bool = False,
            prepend_punctuations: str = "\"'“¿([{-",
            append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
            clip_timestamps: Union[str, List[float]] = "0",
            hallucination_silence_threshold: Optional[float] = None,
            **decode_options,
    ) -> Dict:
        options = {
            "audio": audio_file,
            "verbose": verbose,
            "temperature": temperature,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "condition_on_previous_text": condition_on_previous_text,
            "initial_prompt": initial_prompt,
            "word_timestamps": word_timestamps,
            "prepend_punctuations": prepend_punctuations,
            "append_punctuations": append_punctuations,
            "clip_timestamps": clip_timestamps,
            "hallucination_silence_threshold": hallucination_silence_threshold,
            **decode_options,
        }
        return self._transcribe(**options)

    @abstractmethod
    def _transcribe(self, **options) -> Dict:
        pass


class OpenAIWhisper(BaseWhisper):
    def __init__(self, model_name: str = "base"):
        import whisper

        self.whisper = whisper
        self.model = whisper.load_model(model_name)

    def _transcribe(self, **options) -> Dict:
        return self.model.transcribe(**options)


class MLXWhisper(BaseWhisper):
    def __init__(self, model_name: str = "mlx-community/whisper-tiny-mlx"):
        import mlx_whisper

        self.mlx_whisper = mlx_whisper
        self.model_name = model_name

        # Pre-init the whisper model
        _ = mlx_whisper.transcribe(np.zeros(512, dtype=np.float32), path_or_hf_repo=self.model_name)

    def _transcribe(self, **options) -> Dict:
        return self.mlx_whisper.transcribe(path_or_hf_repo=self.model_name, **options)


def create_whisper() -> BaseWhisper:
    current_os = platform.system().lower()

    if current_os in ["windows", "linux"]:
        return OpenAIWhisper()
    elif current_os == "darwin":  # macOS
        return MLXWhisper()
    else:
        raise ValueError(f"Unsupported operating system: {current_os}")
