from typing import Optional, Union, List

import numpy as np
import torch

from viva.audio.VADModels import VADResult, VADState


class SileroVAD:
    def __init__(self,
                 model: Optional[torch.nn.Module] = None,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30
                 ):

        if model is None:
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                verbose=False
            )

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator supports sampling rates of 8000 or 16000 only.')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000

        # states
        self.triggered: bool = False
        self.temp_end: int = 0
        self.current_sample: int = 0

        self.reset_states()

    def reset_states(self) -> None:
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def process(self, x: Union[np.ndarray, torch.Tensor], window_size: int = 512) -> List[VADResult]:
        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except Exception:
                raise TypeError("Audio cannot be cast to tensor. Please cast it manually.")

        results = []
        for i in range(0, len(x), window_size):
            chunk = x[i:i + window_size]

            if len(chunk) != window_size:
                continue

            result = self.process_sample(chunk)
            if result is not None:
                results.append(result)
        return results

    def process_sample(self, x: Union[np.ndarray, torch.Tensor]) -> Optional[VADResult]:
        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except Exception:
                raise TypeError("Audio cannot be cast to tensor. Please cast it manually.")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if speech_prob >= self.threshold and self.temp_end:
            self.temp_end = 0

        if speech_prob >= self.threshold and not self.triggered:
            self.triggered = True
            speech_start = self.current_sample - self.speech_pad_samples
            return VADResult(VADState.Started, int(speech_start))

        if speech_prob < self.threshold - 0.15 and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples
                self.temp_end = 0
                self.triggered = False
                return VADResult(VADState.Ended, int(speech_end))

        return None
