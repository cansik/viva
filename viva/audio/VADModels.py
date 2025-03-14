from dataclasses import dataclass
from enum import Enum
from typing import List


class VADState(Enum):
    Started = 0
    Ended = 1


@dataclass
class VADResult:
    state: VADState
    sample_position: int


@dataclass
class VADSegment:
    start: int
    end: int


def convert_vad_results_to_segments(results: List[VADResult], max_samples: int) -> List[VADSegment]:
    segments: List[VADSegment] = []
    current_start = None

    for result in results:
        if result.state == VADState.Started:
            # Mark the start of a new segment
            current_start = result.sample_position
        elif result.state == VADState.Ended:
            if current_start is not None:
                # When we see an end and have a start, complete the segment.
                segments.append(VADSegment(start=current_start, end=result.sample_position))
                current_start = None
            else:
                # Optionally, handle the case where an "Ended" is encountered without a corresponding "Started"
                print(f"Warning: End encountered at position {result.sample_position} without a matching start.")

    # Optionally, warn if there's a started segment that never ended
    if current_start is not None:
        segments.append(VADSegment(start=current_start, end=max_samples - 1))

    return segments
