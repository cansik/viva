import time

import cv2
import ffmpegio
import numpy as np

from viva.audio.SileroVAD import SileroVAD
from viva.audio.VADModels import convert_vad_results_to_segments


def main():
    movie_file = "data/florian-short.mov"

    # read the audio from the movie file
    fs, x = ffmpegio.audio.read(str(movie_file), sample_fmt="dbl", ac=1, ar=16000)
    x = x.reshape(-1)

    vad = SileroVAD(sampling_rate=fs, speech_pad_ms_start=300, speech_pad_ms_end=30)

    start = time.time()
    results = vad.process(x.copy())
    end = time.time()
    print("VAD processing time:", end - start)

    vad_segments = convert_vad_results_to_segments(results)

    # read stream info for the video
    video_streams = ffmpegio.probe.video_streams_basic(str(movie_file))
    video_info = video_streams[0]

    # extract video information
    video_duration_seconds = float(video_info["duration"])
    video_fps = float(video_info["frame_rate"])
    total_video_frames = int(video_info["nb_frames"])
    video_width = int(video_info["width"])
    video_height = int(video_info["height"])

    video_frame_count = total_video_frames

    # convert VAD segments into per-frame speaking labels
    speaking_labels = np.full(video_frame_count, False, dtype=bool)
    for segment in vad_segments:
        # convert timestamps from audio samples to seconds
        start_ts = segment.start / fs
        end_ts = segment.end / fs

        start_frame_index = round(start_ts / video_duration_seconds * video_frame_count)
        end_frame_index = round(end_ts / video_duration_seconds * video_frame_count)

        if end_frame_index > video_frame_count:
            end_frame_index = video_frame_count

        speaking_labels[start_frame_index:end_frame_index] = True

    # playback controls variables
    paused = False
    frame_index = 0
    timeline_height = 50  # height in pixels of the timeline bar

    with ffmpegio.open(str(movie_file), "rv", blocksize=100) as fin:
        for frames in fin:
            video_frames: np.ndarray = frames

            for frame_rgb in video_frames:
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                preview = frame.copy()

                # create the timeline bar
                timeline_bar = np.full((timeline_height, video_width, 3), 255, dtype=np.uint8)
                # Map each column (pixel) to a video frame index.
                cols = np.arange(video_width)
                timeline_frame_indices = (cols / video_width * video_frame_count).astype(int)
                timeline_frame_indices = np.clip(timeline_frame_indices, 0, video_frame_count - 1)
                # Mark speaking regions in blue (BGR: (255, 0, 0))
                speaking_mask = speaking_labels[timeline_frame_indices]
                timeline_bar[:, speaking_mask] = (255, 0, 0)

                # Draw a red vertical cursor (BGR: (0, 0, 255)) for the current frame.
                cursor_x = int(frame_index / video_frame_count * video_width)
                cv2.line(timeline_bar, (cursor_x, 0), (cursor_x, timeline_height - 1), (0, 0, 255), thickness=2)

                # Overlay text label for the speaking status.
                current_status = "Speaking" if speaking_labels[frame_index] else "Not Speaking"
                cv2.putText(preview, current_status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=3)
                cv2.putText(preview, current_status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=1)

                # Combine the video frame and timeline bar.
                combined = np.vstack([preview, timeline_bar])
                cv2.imshow("Preview", combined)

                # Use a different waitKey delay based on the play/pause state.
                if paused:
                    # When paused, wait indefinitely until a key is pressed.
                    key = cv2.waitKey(0) & 0xFF
                else:
                    # When playing, wait a short period (1 ms) for a key press.
                    key = cv2.waitKey(1) & 0xFF

                # Process key commands.
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    return
                elif key == ord("p"):
                    # Toggle pause.
                    paused = not paused
                elif key == ord("n") and paused:
                    # Skip one frame while remaining paused.
                    pass

                frame_index += 1
                if frame_index >= video_frame_count:
                    cv2.destroyAllWindows()
                    return


if __name__ == "__main__":
    main()
