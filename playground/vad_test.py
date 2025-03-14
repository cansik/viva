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

    vad = SileroVAD(sampling_rate=fs)

    start = time.time()
    results = vad.process(x.copy())
    end = time.time()
    print("VAD processing time:", end - start)

    vad_segments = convert_vad_results_to_segments(results)

    # read stream info for the video
    video_streams = ffmpegio.probe.video_streams_basic(str(movie_file))
    video_info = video_streams[0]

    # extract video information
    video_duration_ms = float(video_info["duration"] * 1000)
    video_fps = float(video_info["frame_rate"])
    total_video_frames = int(video_duration_ms / video_fps)
    video_frame_length_ms = 1000 / video_fps
    video_width = int(video_info["width"])
    video_height = int(video_info["height"])

    video_frame_count = total_video_frames
    video_duration_seconds = video_duration_ms / 1000

    # convert VAD segments into per-frame speaking labels
    speaking_labels = np.full(video_frame_count, False, dtype=bool)
    for segment in vad_segments:
        # extract timestamps in seconds
        start_ts = segment.start / fs
        end_ts = segment.end / fs

        start_frame_index = round(start_ts / video_duration_seconds * video_frame_count)
        end_frame_index = round(end_ts / video_duration_seconds * video_frame_count)

        if end_frame_index > video_frame_count:
            end_frame_index = video_frame_count

        speaking_labels[start_frame_index:end_frame_index] = True

    # playback video and overlay timeline and text
    frame_index = 0
    timeline_height = 50  # height (in pixels) of the timeline bar
    with ffmpegio.open(str(movie_file), "rv", blocksize=100) as fin:
        for frames in fin:
            video_frames: np.ndarray = frames

            for frame_rgb in video_frames:
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                preview = frame.copy()

                # create the timeline bar
                timeline_bar = np.full((timeline_height, video_width, 3), 255,
                                       dtype=np.uint8)  # initialize with white (non-speaking)
                # map each column (pixel) to a video frame index
                cols = np.arange(video_width)
                timeline_frame_indices = (cols / video_width * video_frame_count).astype(int)
                timeline_frame_indices = np.clip(timeline_frame_indices, 0, video_frame_count - 1)
                # assign blue to columns corresponding to speaking frames (blue in BGR: (255, 0, 0))
                speaking_mask = speaking_labels[timeline_frame_indices]
                timeline_bar[:, speaking_mask] = (255, 0, 0)

                # compute the cursor x-position on the timeline based on the current frame index
                cursor_x = int(frame_index / video_frame_count * video_width)
                # draw a vertical red line (red in BGR: (0, 0, 255)) as a cursor on the timeline
                cv2.line(timeline_bar, (cursor_x, 0), (cursor_x, timeline_height - 1), (0, 0, 255), thickness=2)

                # add text label on the preview image (e.g., top left corner)
                current_status = "Speaking" if speaking_labels[frame_index] else "Not Speaking"
                # To ensure the text is visible on all backgrounds, we first draw black text as an outline
                cv2.putText(preview, current_status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=3)
                cv2.putText(preview, current_status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=1)

                # combine the preview and timeline bar vertically
                combined = np.vstack([preview, timeline_bar])

                cv2.imshow("Preview", combined)
                # wait for 1 ms (adjust if you want to control playback speed)
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    return

                frame_index += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
