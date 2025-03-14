import time

import ffmpegio

from viva.audio.SileroVAD import SileroVAD


def main():
    audio_file = "data/florian-short.mov"

    fs, x = ffmpegio.audio.read(str(audio_file), sample_fmt="dbl", ac=1, ar=16000)
    x = x.reshape(-1)

    vad = SileroVAD(sampling_rate=fs)

    start = time.time()
    results = vad.process(x.copy())
    end = time.time()
    print(end - start)

    print(results)


if __name__ == "__main__":
    main()
