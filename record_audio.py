import pyaudio
import wave
import audioop


# record_audio.py
def record_audio(output_filename, sample_rate=16000, channels=1, chunk_size=1024, silence_threshold=1600, silence_duration=2):
    audio_format = pyaudio.paInt16

    p = pyaudio.PyAudio()

    stream = p.open(
        format=audio_format,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    print("Recording...")

    frames = []
    silence_frames = 0
    silence_reached = False

    while not silence_reached:
        data = stream.read(chunk_size)
        rms = audioop.rms(data, 2)

        if rms > silence_threshold:
            silence_frames = 0
            frames.append(data)
        else:
            silence_frames += 1
            frames.append(data)
            if silence_frames >= (silence_duration * sample_rate) // chunk_size:
                silence_reached = True

    print("Finished recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()