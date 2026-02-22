"""TTS module - streaming text-to-speech via edge-tts."""

import asyncio
import io
import tempfile
import os
from typing import Optional

# Legacy function for backward compatibility
def text_to_speech(api_key, region, text, output_file_path):
    from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig, SpeechSynthesisOutputFormat, ResultReason
    from azure.cognitiveservices.speech.audio import AudioOutputConfig

    speech_config = SpeechConfig(subscription=api_key, region=region)
    speech_config.set_speech_synthesis_output_format(SpeechSynthesisOutputFormat["Riff16Khz16BitMonoPcm"])
    speech_config.speech_synthesis_voice_name = "en-US-JessaNeural"
    audio_output_config = AudioOutputConfig(filename=output_file_path)
    synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)
    result = synthesizer.speak_text_async(text).get()
    if result.reason == ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
    else:
        print("Text to speech conversion successful!")


class TTS:
    """Streaming TTS using edge-tts."""

    def __init__(self, config: dict):
        self.config = config
        self.engine = config.get("tts_engine", "edge-tts")
        self.voice_id = config.get("voice_id", "en-GB-SoniaNeural")
        self.speed = config.get("tts_speed", 1.0)
        self._buffer = []

    async def initialize(self) -> None:
        """Initialize TTS engine."""
        pass

    async def stream_chunk(self, text: str) -> None:
        """Stream a text chunk to audio playback."""
        if not text or not text.strip():
            return
        self._buffer.append(text)

    async def finalize(self) -> None:
        """Synthesize buffered text and play."""
        full_text = "".join(self._buffer).strip()
        self._buffer = []
        if not full_text:
            return
        await self._synthesize_and_play(full_text)

    async def _synthesize_and_play(self, text: str) -> None:
        """Use edge-tts to synthesize and play audio."""
        import edge_tts
        voice = self.voice_id if self.voice_id != "default" else "en-GB-SoniaNeural"
        rate = f"{int((self.speed - 1) * 100)}%" if self.speed != 1.0 else "+0%"
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        if not chunks:
            return
        data = b"".join(chunks)
        await self._play_mp3(data)

    async def _play_mp3(self, data: bytes) -> None:
        """Play mp3 bytes using pydub."""
        try:
            from pydub import AudioSegment
            from pydub.playback import play
            audio = AudioSegment.from_mp3(io.BytesIO(data))
            # Run in executor - play() is blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: play(audio))
        except Exception as e:
            # Fallback: save to temp file and open with default app
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(data)
                    path = f.name
                os.startfile(path)
                await asyncio.sleep(len(data) / 16000)  # rough duration
                try:
                    os.unlink(path)
                except OSError:
                    pass
            except Exception:
                pass
