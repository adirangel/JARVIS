from pydub import AudioSegment
from pydub.playback import play

#play_audio.py
def play_audio(file_path):
    audio = AudioSegment.from_file(file_path, format="wav")
    play(audio)