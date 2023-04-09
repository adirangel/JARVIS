# main.py
import requests
import openai
import pyaudio
import wave
import audioop
import time

from text_to_speech import text_to_speech
from play_audio import play_audio
from record_audio import record_audio
from generate_response import generate_response
from audio_to_text import audio_to_text

with open('api_key.txt', 'r') as file:
    api_key = file.read().strip()

with open('azure_api.txt', 'r') as file:
    azure_speech_key = file.read().strip()
# Now you can use the api_key variable in your code.



conversation_history = []
max_turns = 15

while True:
    delay_seconds = 2
    print(f"Starting recording in {delay_seconds} seconds...")
    time.sleep(delay_seconds)

    audio_file_path = "output.wav"
    record_audio(audio_file_path)

    transcribed_text = audio_to_text(api_key, audio_file_path)
    print("Transcribed Text:", transcribed_text)

    if transcribed_text.strip() == "":
        print("No input detected, stopping.")
        break

    conversation_history.append(transcribed_text)

    response = generate_response(api_key, conversation_history, "Jarvis is an AI developed by the Angel Team.")
    print("Response:", response)

    conversation_history.append(response)

    if len(conversation_history) > max_turns:
        conversation_history.pop(0)
        conversation_history.pop(0)

    conversation_history.append(response)

    azure_speech_region = "eastus"
    output_speech_file = "response.wav"
    text_to_speech(azure_speech_key, azure_speech_region, response, output_speech_file)

    play_audio(output_speech_file)
