import time
import argparse

from text_to_speech import text_to_speech
from play_audio import play_audio
from record_audio import record_audio
from generate_response import generate_response
from audio_to_text import audio_to_text
from handle_response import handle_response
from fetch_article import fetch_article
from bing_search import get_bing_search_results
from search_and_analyze import search_and_analyze
from database import create_connection, insert_search_data, get_search_data

with open('api_key.txt', 'r') as file:
    api_key = file.read().strip()

with open('azure_api.txt', 'r') as file:
    azure_speech_key = file.read().strip()

with open('bing_api_key.txt', 'r') as file:
    bing_api_key = file.read().strip()

azure_speech_region = "eastus"

conversation_history = []
max_turns = 15

db_file = "search_data.db"
conn = create_connection(db_file)
cache_duration = 86400

parser = argparse.ArgumentParser()
parser.add_argument('--speak', action='store_true', help='Use voice input (default mode)')
parser.add_argument('--prompt', action='store_true', help='Use text input (chat mode)')
args = parser.parse_args()

while True:
    if args.prompt:
        transcribed_text = input("Enter your text: ")
        print("Transcribed Text:", transcribed_text)
        conversation_history.append(transcribed_text)
    else:
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

    current_time = time.time()
    search_data = get_search_data(conn, transcribed_text)
    if search_data and (current_time - float(search_data[2])) < cache_duration:
        summaries = search_data[1]
    else:
        # Clear conversation history if the user repeats the question
        if len(conversation_history) >= 2 and conversation_history[-2] == transcribed_text:
            conversation_history = [transcribed_text]

        search_results, summaries = search_and_analyze(transcribed_text, bing_api_key, api_key, conversation_history)
        if search_results:
            insert_search_data(conn, transcribed_text, search_results, summaries, current_time)
        else:
            response = generate_response(api_key, conversation_history, transcribed_text)
            summaries = response

    conversation_history.append(summaries)

    if len(conversation_history) > max_turns:
        conversation_history.pop(0)
        conversation_history.pop(0)

    handled_response = handle_response(summaries)
    print("Jarvis' response:", handled_response)

    output_speech_file = "response.wav"
    text_to_speech(azure_speech_key, azure_speech_region, handled_response, output_speech_file)

    play_audio(output_speech_file)
